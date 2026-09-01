# -*- coding: utf-8 -*-
"""較嚴格的低複雜度 AI 選股驗證。

此模組刻意不用 Meta-model、HMM 或每折特徵搜尋：固定六個可解釋特徵、
max_depth=2 的 LightGBM，且每 20 個交易日才允許調整一次持股。結果必須通過
OOS 夏普、006208 相對績效、夏普衰減與換手率四道門檻才可標示為「可用」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from core import evaluation as ev
from core.base_model import LGBMBaseModel
from core.wfa_backtest import _mask, _stack, build_wfa_dataset, make_folds
from core.data_pipeline import get_stock_name


def eligibility_gate(fin: dict, robustness: dict) -> dict:
    """回傳逐項啟用門檻；任一項未過即不可將模型視為可交易。"""
    checks = {
        "oos_sharpe_positive": float(robustness.get("oos_sharpe", 0.0)) > 0.0,
        "beats_006208": float(fin.get("strat_cagr", 0.0)) > float(fin.get("bench_cagr", 0.0)),
        "sharpe_decay_within_50pct": float(robustness.get("sharpe_decay", np.inf)) < config.CONSERVATIVE_MAX_SHARPE_DECAY,
        "low_turnover": float(robustness.get("turnover", np.inf)) <= config.CONSERVATIVE_MAX_TURNOVER,
    }
    return {"eligible": all(checks.values()), "checks": checks}


def build_low_turnover_portfolio(scores: pd.DataFrame, daily_returns: pd.DataFrame,
                                 rebalance_days: int = None, top_k: int = None,
                                 cost: float = None) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """以固定週期、前 K 名等權建倉，回傳淨報酬、權重及每日總換手。

    分數只在再平衡日讀取；中間日完全沿用上一期權重，因此不會因每日模型雜訊換股。
    """
    rebalance_days = rebalance_days or config.CONSERVATIVE_REBALANCE_DAYS
    top_k = top_k or config.CONSERVATIVE_TOP_K
    cost = config.COST_PER_TURNOVER if cost is None else cost
    scores = scores.sort_index().replace([np.inf, -np.inf], np.nan)
    daily_returns = daily_returns.reindex(scores.index).reindex(columns=scores.columns).fillna(0.0)
    if scores.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float)

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    current = pd.Series(0.0, index=scores.columns)
    for i, date in enumerate(scores.index):
        if i % rebalance_days == 0:
            ranked = scores.loc[date].dropna().sort_values(ascending=False).head(top_k)
            current = pd.Series(0.0, index=scores.columns)
            if len(ranked):
                current.loc[ranked.index] = 1.0 / len(ranked)
        weights.loc[date] = current

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    gross = (weights.shift(1).fillna(0.0) * daily_returns).sum(axis=1)
    return gross - turnover * cost, weights, turnover


def _fold_scores(model: LGBMBaseModel, data: dict, symbols: list[str], lo, hi):
    scores, returns = {}, {}
    for symbol in symbols:
        item = data[symbol]
        mask = _mask(item["X"].index, lo, hi)
        if mask.any():
            X = item["X"].loc[mask]
            scores[symbol] = model.score(X)
            returns[symbol] = item["daily_ret"].reindex(X.index).fillna(0.0)
    return pd.DataFrame(scores).sort_index(), pd.DataFrame(returns).sort_index()


def run_conservative_wfa(symbols=None, verbose: bool = True) -> dict:
    """執行 purged expanding WFA，並強制以嚴格 OOS 門檻決定是否可用。"""
    panel, data, market_returns = build_wfa_dataset(symbols or config.WATCHLIST)
    symbols = list(data)
    dates = pd.DatetimeIndex(sorted(set().union(*[data[s]["X"].index for s in symbols])))
    folds = make_folds(dates, config.WFA_TRAIN_MIN_DAYS, config.WFA_TEST_DAYS,
                       config.WFA_PURGE_DAYS, config.WFA_EXPANDING, config.WFA_TRAIN_WINDOW)
    if not folds:
        raise RuntimeError(f"資料不足以進行保守 AI WFA（目前 {len(dates)} 個交易日）。")

    is_sharpes, oos_parts, turnover_total = [], [], 0.0
    used_folds = 0
    for number, (train_lo_i, train_hi_i, test_lo_i, test_hi_i) in enumerate(folds, start=1):
        train_lo, train_hi = dates[train_lo_i], dates[train_hi_i - 1]
        test_lo, test_hi = dates[test_lo_i], dates[test_hi_i - 1]
        X_train, y_train, _ = _stack(data, symbols, train_lo, train_hi)
        if X_train is None or len(X_train) < config.WFA_MIN_TRAIN_SAMPLES:
            continue
        model = LGBMBaseModel(
            feature_cols=config.CONSERVATIVE_FEATURE_COLS,
            params=config.CONSERVATIVE_LGBM_PARAMS,
            low_q=config.CONSERVATIVE_SIGNAL_LOW_Q,
            high_q=config.CONSERVATIVE_SIGNAL_HIGH_Q,
        ).fit(X_train, y_train)

        is_scores, is_returns = _fold_scores(model, data, symbols, train_lo, train_hi)
        is_net, _, _ = build_low_turnover_portfolio(is_scores, is_returns)
        if len(is_net) > 1:
            is_sharpes.append(ev.annualized_sharpe(is_net))

        oos_scores, oos_returns = _fold_scores(model, data, symbols, test_lo, test_hi)
        oos_net, _, turnover = build_low_turnover_portfolio(oos_scores, oos_returns)
        if not oos_net.empty:
            oos_parts.append(oos_net)
            turnover_total += float(turnover.sum())
            used_folds += 1
        if verbose:
            print(f"  conservative fold {number}/{len(folds)}: {test_lo.date()}~{test_hi.date()} ({len(X_train)} train rows)")

    if not oos_parts:
        raise RuntimeError("保守 AI 無法產生 OOS 報酬。")
    oos_returns = pd.concat(oos_parts).sort_index()
    raw_benchmark = ev.load_benchmark_returns(config.BENCHMARK_SYMBOL)
    benchmark_label = config.BENCHMARK_SYMBOL
    if raw_benchmark.empty:
        benchmark = market_returns.reindex(oos_returns.index).fillna(0.0)
        benchmark_label = "等權大盤代理"
    else:
        benchmark = raw_benchmark.reindex(oos_returns.index).fillna(0.0)
    fin = ev.financial_report(oos_returns, benchmark)
    robustness = {
        "is_sharpe": float(np.mean(is_sharpes)) if is_sharpes else 0.0,
        "oos_sharpe": ev.annualized_sharpe(oos_returns),
        "turnover": turnover_total / (len(oos_returns) / ev.PERIODS_PER_YEAR),
    }
    robustness["sharpe_decay"] = ev.sharpe_decay(robustness["is_sharpe"], robustness["oos_sharpe"])
    fin["turnover"] = robustness["turnover"]
    gate = eligibility_gate(fin, robustness)
    return {
        "n_folds": used_folds,
        "symbols": symbols,
        "benchmark": benchmark_label,
        "fin": fin,
        "rob": robustness,
        "gate": gate,
        "oos_returns": oos_returns,
    }


def run_conservative_predictions(symbols=None, validate: bool = True) -> dict:
    """用全部已知標籤訓練保守模型，對最新一根 K 棒產生前 K 名預測標的。

    validate=True 時先執行完整 purged WFA；未通過四道門檻即回傳空標的，
    不以舊模型或假資料替代。
    """
    validation = run_conservative_wfa(symbols=symbols, verbose=False) if validate else None
    if validation is not None and not validation["gate"]["eligible"]:
        return {"eligible": False, "predictions": [], "holdings": [], "validation": validation}

    panel, data, _ = build_wfa_dataset(symbols or config.WATCHLIST)
    symbols = list(data)
    X_train, y_train, _ = _stack(
        data, symbols,
        min(item["X"].index.min() for item in data.values()),
        max(item["X"].index.max() for item in data.values()),
    )
    if X_train is None or len(X_train) < config.WFA_MIN_TRAIN_SAMPLES:
        raise RuntimeError("資料不足，無法訓練新版模型產生最新預測。")
    model = LGBMBaseModel(
        feature_cols=config.CONSERVATIVE_FEATURE_COLS,
        params=config.CONSERVATIVE_LGBM_PARAMS,
        low_q=config.CONSERVATIVE_SIGNAL_LOW_Q,
        high_q=config.CONSERVATIVE_SIGNAL_HIGH_Q,
    ).fit(X_train, y_train)

    rows = []
    for symbol in symbols:
        feature = panel[symbol].replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
        if feature.empty:
            continue
        latest = feature.iloc[[-1]]
        score = float(model.score(latest).iloc[0])
        if not np.isfinite(score):
            continue
        rows.append({
            "symbol": symbol,
            "name": get_stock_name(symbol),
            "score": score,
            "price": float(latest["close"].iloc[0]),
            "asof": str(latest.index[-1].date()),
        })
    predictions = sorted(rows, key=lambda row: row["score"], reverse=True)[:config.CONSERVATIVE_TOP_K]
    if not predictions:
        raise RuntimeError("最新特徵不足，無法產生新版模型預測。")
    weight = 1.0 / len(predictions)
    for row in predictions:
        row["weight"] = weight
    return {
        "eligible": True,
        "predictions": predictions,
        "holdings": [row["symbol"] for row in predictions],
        "names": {row["symbol"]: row["name"] for row in predictions},
        "validation": validation,
    }


def format_conservative_report(result: dict) -> str:
    """供 CLI 或 UI 顯示的可解釋摘要。"""
    fin, rob, gate = result["fin"], result["rob"], result["gate"]
    status = "通過，可繼續觀察" if gate["eligible"] else "未通過，不啟用為交易建議"
    labels = {
        "oos_sharpe_positive": "OOS Sharpe 為正",
        "beats_006208": "績效優於 006208",
        "sharpe_decay_within_50pct": "Sharpe 衰減低於 50%",
        "low_turnover": "年化換手低於上限",
    }
    checks = "；".join(f"{labels[k]}：{'通過' if v else '未通過'}" for k, v in gate["checks"].items())
    return (f"保守 AI：{status}\n"
            f"OOS Sharpe {rob['oos_sharpe']:.2f}｜策略 CAGR {fin['strat_cagr']*100:+.1f}%｜"
            f"{result['benchmark']} CAGR {fin['bench_cagr']*100:+.1f}%\n"
            f"IS→OOS Sharpe 衰減 {rob['sharpe_decay']*100:.1f}%｜年化換手 {rob['turnover']:.1f}x\n{checks}")
