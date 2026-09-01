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




def _alpha_X(frame: pd.DataFrame) -> pd.DataFrame:
    """固定欄位並將不可用特徵中性化，確保訓練／預測完全同形。"""
    X = frame.copy()
    for col in config.ALPHA_FEATURE_COLS:
        if col not in X:
            X[col] = 0.0
    return X[config.ALPHA_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _forward_alpha_targets(panel: dict) -> dict:
    """未來 N 日個股報酬減 006208 報酬；只作訓練標籤，不進特徵。"""
    from core.data_pipeline import load_ohlcv
    bench = load_ohlcv(config.BENCHMARK_SYMBOL)
    if bench is None or bench.empty or "close" not in bench:
        raise RuntimeError("缺少 006208 收盤資料，無法建立相對超額報酬標籤。")
    bench_close = bench["close"].astype(float).sort_index()
    bench_fwd = bench_close.shift(-config.FUTURE_N) / bench_close - 1.0
    targets = {}
    for symbol, feat in panel.items():
        close = feat["close"].astype(float)
        stock_fwd = close.shift(-config.FUTURE_N) / close - 1.0
        targets[symbol] = (stock_fwd - bench_fwd.reindex(close.index)).rename("future_alpha")
    return targets


def _stack_alpha(data: dict, targets: dict, symbols: list[str], lo, hi):
    Xs, ys = [], []
    for symbol in symbols:
        X = data[symbol]["X"]
        mask = _mask(X.index, lo, hi)
        y = targets[symbol].reindex(X.index)
        valid = mask & y.notna()
        if valid.any():
            Xs.append(_alpha_X(X.loc[valid]))
            ys.append(y.loc[valid])
    if not Xs:
        return None, None
    return pd.concat(Xs), pd.concat(ys)


def _alpha_fold_scores(model, data: dict, symbols: list[str], lo, hi):
    scores, returns = {}, {}
    for symbol in symbols:
        item = data[symbol]
        mask = _mask(item["X"].index, lo, hi)
        if mask.any():
            X = item["X"].loc[mask]
            scores[symbol] = pd.Series(model.predict(_alpha_X(X)), index=X.index)
            returns[symbol] = item["daily_ret"].reindex(X.index).fillna(0.0)
    return pd.DataFrame(scores).sort_index(), pd.DataFrame(returns).sort_index()


def run_relative_alpha_wfa(symbols=None, verbose: bool = True) -> dict:
    """候選模型：直接預測未來 20 日打贏 006208 的相對超額報酬。"""
    import lightgbm as lgb

    panel, data, _ = build_wfa_dataset(symbols or config.WATCHLIST)
    symbols = list(data)
    targets = _forward_alpha_targets(panel)
    dates = pd.DatetimeIndex(sorted(set().union(*[data[s]["X"].index for s in symbols])))
    folds = make_folds(dates, config.WFA_TRAIN_MIN_DAYS, config.WFA_TEST_DAYS,
                       config.WFA_PURGE_DAYS, config.WFA_EXPANDING, config.WFA_TRAIN_WINDOW)
    is_sharpes, oos_parts, turnover_total, used_folds = [], [], 0.0, 0
    for number, (tl, th, sl, sh) in enumerate(folds, start=1):
        tr_lo, tr_hi = dates[tl], dates[th - 1]
        te_lo, te_hi = dates[sl], dates[sh - 1]
        X_train, y_train = _stack_alpha(data, targets, symbols, tr_lo, tr_hi)
        if X_train is None or len(X_train) < config.WFA_MIN_TRAIN_SAMPLES:
            continue
        model = lgb.LGBMRegressor(objective="regression", **config.ALPHA_LGBM_PARAMS)
        model.fit(X_train, y_train)
        is_scores, is_returns = _alpha_fold_scores(model, data, symbols, tr_lo, tr_hi)
        is_net, _, _ = build_low_turnover_portfolio(is_scores, is_returns)
        if len(is_net) > 1:
            is_sharpes.append(ev.annualized_sharpe(is_net))
        oos_scores, oos_daily = _alpha_fold_scores(model, data, symbols, te_lo, te_hi)
        oos_net, _, turnover = build_low_turnover_portfolio(oos_scores, oos_daily)
        if not oos_net.empty:
            oos_parts.append(oos_net)
            turnover_total += float(turnover.sum())
            used_folds += 1
        if verbose:
            print(f"  relative-alpha fold {number}/{len(folds)}: {te_lo.date()}~{te_hi.date()}")
    if not oos_parts:
        raise RuntimeError("相對超額報酬候選模型無法產生 OOS 報酬。")
    oos_returns = pd.concat(oos_parts).sort_index()
    benchmark = ev.load_benchmark_returns(config.BENCHMARK_SYMBOL).reindex(oos_returns.index).fillna(0.0)
    fin = ev.financial_report(oos_returns, benchmark)
    rob = {"is_sharpe": float(np.mean(is_sharpes)) if is_sharpes else 0.0,
           "oos_sharpe": ev.annualized_sharpe(oos_returns),
           "turnover": turnover_total / (len(oos_returns) / ev.PERIODS_PER_YEAR)}
    rob["sharpe_decay"] = ev.sharpe_decay(rob["is_sharpe"], rob["oos_sharpe"])
    fin["turnover"] = rob["turnover"]
    return {"n_folds": used_folds, "fin": fin, "rob": rob,
            "gate": eligibility_gate(fin, rob), "oos_returns": oos_returns}
def run_relative_alpha_predictions(symbols=None, validate: bool = True) -> dict:
    """以相對超額報酬模型產生最新前 3 名；未通過 OOS 門檻就不出標的。"""
    import lightgbm as lgb

    validation = run_relative_alpha_wfa(symbols=symbols, verbose=False) if validate else None
    if validation is not None and not validation["gate"]["eligible"]:
        return {"eligible": False, "predictions": [], "holdings": [], "validation": validation}
    panel, data, _ = build_wfa_dataset(symbols or config.WATCHLIST)
    symbols = list(data)
    targets = _forward_alpha_targets(panel)
    all_dates = pd.DatetimeIndex(sorted(set().union(*[data[s]["X"].index for s in symbols])))
    X_train, y_train = _stack_alpha(data, targets, symbols, all_dates[0], all_dates[-1])
    if X_train is None or len(X_train) < config.WFA_MIN_TRAIN_SAMPLES:
        raise RuntimeError("資料不足，無法訓練相對超額報酬模型。")
    model = lgb.LGBMRegressor(objective="regression", **config.ALPHA_LGBM_PARAMS)
    model.fit(X_train, y_train)
    rows = []
    for symbol in symbols:
        feature = panel[symbol].replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
        if feature.empty:
            continue
        latest = feature.iloc[[-1]]
        score = float(model.predict(_alpha_X(latest))[0])
        if np.isfinite(score):
            rows.append({"symbol": symbol, "name": get_stock_name(symbol), "score": score,
                         "price": float(latest["close"].iloc[0]),
                         "asof": str(latest.index[-1].date())})
    predictions = sorted(rows, key=lambda row: row["score"], reverse=True)[:config.CONSERVATIVE_TOP_K]
    if not predictions:
        raise RuntimeError("最新特徵不足，無法產生相對超額報酬預測。")
    for row in predictions:
        row["weight"] = 1.0 / len(predictions)
    return {"eligible": True, "predictions": predictions,
            "holdings": [row["symbol"] for row in predictions],
            "names": {row["symbol"]: row["name"] for row in predictions},
            "validation": validation}


def format_relative_alpha_report(result: dict) -> str:
    """相對超額報酬模型的簡短研究摘要。"""
    fin, rob = result["fin"], result["rob"]
    return (f"相對超額報酬模型：{'通過' if result['gate']['eligible'] else '未通過'}\n"
            f"OOS Sharpe {rob['oos_sharpe']:.2f}｜策略 CAGR {fin['strat_cagr']*100:+.1f}%｜"
            f"006208 CAGR {fin['bench_cagr']*100:+.1f}%｜年化換手 {rob['turnover']:.1f}x")
