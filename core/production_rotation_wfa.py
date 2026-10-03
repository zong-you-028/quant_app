# -*- coding: utf-8 -*-
"""正式完整輪動策略的 frozen-rule walk-forward 驗證。

直接呼叫 production ``run_rotation()``，因此包含正式 K、skip、絕對動能、
外資急賣、SOX regime 與交易成本。規則與參數在所有 OOS folds 中固定，
不使用 OOS 選參；每折僅作向前推進的獨立績效觀察。
"""
from __future__ import annotations

import pandas as pd

import config
from core import evaluation as ev
from core.data_pipeline import ensure_data, load_ohlcv
from core.rotation import run_rotation


def make_oos_folds(index, train_min_days=500, test_days=120):
    """建立 expanding、互不重疊的 OOS 日期 folds。"""
    idx = pd.DatetimeIndex(index).sort_values().unique()
    folds = []
    start = int(train_min_days)
    while start < len(idx):
        stop = min(start + int(test_days), len(idx))
        folds.append({
            "train_start": idx[0], "train_end": idx[start - 1],
            "test_start": idx[start], "test_end": idx[stop - 1],
            "test_index": idx[start:stop],
        })
        start = stop
    return folds


def _metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    return {
        "cagr": ev.annualized_return(returns),
        "sharpe": ev.annualized_sharpe(returns),
        "mdd": ev.max_drawdown(equity),
        "total": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
    }


def run_production_wfa(train_min_days=500, test_days=120, verbose=True) -> dict:
    """以凍結的正式策略跑多折 expanding walk-forward OOS 驗證。"""
    production = run_rotation()
    strategy = production["full_equity"].pct_change().rename("strategy")

    benchmark_symbol = getattr(config, "BENCHMARK_SYMBOL", "006208")
    ensure_data(benchmark_symbol)
    benchmark = (load_ohlcv(benchmark_symbol)["close"].astype(float)
                 .sort_index().pct_change().rename("benchmark"))
    joined = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    if len(joined) <= train_min_days:
        raise RuntimeError(f"共同資料僅 {len(joined)} 日，不足 {train_min_days} 日暖身期")

    folds = make_oos_folds(joined.index, train_min_days, test_days)
    rows, oos_parts = [], []
    for number, fold in enumerate(folds, start=1):
        test = joined.reindex(fold["test_index"]).dropna()
        sm, bm = _metrics(test["strategy"]), _metrics(test["benchmark"])
        rows.append({
            "fold": number,
            "train_start": fold["train_start"], "train_end": fold["train_end"],
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "days": len(test),
            "strategy_cagr": sm["cagr"], "strategy_sharpe": sm["sharpe"],
            "strategy_mdd": sm["mdd"], "strategy_total": sm["total"],
            "benchmark_cagr": bm["cagr"], "benchmark_sharpe": bm["sharpe"],
            "benchmark_mdd": bm["mdd"], "benchmark_total": bm["total"],
            "beat_benchmark": sm["total"] > bm["total"],
        })
        oos_parts.append(test)

    table = pd.DataFrame(rows)
    oos = pd.concat(oos_parts).sort_index()
    strategy_oos, benchmark_oos = _metrics(oos["strategy"]), _metrics(oos["benchmark"])
    summary = {
        "folds": len(table), "oos_days": len(oos),
        "start": oos.index[0], "end": oos.index[-1],
        "strategy": strategy_oos, "benchmark": benchmark_oos,
        "winning_folds": int(table["beat_benchmark"].sum()),
        "positive_folds": int((table["strategy_total"] > 0).sum()),
    }

    if verbose:
        print("=== 正式完整策略 Frozen Walk-Forward ===")
        print(f"正式參數: mom={config.ROTATION_MOM_DAYS}, skip={config.ROTATION_SKIP_DAYS}, "
              f"K={config.ROTATION_TOP_K}, rebal={config.ROTATION_REBAL_DAYS}, "
              f"cost={config.COST_PER_TURNOVER:.4f}")
        print(f"閘門: abs={config.ROTATION_ABS_MOM}, fastsell={config.ROTATION_FASTSELL_GATE}, "
              f"SOX={config.ROTATION_SOX_GATE}/{config.ROTATION_SOX_MA}MA lag={config.ROTATION_SOX_LAG}")
        for row in rows:
            print(f"Fold {row['fold']:02d} {row['test_start'].date()}~{row['test_end'].date()} "
                  f"({row['days']:3d}d) | 策略 total {row['strategy_total']*100:+7.1f}% "
                  f"Sharpe {row['strategy_sharpe']:+5.2f} MDD {row['strategy_mdd']*100:6.1f}% | "
                  f"006208 {row['benchmark_total']*100:+7.1f}%")
        print("\n=== 合併純 OOS ===")
        print(f"{summary['start'].date()}~{summary['end'].date()}，{summary['oos_days']} 日 / "
              f"{summary['folds']} folds")
        print(f"策略: CAGR {strategy_oos['cagr']*100:+.1f}%  Sharpe {strategy_oos['sharpe']:.2f} "
              f"MDD {strategy_oos['mdd']*100:.1f}%  Total {strategy_oos['total']*100:+.1f}%")
        print(f"006208: CAGR {benchmark_oos['cagr']*100:+.1f}%  Sharpe {benchmark_oos['sharpe']:.2f} "
              f"MDD {benchmark_oos['mdd']*100:.1f}%  Total {benchmark_oos['total']*100:+.1f}%")
        print(f"正報酬 folds {summary['positive_folds']}/{summary['folds']}；"
              f"打贏 006208 folds {summary['winning_folds']}/{summary['folds']}")
    return {"production": production, "folds": table, "oos": oos, "summary": summary}


if __name__ == "__main__":
    run_production_wfa()
