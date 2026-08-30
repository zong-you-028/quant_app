# -*- coding: utf-8 -*-
"""
eval_strategy.py - 策略評測harness:跨 WATCHLIST 比較策略 vs 買進持有。
逐檔跑完整 pipeline,彙總:
  - win_rate(贏過同期間買進持有的回合比例)
  - 策略年化(CAGR,含成本) vs 買進持有年化
  - 最大回撤(策略 vs 買進持有)
  - 全體:平均 CAGR、有幾檔「年化贏過買進持有」、平均 MDD
用於改策略前後做客觀對照(不調個股參數,避免過擬合)。
"""
import numpy as np
import pandas as pd

import config
from core.data_pipeline import ensure_data, build_features
from core.ml_strategy import label_data, prepare_xy, train_model, predict_series
from core.backtest_engine import run_backtest, _max_drawdown


def _cagr(total_return, years):
    base = 1.0 + total_return
    if years > 0 and base > 0:
        return base ** (1.0 / years) - 1.0
    return total_return


def eval_symbol(symbol: str) -> dict:
    ensure_data(symbol)
    feat = build_features(symbol)
    lab = label_data(feat)
    X, y = prepare_xy(lab)
    model, val_acc = train_model(X, y)
    sig = predict_series(model, feat)
    bt = run_backtest(sig, feat["ret"])
    years = (feat.index[-1] - feat.index[0]).days / 365.25
    bh_mdd = _max_drawdown(bt["buy_hold_equity"])
    return {
        "symbol": symbol,
        "val_acc": val_acc,
        "win_rate": bt["win_rate"],
        "cagr": _cagr(bt["total_return"], years),
        "bh_cagr": _cagr(bt["buy_hold_return"], years),
        "mdd": bt["max_drawdown"],
        "bh_mdd": bh_mdd,
        "n_changes": bt["n_changes"],
    }


def main(symbols=None):
    symbols = symbols or config.WATCHLIST
    rows = []
    for s in symbols:
        try:
            rows.append(eval_symbol(s))
        except Exception as ex:
            print(f"[skip] {s}: {ex}")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    show = df.copy()
    for c in ["win_rate", "cagr", "bh_cagr", "mdd", "bh_mdd", "val_acc"]:
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))
    print("\n=== 全體彙總 ===")
    print(f"檔數                : {len(df)}")
    print(f"平均 win_rate(贏過抱住): {df['win_rate'].mean()*100:.1f}%")
    print(f"平均 策略CAGR        : {df['cagr'].mean()*100:.1f}%")
    print(f"平均 買進持有CAGR    : {df['bh_cagr'].mean()*100:.1f}%")
    print(f"年化贏過買進持有檔數 : {(df['cagr'] > df['bh_cagr']).sum()} / {len(df)}")
    print(f"平均 策略MDD         : {df['mdd'].mean()*100:.1f}%  (買進持有 {df['bh_mdd'].mean()*100:.1f}%)")
    print(f"MDD較淺(較抗跌)檔數 : {(df['mdd'] > df['bh_mdd']).sum()} / {len(df)}")
    print(f"平均 換倉次數        : {df['n_changes'].mean():.0f}")
    return df


if __name__ == "__main__":
    main()
