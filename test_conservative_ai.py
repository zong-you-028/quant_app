# -*- coding: utf-8 -*-
import pandas as pd

import config
from core.conservative_ai import build_low_turnover_portfolio, eligibility_gate


def test_gate_requires_all_four_conditions():
    fin = {"strat_cagr": 0.12, "bench_cagr": 0.08}
    rob = {"oos_sharpe": 0.70, "sharpe_decay": 0.30,
           "turnover": config.CONSERVATIVE_MAX_TURNOVER}
    assert eligibility_gate(fin, rob)["eligible"]

    rob["sharpe_decay"] = 0.50
    rejected = eligibility_gate(fin, rob)
    assert not rejected["eligible"]
    assert not rejected["checks"]["sharpe_decay_within_50pct"]


def test_portfolio_only_changes_on_rebalance_days():
    idx = pd.bdate_range("2024-01-02", periods=45)
    # 第 10 日排名翻轉，但第 20 日前不可調倉；第 20 日才會切換。
    scores = pd.DataFrame({
        "A": [2.0] * 10 + [0.0] * 35,
        "B": [0.0] * 10 + [3.0] * 35,
    }, index=idx)
    returns = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    _, weights, turnover = build_low_turnover_portfolio(
        scores, returns, rebalance_days=20, top_k=1, cost=0.0)

    assert weights.loc[idx[10], "A"] == 1.0
    assert weights.loc[idx[19], "A"] == 1.0
    assert weights.loc[idx[20], "B"] == 1.0
    assert turnover.iloc[1:20].sum() == 0.0
    assert turnover.iloc[20] == 2.0


if __name__ == "__main__":
    test_gate_requires_all_four_conditions()
    test_portfolio_only_changes_on_rebalance_days()
    print("test_conservative_ai: OK")
