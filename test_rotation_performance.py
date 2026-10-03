import pandas as pd
import pytest

from core.rotation import _aligned_performance
from core.production_rotation_wfa import make_oos_folds


def test_aligned_performance_uses_common_dates_and_lookback():
    idx = pd.to_datetime(["2024-01-02", "2024-12-31", "2025-01-02", "2025-06-30"])
    strategy = pd.Series([0.50, 0.10, 0.20, 0.05], index=idx)
    benchmark = pd.Series([0.05, 0.10, 0.00], index=idx[1:])

    result = _aligned_performance(strategy, benchmark, lookback_days=365)

    assert result["start"] == pd.Timestamp("2024-12-31")
    assert result["end"] == pd.Timestamp("2025-06-30")
    assert result["equity"].iloc[-1] == pytest.approx(1.10 * 1.20 * 1.05)
    assert result["benchmark_equity"].iloc[-1] == pytest.approx(1.05 * 1.10)
    assert result["equity"].index.equals(result["benchmark_equity"].index)


def test_production_wfa_folds_are_forward_and_non_overlapping():
    idx = pd.bdate_range("2020-01-01", periods=23)
    folds = make_oos_folds(idx, train_min_days=10, test_days=5)
    assert [len(f["test_index"]) for f in folds] == [5, 5, 3]
    assert folds[0]["train_end"] < folds[0]["test_start"]
    assert folds[0]["test_end"] < folds[1]["test_start"]
    assert folds[-1]["test_end"] == idx[-1]
