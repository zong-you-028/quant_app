import pandas as pd
import pytest

from core.rotation import _aligned_performance


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
