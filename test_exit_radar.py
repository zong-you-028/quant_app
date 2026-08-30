# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from core.exit_radar import RadarSettings, add_indicators, analyze_exit_radar, find_pivots


def synthetic(n=300, trend=0.001, seed=11):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(trend, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = close * 0.008
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close,
                         "volume": rng.integers(1000, 5000, n).astype(float)},
                        index=pd.bdate_range("2025-01-01", periods=n))


def test_indicators_and_atr_modes():
    df = synthetic()
    out = add_indicators(df)
    for col in ("sma20", "sma60", "sma240", "macd", "rsi14", "atr14", "chandelier"):
        assert np.isfinite(out[col].iloc[-1])
    stops = [analyze_exit_radar(df, RadarSettings(atr_mode=m))["atr_stop"]
             for m in ("aggressive", "balanced", "loose")]
    assert stops[0] > stops[1] > stops[2]


def test_insufficient_and_dirty_data():
    df = synthetic(80)
    df.iloc[3, df.columns.get_loc("close")] = np.inf
    result = analyze_exit_radar(df)
    assert result["available"] is False
    assert result["bars"] == 79


def test_confirmed_pivots_do_not_use_last_bars():
    df = add_indicators(synthetic())
    highs, lows = find_pivots(df, 3, 3)
    assert all(confirm_at <= len(df) - 1 for _, _, confirm_at in highs + lows)
    assert all(date <= df.index[-4] for date, _, _ in highs + lows)


def test_breakdown_reaches_orange_or_red():
    df = synthetic(trend=0.0012)
    # 最後一天放量跌破中期均線與近期結構。
    df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-2] * .93
    df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-2] * .86
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-2] * .94
    df.iloc[-1, df.columns.get_loc("low")] = df["close"].iloc[-2] * .84
    df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].iloc[-21:-1].mean() * 3
    result = analyze_exit_radar(df)
    assert result["level"] in {"orange", "red"}
    assert any(s["kind"] == "confirm" for s in result["signals"])


def test_flat_chop_does_not_duplicate_codes():
    df = synthetic(trend=0, seed=9)
    result = analyze_exit_radar(df)
    codes = [s["code"] for s in result["signals"]]
    assert len(codes) == len(set(codes))


if __name__ == "__main__":
    test_indicators_and_atr_modes()
    test_insufficient_and_dirty_data()
    test_confirmed_pivots_do_not_use_last_bars()
    test_breakdown_reaches_orange_or_red()
    test_flat_chop_does_not_duplicate_codes()
    print("exit radar tests PASSED")
