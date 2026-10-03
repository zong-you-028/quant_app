"""OTC updates use the correct exchange and do not hide missing intervals."""
import pandas as pd
import pytest
from core import data_pipeline as dp, tpex_prices


def test_otc_routing_preserves_chip_refresh(monkeypatch):
    seen = []
    monkeypatch.setattr(dp.config, "UNIVERSE_MARKETS", {"6488": "tpex"}, raising=False)
    monkeypatch.setattr(dp, "has_symbol", lambda symbol: True)
    monkeypatch.setattr(dp, "fetch_tpex_latest_data", lambda symbol: seen.append(("otc", symbol)))
    monkeypatch.setattr(dp, "fetch_twse_latest_data", lambda symbol: seen.append(("listed", symbol)))
    monkeypatch.setattr(dp, "chip_needs_update", lambda symbol: True)
    monkeypatch.setattr(dp, "fetch_chip_data", lambda symbol: seen.append(("chip", symbol)))
    dp._refresh_market_data("6488")
    dp._refresh_market_data("2330")
    assert seen == [("otc", "6488"), ("chip", "6488"), ("listed", "2330"), ("chip", "2330")]


def test_otc_gap_uses_every_missing_month(monkeypatch):
    monkeypatch.setattr(dp, "last_ohlcv_date", lambda symbol: pd.Timestamp("2026-06-30"))
    monkeypatch.setattr(dp, "_last_trading_day", lambda: pd.Timestamp("2026-10-02"))
    seen = []
    frame = pd.DataFrame({"open": [100.], "high": [101.], "low": [99.], "close": [100.], "volume": [1000.]}, index=pd.DatetimeIndex(["2026-10-02"], name="date"))
    def recent(symbol, start, end, timeout):
        seen.append((symbol, start, end))
        return frame
    monkeypatch.setattr(tpex_prices, "recent_prices", recent)
    monkeypatch.setattr(tpex_prices, "latest_prices", lambda *args: pytest.fail("A daily bar cannot repair a gap"))
    merged = []
    monkeypatch.setattr(dp, "_merge_twse_prices", lambda symbol, fresh: merged.append(fresh))
    dp.fetch_tpex_latest_data("6488")
    assert seen == [("6488", pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-02"))]
    assert merged[0].date.tolist() == ["2026-10-02"]
    assert merged[0].symbol.tolist() == ["6488"]


def test_missing_otc_never_attempts_twse_history_fallback(monkeypatch):
    monkeypatch.setattr(dp.config, "UNIVERSE_MARKETS", {"6488": "tpex"}, raising=False)
    monkeypatch.setattr(dp, "init_db", lambda: None)
    def denied(*args): raise dp.FinMindBlockedError("quota")
    monkeypatch.setattr(dp, "_finmind_get", denied)
    monkeypatch.setattr(dp, "fetch_twse_recent_data", lambda symbol: pytest.fail("OTC must not query TWSE"))
    with pytest.raises(dp.SourceUnavailableError, match="完整歷史"):
        dp.fetch_real_data("6488")


def test_missing_otc_latest_preserves_cache(monkeypatch):
    monkeypatch.setattr(dp, "last_ohlcv_date", lambda symbol: pd.Timestamp("2026-10-01"))
    monkeypatch.setattr(dp, "_last_trading_day", lambda: pd.Timestamp("2026-10-02"))
    monkeypatch.setattr(tpex_prices, "latest_prices", lambda *args: pd.DataFrame())
    monkeypatch.setattr(dp, "_merge_twse_prices", lambda *args: pytest.fail("Empty quotes cannot be merged"))
    with pytest.raises(dp.SourceUnavailableError):
        dp.fetch_tpex_latest_data("6488")


def test_unpublished_future_otc_quotes_preserve_cache(monkeypatch):
    monkeypatch.setattr(dp, "last_ohlcv_date", lambda symbol: pd.Timestamp("2026-10-01"))
    monkeypatch.setattr(dp, "_last_trading_day", lambda: pd.Timestamp("2026-10-02"))
    frame = pd.DataFrame({"open": [100.], "high": [101.], "low": [99.], "close": [100.], "volume": [1000.]}, index=pd.DatetimeIndex(["2026-10-05"], name="date"))
    monkeypatch.setattr(tpex_prices, "latest_prices", lambda *args: frame)
    monkeypatch.setattr(dp, "_merge_twse_prices", lambda *args: pytest.fail("Future quotes cannot be merged"))
    with pytest.raises(dp.SourceUnavailableError, match="已可發布日期"):
        dp.fetch_tpex_latest_data("6488")
