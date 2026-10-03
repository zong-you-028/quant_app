"""Gap backfill and chip failures must not silently create fresh signals."""
import sqlite3

import pandas as pd
import pytest

from core import data_pipeline as dp


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = str(tmp_path / "isolated-market.db")
    monkeypatch.setattr(dp.config, "DB_PATH", path)
    monkeypatch.setattr(dp.config, "DATA_SOURCE", "finmind")
    monkeypatch.setattr(dp.config, "ROTATION_FASTSELL_GATE", True)
    monkeypatch.setattr(dp, "_hydrate_market_seed", lambda conn: None)
    monkeypatch.setattr(dp, "_last_trading_day", lambda *args: pd.Timestamp("2026-10-02"))
    monkeypatch.setattr(dp, "_TWSE_SNAPSHOT_CACHE", None)
    monkeypatch.setattr(dp, "_FINMIND_BLOCKED_UNTIL", None)
    monkeypatch.setattr(dp.time, "sleep", lambda duration: None)

    def no_network(*args, **kwargs):
        raise AssertionError("Integrity tests must not access the network")

    monkeypatch.setattr(dp.requests, "get", no_network)
    dp.ensure_db()
    return path


def prices(symbol, dates):
    return pd.DataFrame([{"symbol": symbol, "date": date, "open": 100., "high": 101.,
                          "low": 99., "close": 100., "volume": 1000.} for date in dates])


def put_prices(path, symbol, dates):
    with sqlite3.connect(path) as conn:
        prices(symbol, dates).to_sql("ohlcv", conn, if_exists="append", index=False)


def put_chip(path, symbol, date):
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO chip_weekly VALUES (?,?,?,?)", (symbol, date, 50., 100.))


def chips_response(date="2026-10-02"):
    return pd.DataFrame({"date": [date], "ForeignInvestmentShares": [60.], "NumberOfSharesIssued": [100.]})


def test_multimonth_gap_backfills_every_requested_month(store, monkeypatch):
    put_prices(store, "2330", ["2026-06-30"])
    seen = []

    class Response:
        def __init__(self, month):
            self.month = month

        def raise_for_status(self):
            pass

        def json(self):
            month = int(self.month[4:6])
            return {"stat": "OK", "data": [
                [f"115/{month:02d}/01", "1000", "100000", "100", "101", "99", "100", "0", "10"],
                [f"115/{month:02d}/02", "1000", "100000", "100", "101", "99", "100", "0", "10"],
            ]}

    def get(url, **kwargs):
        seen.append(kwargs["params"]["date"])
        return Response(seen[-1])

    monkeypatch.setattr(dp.requests, "get", get)
    monkeypatch.setattr(dp, "_twse_latest_snapshot", lambda: (_ for _ in ()).throw(
        AssertionError("A one-row snapshot cannot repair this gap")))
    dp.fetch_twse_latest_data("2330")
    assert seen == ["20261001", "20260901", "20260801", "20260701"]
    stored = dp.load_ohlcv("2330")
    assert len(stored) == 9
    assert pd.Timestamp("2026-07-01") in stored.index
    assert pd.Timestamp("2026-09-02") in stored.index
    assert stored.index.max() == pd.Timestamp("2026-10-02")


def test_gap_failure_does_not_advance_latest_date(store, monkeypatch):
    put_prices(store, "2330", ["2026-09-01"])
    monkeypatch.setattr(dp, "_twse_recent_prices", lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("missing month timeout")))
    with pytest.raises(RuntimeError, match="missing month"):
        dp.fetch_twse_latest_data("2330")
    assert dp.last_ohlcv_date("2330") == pd.Timestamp("2026-09-01")


def test_missing_backfill_month_does_not_mask_gap_with_current_prices(store, monkeypatch):
    put_prices(store, "2330", ["2026-08-31"])

    class Response:
        def __init__(self, month):
            self.month = month

        def raise_for_status(self):
            pass

        def json(self):
            if self.month == "20260901":
                return {"stat": "No data", "data": []}
            return {"stat": "OK", "data": [
                ["115/10/02", "1000", "100000", "100", "101", "99", "100", "0", "10"]]}

    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs:
                        Response(kwargs["params"]["date"]))
    with pytest.raises(RuntimeError, match="2026-09.*缺少補段"):
        dp.fetch_twse_latest_data("2330")
    assert dp.last_ohlcv_date("2330") == pd.Timestamp("2026-08-31")


def test_price_write_failure_rolls_back_existing_history(store):
    put_prices(store, "2330", ["2026-09-01"])
    with sqlite3.connect(store) as conn:
        conn.execute("CREATE TRIGGER reject_new_price BEFORE INSERT ON ohlcv "
                     "WHEN NEW.date = '2026-10-02' BEGIN SELECT RAISE(ABORT, 'write failed'); END")
    before = dp.load_ohlcv("2330")
    with pytest.raises(sqlite3.IntegrityError, match="write failed"):
        dp._merge_twse_prices("2330", prices("2330", ["2026-10-02"]))
    pd.testing.assert_frame_equal(dp.load_ohlcv("2330"), before)


def test_fresh_prices_with_stale_chips_trigger_independent_refresh(store, monkeypatch):
    put_prices(store, "2330", ["2026-10-02"])
    put_chip(store, "2330", "2026-08-31")
    assert dp.needs_update("2330")
    calls = []
    monkeypatch.setattr(dp, "fetch_twse_latest_data", lambda symbol: None)
    monkeypatch.setattr(dp, "_finmind_get", lambda dataset, symbol, start:
                        calls.append((dataset, symbol, start)) or chips_response())
    result = dp.update_symbols(["2330"], ignore_throttle=True)
    assert calls == [("TaiwanStockShareholding", "2330", "2026-08-31")]
    assert result["updated"] == 1
    assert result["failed"] == result["stale"] == 0
    assert result["chip_dates"] == {"2330": "2026-10-02"}
    assert dp.load_chip_weekly("2330").index.tolist() == [pd.Timestamp("2026-08-31"), pd.Timestamp("2026-10-02")]
    assert not dp.needs_update("2330")


@pytest.mark.parametrize("response", [None, "empty", "old", "bad"])
def test_chip_failure_preserves_cache_and_remains_not_current(store, monkeypatch, response):
    put_prices(store, "2330", ["2026-10-02"])
    put_chip(store, "2330", "2026-08-31")
    before = dp.load_chip_weekly("2330")
    monkeypatch.setattr(dp, "fetch_twse_latest_data", lambda symbol: None)

    def upstream(*args):
        if response is None:
            raise RuntimeError("FinMind quota")
        if response == "empty":
            return pd.DataFrame()
        if response == "bad":
            return chips_response().assign(NumberOfSharesIssued=-1)
        return chips_response("2026-08-31").assign(ForeignInvestmentShares=50.)

    monkeypatch.setattr(dp, "_finmind_get", upstream)
    result = dp.update_symbols(["2330"], ignore_throttle=True)
    assert result["updated"] == result["current"] == 0
    assert result["failed"] + result["stale"] == 1
    assert result["stale_chip_symbols"] == ["2330"]
    assert result["chip_dates"] == {"2330": "2026-08-31"}
    pd.testing.assert_frame_equal(dp.load_chip_weekly("2330"), before)
    assert dp.needs_update("2330")


def test_full_price_refresh_chip_error_does_not_delete_old_chips(store, monkeypatch):
    put_prices(store, "2330", ["2026-09-01"])
    put_chip(store, "2330", "2026-08-31")
    before = dp.load_chip_weekly("2330")

    def upstream(dataset, *args):
        if dataset == "TaiwanStockShareholding":
            raise RuntimeError("shareholding unavailable")
        return pd.DataFrame({"date": ["2026-10-02"], "open": [100.], "max": [101.],
                             "min": [99.], "close": [100.], "Trading_Volume": [1000.]})

    monkeypatch.setattr(dp, "_finmind_get", upstream)
    dp.fetch_real_data("2330")
    assert dp.last_ohlcv_date("2330") == pd.Timestamp("2026-10-02")
    pd.testing.assert_frame_equal(dp.load_chip_weekly("2330"), before)


def test_benchmark_does_not_require_stock_shareholding(store):
    put_prices(store, dp.config.BENCHMARK_SYMBOL, ["2026-10-02"])
    assert not dp.chip_needs_update(dp.config.BENCHMARK_SYMBOL)
    assert not dp.needs_update(dp.config.BENCHMARK_SYMBOL)


def test_chip_seven_day_grace_boundary(store):
    put_prices(store, "2330", ["2026-10-02"])
    put_chip(store, "2330", "2026-09-25")
    put_prices(store, "2317", ["2026-10-02"])
    put_chip(store, "2317", "2026-09-24")
    assert not dp.chip_needs_update("2330")
    assert dp.chip_needs_update("2317")
