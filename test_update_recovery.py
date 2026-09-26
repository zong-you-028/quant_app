import datetime as dt
from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pytest

from core import data_pipeline as dp


@pytest.fixture
def state(monkeypatch):
    state = SimpleNamespace(
        target=pd.Timestamp("2026-09-24"), dates={}, saved=[], sleeps=[],
    )
    monkeypatch.setattr(dp.config, "DATA_SOURCE", "finmind")
    monkeypatch.setattr(dp, "_last_trading_day", lambda *args: state.target)
    monkeypatch.setattr(dp, "last_ohlcv_date", state.dates.get)
    monkeypatch.setattr(dp, "_meta_get", lambda key: None)
    monkeypatch.setattr(dp, "_meta_set", lambda *args: state.saved.append(args))
    monkeypatch.setattr(dp.time, "sleep", state.sleeps.append)
    monkeypatch.setattr(dp, "_TWSE_SNAPSHOT_CACHE", None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery tests must not use the live database or network")

    monkeypatch.setattr(dp, "get_conn", forbidden)
    monkeypatch.setattr(dp.requests, "get", forbidden)
    return state


def snapshot_response():
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"Date": "1150924", "Code": symbol, "OpeningPrice": "100",
                 "HighestPrice": "102", "LowestPrice": "99",
                 "ClosingPrice": "101", "TradeVolume": "1,000"}
                for symbol in ("2330", "2317")
            ]

    return Response()


def use_fake_price_store(monkeypatch, state):
    monkeypatch.setattr(dp, "has_symbol", lambda symbol: symbol in state.dates)
    merges = []

    def merge(symbol, rows):
        merges.append(symbol)
        state.dates[symbol] = pd.to_datetime(rows["date"]).max()

    monkeypatch.setattr(dp, "_merge_twse_prices", merge)
    return merges


def test_first_snapshot_timeout_recovers_in_same_update(monkeypatch, state):
    state.dates.update(dict.fromkeys(("2330", "2317"), state.target - pd.Timedelta(days=1)))
    merges = use_fake_price_store(monkeypatch, state)
    requests = []

    def get(*args, **kwargs):
        requests.append(1)
        if len(requests) == 1:
            raise dp.requests.Timeout("temporary timeout")
        return snapshot_response()

    monkeypatch.setattr(dp.requests, "get", get)
    result = dp.update_symbols(["2330", "2317"], ignore_throttle=True)

    assert merges == ["2317", "2330"]
    assert len(requests) == 3
    assert result["updated"] == 2
    assert result["recovered"] == 1
    assert result["retried_symbols"] == ["2330"]
    assert result["failed_symbols"] == result["stale_symbols"] == []
    assert result["errors"] == {}
    assert result["asof"] == result["latest_asof"] == "2026-09-24"
    assert [key for key, _ in state.saved] == ["last_refresh", "last_refresh_target"]
    assert state.sleeps == [1.5]


def test_stale_snapshot_is_replaced_once_for_the_retry_round(monkeypatch, state):
    old = state.target - pd.Timedelta(days=1)
    state.dates.update(dict.fromkeys(("2330", "2317"), old))
    use_fake_price_store(monkeypatch, state)
    cached = pd.DataFrame([{"symbol": s, "date": "2026-09-23"} for s in state.dates])
    monkeypatch.setattr(dp, "_TWSE_SNAPSHOT_CACHE", (dt.datetime.now(), cached))
    requests = []
    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs:
                        requests.append(1) or snapshot_response())

    result = dp.update_symbols(["2330", "2317"], ignore_throttle=True)

    assert len(requests) == 1
    assert result["updated"] == result["recovered"] == 2
    assert result["stale"] == result["failed"] == 0
    assert result["data_dates"] == {"2330": "2026-09-24", "2317": "2026-09-24"}
    assert result["retried_symbols"] == ["2330", "2317"]


def test_partial_history_uses_twse_on_retry(monkeypatch, state):
    calls = []
    monkeypatch.setattr(dp, "has_symbol", lambda symbol: symbol in state.dates)

    def history(symbol):
        calls.append(("finmind", symbol))
        state.dates[symbol] = state.target - pd.Timedelta(days=1)

    def latest(symbol):
        calls.append(("twse", symbol))
        state.dates[symbol] = state.target

    monkeypatch.setattr(dp, "fetch_real_data", history)
    monkeypatch.setattr(dp, "fetch_twse_latest_data", latest)
    result = dp.update_symbols(["2330"], ignore_throttle=True)

    assert calls == [("finmind", "2330"), ("twse", "2330")]
    assert result["updated"] == result["recovered"] == 1
    assert result["failed"] == result["stale"] == 0


def test_mixed_results_are_final_unique_and_do_not_throttle(monkeypatch, state):
    old = state.target - pd.Timedelta(days=1)
    state.dates.update({"CURRENT": state.target, "OK": old, "LATE": old, "STALE": old})
    calls = Counter()
    progress = []

    def refresh(symbol):
        calls[symbol] += 1
        if symbol == "FAIL" or (symbol == "LATE" and calls[symbol] == 1):
            raise RuntimeError(f"timeout {calls[symbol]}")
        if symbol != "STALE":
            state.dates[symbol] = state.target

    monkeypatch.setattr(dp, "_refresh_market_data", refresh)
    result = dp.update_symbols(
        ["current", "OK", " late ", "STALE", "FAIL", "LATE", "", None],
        ignore_throttle=True, progress=lambda *args: progress.append(args),
    )

    assert calls == {"OK": 1, "LATE": 2, "STALE": 3, "FAIL": 3}
    assert [result[k] for k in ("updated", "current", "stale", "failed")] == [2, 1, 1, 1]
    assert result["failed_symbols"] == ["FAIL"]
    assert result["stale_symbols"] == ["STALE"]
    assert result["errors"] == {"FAIL": "timeout 3"}
    assert result["retried_symbols"] == ["LATE", "STALE", "FAIL"]
    assert result["recovered"] == 1
    assert result["asof"] is None
    assert result["data_dates"]["STALE"] == "2026-09-23"
    assert result["data_dates"]["FAIL"] is None
    assert state.saved == []
    assert state.sleeps == [1.5, 3.0]
    assert [p[0] for p in progress] == sorted(p[0] for p in progress)
    assert all(0 < done <= total == 5 for done, total, _, _ in progress)


def test_committed_data_is_verified_after_error_without_refetch(monkeypatch, state):
    state.dates["2330"] = state.target - pd.Timedelta(days=1)
    calls = []

    def refresh(symbol):
        calls.append(symbol)
        state.dates[symbol] = state.target
        raise RuntimeError("error after saving prices")

    monkeypatch.setattr(dp, "_refresh_market_data", refresh)
    result = dp.update_symbols(["2330"], ignore_throttle=True)

    assert calls == ["2330"]
    assert result["updated"] == result["recovered"] == 1
    assert result["failed"] == 0
    assert result["errors"] == {}


@pytest.mark.parametrize("source", ["finmind", "synthetic"])
def test_force_preserves_original_dates_and_does_not_refetch_success(monkeypatch, state, source):
    monkeypatch.setattr(dp.config, "DATA_SOURCE", source)
    state.dates.update({"OK": state.target, "RETRY": state.target})
    calls = Counter()

    def refresh(symbol):
        calls[symbol] += 1
        if symbol == "RETRY" and calls[symbol] == 1:
            raise RuntimeError("temporary failure")

    monkeypatch.setattr(dp, "_refresh_market_data", refresh)
    result = dp.update_symbols(["OK", "RETRY"], force=True)

    assert calls == {"OK": 1, "RETRY": 2}
    assert result["current"] == 2
    assert result["updated"] == result["failed"] == 0
    assert result["recovered"] == 1


def test_synthetic_stale_data_does_not_trigger_network_retry(monkeypatch, state):
    monkeypatch.setattr(dp.config, "DATA_SOURCE", "synthetic")
    state.dates["2330"] = state.target - pd.Timedelta(days=1)
    calls = []
    monkeypatch.setattr(dp, "_refresh_market_data", calls.append)

    result = dp.update_symbols(["2330"], ignore_throttle=True)

    assert calls == state.sleeps == result["retried_symbols"] == []
    assert result["stale"] == 1
    assert result["asof"] == "2026-09-23"
    assert state.saved == []


def test_recent_refresh_does_not_hide_pending_target(monkeypatch, state):
    state.dates.update({"OK": state.target, "RETRY": state.target - pd.Timedelta(days=1)})
    meta = {"last_refresh": dt.datetime.now().isoformat(), "last_refresh_target": "2026-09-24"}
    monkeypatch.setattr(dp, "_meta_get", meta.get)
    calls = []

    def refresh(symbol):
        calls.append(symbol)
        if len(calls) == 2:
            state.dates[symbol] = state.target

    monkeypatch.setattr(dp, "_refresh_market_data", refresh)
    result = dp.update_symbols(["OK", "RETRY"])

    assert calls == ["RETRY", "RETRY"]
    assert not result["throttled"]
    assert result["updated"] == result["current"] == result["recovered"] == 1
    assert "自動補抓完成 1 檔" in dp.format_update_status(result)
