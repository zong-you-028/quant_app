"""Bounded updates preserve completed rows and do not multiply retries."""
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from core import data_pipeline as dp

REAL_REFRESH = dp._refresh_market_data

@pytest.fixture
def cache(monkeypatch):
    state = SimpleNamespace(now=0., dates={}, saved=[], calls=[])
    target = pd.Timestamp("2026-10-02")
    monkeypatch.setattr(dp, "_last_trading_day", lambda *args: target)
    monkeypatch.setattr(dp, "last_ohlcv_date", state.dates.get)
    monkeypatch.setattr(dp, "_chip_required", lambda s: False)
    monkeypatch.setattr(dp, "chip_needs_update", lambda *args: False)
    monkeypatch.setattr(dp, "_meta_get", lambda k: None)
    monkeypatch.setattr(dp, "_meta_set", lambda *args: state.saved.append(args))
    monkeypatch.setattr(dp.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(dp.time, "sleep", lambda t: None)
    monkeypatch.setattr(dp.config, "DATA_SOURCE", "finmind")

    def no_io(*args, **kwargs):
        raise AssertionError("must not access live storage or network")

    monkeypatch.setattr(dp, "get_conn", no_io)
    monkeypatch.setattr(dp.requests, "get", no_io)

    def refresh(symbol):
        state.calls.append(symbol)
        state.now += 5
        state.dates[symbol] = target

    monkeypatch.setattr(dp, "_refresh_market_data", refresh)
    return state


def test_budget_resume_does_not_refetch_completed_symbols(cache):
    first = dp.update_symbols(["A", "B", "C"], time_budget_seconds=6, max_attempts=1)
    assert first["updated"] == 2
    assert first["pending_symbols"] == ["C"]
    assert first["time_budget_reached"]
    assert cache.saved == []  # incomplete batches never enter six-hour throttle
    second = dp.update_symbols(["A", "B", "C"], time_budget_seconds=6, max_attempts=1)
    assert cache.calls == ["A", "B", "C"]
    assert second["current"] == 2 and second["updated"] == 1
    assert second["pending"] == 0
    assert len(cache.saved) == 2
    assert not vars(dp._UPDATE_CONTEXT)  # no deadline leakage to unrelated callers


def test_single_flight_blocks_duplicate_without_waiting(cache, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    results = []

    def refresh(s):
        entered.set()
        assert release.wait(3)
        cache.dates[s] = pd.Timestamp("2026-10-02")

    monkeypatch.setattr(dp, "_refresh_market_data", refresh)
    worker = threading.Thread(target=lambda: results.append(dp.update_symbols(["A"], time_budget_seconds=90)))
    worker.start()
    try:
        assert entered.wait(3)
        blocked = dp.update_symbols([" a ", "A", "", None], time_budget_seconds=90)
        assert blocked["busy"] and blocked["pending"] == 1
        assert "未啟動重複下載" in dp.format_update_status(blocked)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and results[0]["updated"] == 1


def test_bounded_finmind_timeout_has_one_request_per_pass(cache, monkeypatch):
    requests = []

    def get(*args, **kwargs):
        requests.append(kwargs["timeout"])
        raise dp.requests.Timeout("offline")

    monkeypatch.setattr(dp.requests, "get", get)
    monkeypatch.setattr(dp, "_FINMIND_BLOCKED_UNTIL", None)
    monkeypatch.setattr(dp, "_refresh_market_data", lambda s: dp._finmind_get("TaiwanStockPrice", s, "2026-10-01"))
    res = dp.update_symbols(["A"], time_budget_seconds=5, max_attempts=2)
    assert requests == [5., 5.]  # two batch passes, no nested three-attempt loop
    assert res["failed"] == 1 and "offline" in res["errors"]["A"]


@pytest.mark.parametrize("http_status,body_status", [(403, 403), (200, 429)])
def test_blocked_source_opens_circuit_without_retries(cache, monkeypatch, http_status, body_status):
    requests = []
    response = SimpleNamespace(status_code=http_status, reason="blocked",
                               json=lambda: {"status": body_status, "msg": "quota"},
                               raise_for_status=lambda: None)
    monkeypatch.setattr(dp.requests, "get", lambda *a, **kw: requests.append(kw) or response)
    monkeypatch.setattr(dp, "_FINMIND_BLOCKED_UNTIL", None)
    monkeypatch.setattr(dp, "_refresh_market_data", lambda s: dp._finmind_get("TaiwanStockShareholding", s, "2026-10-01"))
    res = dp.update_symbols(["A", "B"], time_budget_seconds=90, max_attempts=2)
    assert len(requests) == 1
    assert res["failed"] == 2 and res["retried_symbols"] == []


def test_chip_only_retry_skips_already_current_price(cache, monkeypatch):
    cache.dates["A"] = pd.Timestamp("2026-10-02")
    chip = []
    # Restore the orchestration function; its sub-operations stay isolated.
    monkeypatch.setattr(dp, "_refresh_market_data", REAL_REFRESH)
    monkeypatch.setattr(dp, "has_symbol", lambda s: True)
    monkeypatch.setattr(dp, "needs_update", lambda s: not chip)
    monkeypatch.setattr(dp, "chip_needs_update", lambda *args: not chip)
    monkeypatch.setattr(dp, "fetch_twse_latest_data", lambda s: pytest.fail("fresh prices must not download again"))
    monkeypatch.setattr(dp, "fetch_chip_data", chip.append)
    dp.update_symbols(["A"], time_budget_seconds=90, max_attempts=1)
    assert chip == ["A"]
