"""Synthetic-only storage/clock/source tests for resumable app backfills."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from core import market_backfill as bf


@pytest.fixture
def store(monkeypatch):
    state = SimpleNamespace(symbol="2330", latest=pd.Timestamp("2026-01-02"),
                            target=pd.Timestamp("2026-03-20"), meta={}, calls=[],
                            merges=[], delegated=[], chips=[], clock=0., deadline=None,
                            tpex=False, cached=True, source_overrides={})
    dp = bf.dp
    monkeypatch.setattr(dp, "has_symbol", lambda s: state.cached)
    monkeypatch.setattr(dp, "last_ohlcv_date", lambda s: state.latest)
    monkeypatch.setattr(dp, "_last_trading_day", lambda: state.target)
    monkeypatch.setattr(dp, "_is_tpex", lambda s: state.tpex)
    monkeypatch.setattr(dp, "_chip_required", lambda s: False)
    monkeypatch.setattr(dp, "chip_needs_update", lambda s: True)
    monkeypatch.setattr(dp, "fetch_chip_data", state.chips.append)
    monkeypatch.setattr(dp, "_update_phase", lambda *a: None)
    monkeypatch.setattr(dp, "_meta_get", state.meta.get)
    monkeypatch.setattr(dp, "_meta_set", lambda k, v: state.meta.__setitem__(k, v))
    monkeypatch.setattr(bf, "ORIGINAL_REFRESH", state.delegated.append)

    def forbidden(*args, **kwargs):
        raise AssertionError("backfill tests must not access real DB or HTTP")

    monkeypatch.setattr(dp, "get_conn", forbidden)
    monkeypatch.setattr(dp.requests, "get", forbidden)

    def timeout():
        remaining = 1000. if state.deadline is None else state.deadline - state.clock
        if remaining <= 0:
            raise dp.UpdateBudgetExceeded("synthetic batch deadline")
        return min(8., remaining)

    monkeypatch.setattr(dp, "_request_timeout", timeout)

    def response(route, symbol, start, end):
        month = pd.Timestamp(end).to_period("M")
        state.calls.append((route, str(month), pd.Timestamp(start), pd.Timestamp(end)))
        allowance = timeout()
        state.clock += min(8., allowance)
        if allowance < 8.:
            raise dp.UpdateBudgetExceeded("synthetic monthly response exceeded batch")
        if str(month) in state.source_overrides:
            value = state.source_overrides[str(month)]
            if isinstance(value, Exception):
                raise value
            return value.copy()
        dates = pd.bdate_range(start, end)
        return pd.DataFrame({"symbol": symbol, "date": dates, "open": 100., "high": 102.,
                             "low": 99., "close": 101., "volume": 1000.})

    monkeypatch.setattr(dp, "_twse_recent_prices", lambda s, asof, start: response("twse", s, start, asof))
    from core import tpex_prices

    def otc(symbol, start, asof, timeout):
        return response("tpex", symbol, start, asof).drop(columns="symbol").set_index("date")

    monkeypatch.setattr(tpex_prices, "recent_prices", otc)

    def merge(symbol, rows):
        state.merges.append(rows.copy())
        state.latest = max(state.latest, pd.to_datetime(rows["date"]).max())

    monkeypatch.setattr(dp, "_merge_twse_prices", merge)
    return state


def test_budgeted_months_resume_after_committed_chunks(store):
    persisted = []

    def checkpoint(symbol):
        value = json.loads(store.meta[bf.PREFIX + symbol])
        assert value["stored_last"] == store.latest.strftime("%Y-%m-%d")
        persisted.append(value["completed_through"])

    for _ in range(2):
        store.deadline = store.clock + 10.
        with pytest.raises(bf.dp.UpdateBudgetExceeded):
            bf.refresh_symbol("2330", checkpoint=checkpoint)
    store.deadline = store.clock + 10.
    bf.refresh_symbol("2330", checkpoint=checkpoint)
    assert store.latest == store.target
    assert persisted == ["2026-01-31", "2026-02-28", "2026-03-20"]
    assert [str(pd.to_datetime(f.date).min().to_period("M")) for f in store.merges] == ["2026-01", "2026-02", "2026-03"]
    assert [c[1] for c in store.calls].count("2026-01") == 1
    assert [c[1] for c in store.calls] == ["2026-01", "2026-02", "2026-02", "2026-03", "2026-03"]


@pytest.mark.parametrize("tpex", [False, True])
def test_exchange_routing_and_no_duplicate_price_refresh(store, monkeypatch, tpex):
    store.tpex = tpex
    monkeypatch.setattr(bf.dp, "_chip_required", lambda s: True)
    bf.refresh_symbol("2330")
    assert {call[0] for call in store.calls} == ({"tpex"} if tpex else {"twse"})
    assert store.chips == ["2330"] and store.delegated == []
    assert store.latest == store.target


def test_failed_middle_month_never_advances_to_later_month(store):
    store.source_overrides["2026-02"] = bf.dp.SourceUnavailableError("missing month")
    with pytest.raises(bf.dp.SourceUnavailableError, match="missing month"):
        bf.refresh_symbol("2330")
    assert store.latest == pd.Timestamp("2026-01-30")
    assert [c[1] for c in store.calls] == ["2026-01", "2026-02"]
    assert json.loads(store.meta[bf.PREFIX + "2330"])["completed_through"] == "2026-01-31"


@pytest.mark.parametrize("poison", ["empty", "negative", "wrong_symbol", "future", "duplicate", "nan"])
def test_invalid_month_preserves_committed_rows(store, poison):
    frame = pd.DataFrame({"symbol": ["2330"], "date": [pd.Timestamp("2026-02-02")],
                          "open": [100.], "high": [102.], "low": [99.], "close": [101.], "volume": [1000.]})
    if poison == "empty":
        frame = frame.iloc[:0]
    elif poison == "negative":
        frame.loc[0, "close"] = -1.
    elif poison == "wrong_symbol":
        frame.loc[0, "symbol"] = "2317"
    elif poison == "future":
        frame.loc[0, "date"] = pd.Timestamp("2026-03-02")
    elif poison == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif poison == "nan":
        frame.loc[0, "open"] = float("nan")
    store.source_overrides["2026-02"] = frame
    with pytest.raises(bf.dp.SourceUnavailableError):
        bf.refresh_symbol("2330")
    assert store.latest == pd.Timestamp("2026-01-30")
    assert len(store.merges) == 1 and not any(c[1] == "2026-03" for c in store.calls)


def test_no_progress_merge_is_explicit_failure(store, monkeypatch):
    monkeypatch.setattr(bf.dp, "_merge_twse_prices", lambda *a: None)
    with pytest.raises(bf.dp.SourceUnavailableError, match="未保存前進"):
        bf.refresh_symbol("2330")
    assert store.latest == pd.Timestamp("2026-01-02") and not store.meta


def test_partial_current_month_is_saved_once_without_false_completion(store):
    frame = pd.DataFrame({"symbol": ["2330"], "date": [pd.Timestamp("2026-03-02")],
                          "open": [100.], "high": [102.], "low": [99.], "close": [101.], "volume": [1000.]})
    store.source_overrides["2026-03"] = frame
    bf.refresh_symbol("2330")
    assert store.latest == pd.Timestamp("2026-03-02")
    assert [c[1] for c in store.calls].count("2026-03") == 1
    assert json.loads(store.meta[bf.PREFIX + "2330"])["completed_through"] == "2026-03-02"
    assert not store.chips


def test_holiday_tail_of_old_month_can_finish_without_inventing_a_quote(store):
    store.latest = pd.Timestamp("2026-01-28")
    frame = pd.DataFrame({"symbol": ["2330"], "date": [store.latest],
                          "open": [100.], "high": [102.], "low": [99.], "close": [101.], "volume": [1000.]})
    store.source_overrides["2026-01"] = frame
    bf.refresh_symbol("2330")
    assert store.latest == store.target
    assert all(pd.to_datetime(chunk.date).min().month != 1 for chunk in store.merges)
    assert [c[1] for c in store.calls] == ["2026-01", "2026-02", "2026-03"]


def test_unanchored_cursor_cannot_skip_missing_month(store):
    store.meta[bf.PREFIX + "2330"] = json.dumps({"version": 1, "symbol": "2330",
                                                  "completed_through": "2026-02-28", "stored_last": "2026-01-01"})
    bf.refresh_symbol("2330")
    assert store.calls[0][1] == "2026-01"


@pytest.mark.parametrize("through", ["2026-02-28", "2026-01-20"])
def test_anchored_cursor_cannot_claim_later_month_or_unfinished_month_tail(store, through):
    store.meta[bf.PREFIX + "2330"] = json.dumps({"version": 1, "symbol": "2330",
                                                  "completed_through": through, "stored_last": "2026-01-02"})
    bf.refresh_symbol("2330")
    assert store.calls[0][1] == "2026-01"


def test_external_checkpoint_failure_keeps_local_chunk_and_resume_cursor(store):
    def fail(symbol):
        raise RuntimeError("durable storage offline")

    with pytest.raises(RuntimeError, match="durable storage"):
        bf.refresh_symbol("2330", checkpoint=fail)
    assert store.latest == pd.Timestamp("2026-01-30")
    assert json.loads(store.meta[bf.PREFIX + "2330"])["completed_through"] == "2026-01-31"
    bf.refresh_symbol("2330")
    assert [c[1] for c in store.calls].count("2026-01") == 1


@pytest.mark.parametrize("case", ["uncached", "fresh", "single_month"])
def test_ordinary_refresh_delegates_to_captured_original(store, monkeypatch, case):
    if case == "uncached":
        store.cached = False
    elif case == "fresh":
        store.latest = store.target
    else:
        store.latest = pd.Timestamp("2026-03-02")
    monkeypatch.setattr(bf.dp, "_refresh_market_data", bf.refresh_symbol)
    bf.dp._refresh_market_data("2330")
    assert store.delegated == ["2330"] and not store.calls
