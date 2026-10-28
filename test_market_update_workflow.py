"""A logical update finishes many disposable chunks without live services."""
from collections import Counter
import importlib
from types import SimpleNamespace

import pytest

from core import market_update_workflow as workflow


def reply(queue, statuses, *, attempted=(), dates=None, chips=None, **extra):
    result = {"symbol_statuses": {s: statuses.get(s, "pending") for s in queue},
              "attempted_symbols": list(attempted), "data_dates": dates or {},
              "chip_dates": chips or {}, "errors": {}, "expected_asof": "2026-10-15",
              "nonretryable_symbols": [], "stale_chip_symbols": [], "on_saved_errors": {}}
    for status in workflow.STATUSES:
        matching = [s for s, value in result["symbol_statuses"].items() if value == status]
        result[status] = len(matching)
        result[status + "_symbols"] = matching
    result["time_budget_reached"] = bool(result["pending"])
    result.update(extra)
    return result


@pytest.fixture
def sources(monkeypatch):
    state = SimpleNamespace(events=[], calls=[], now=0.)
    monkeypatch.setattr(workflow.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(workflow.dmu, "restore_cache",
                        lambda: state.events.append("restore") or {"status": "restored"}, raising=False)
    monkeypatch.setattr(workflow.dmu, "storage_status",
                        lambda: {"persistent": True}, raising=False)
    monkeypatch.setattr(workflow.dmu, "refresh_sox",
                        lambda **kw: state.events.append("SOX") or
                        SimpleNamespace(empty=False, index=["2026-10-15"]))
    def forbidden(*args, **kwargs):
        raise AssertionError("Test must install a synthetic batch source")
    monkeypatch.setattr(workflow.dmu, "update_symbols", forbidden)
    return state


def test_151_symbol_workload_finishes_automatic_chunks_with_one_user_call(sources, monkeypatch):
    symbols = [str(1000 + i) for i in range(150)] + ["006208", "1000", ""]
    saved, starts, progress = [], [], []
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        assert kwargs["max_attempts"] == 1
        deadline = sources.now + kwargs["time_budget_seconds"]
        statuses, attempted, dates = {}, [], {}
        for symbol in queue:
            if sources.now + 8 > deadline:
                break
            sources.events.append(symbol)
            kwargs["on_start"](len(attempted), len(queue), symbol, "更新行情")
            sources.now += 8
            attempted.append(symbol)
            statuses[symbol] = "updated"
            dates[symbol] = "2026-10-15"
            kwargs["on_saved"](symbol)
            kwargs["progress"](len(attempted), len(queue), symbol, "updated")
        return reply(queue, statuses, attempted=attempted, dates=dates)
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(symbols, startup=True, on_saved=saved.append,
                                 on_start=lambda *args: starts.append(args),
                                 progress=lambda *args: progress.append(args))
    assert sources.events[:3] == ["restore", "SOX", "006208"]
    assert result["updated"] == result["total_symbols"] == 151
    assert result["pending"] == result["failed"] == result["stale"] == 0
    assert result["batches"] == 14 and result["automatic_continuation"]
    assert Counter(saved) == dict.fromkeys(symbols[:-2], 1)
    assert result["asof"] == result["sox_asof"] == "2026-10-15"
    assert result["elapsed_seconds"] == 151 * 8
    assert progress[-1][:2] == (151, 151)
    assert all(0 <= args[0] <= args[1] == 151 for args in progress)
    assert any("自動接續" in args[3] for args in starts)
    assert result["continuation_stop_reason"] == ""


def test_failed_front_waits_until_unattempted_work_finishes_before_one_retry(sources, monkeypatch):
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        if len(sources.calls) == 1:
            return reply(queue, {"FAIL": "failed"}, attempted=["FAIL"], errors={"FAIL": "timeout"})
        return reply(queue, dict.fromkeys(queue, "updated"), attempted=queue,
                     dates=dict.fromkeys(queue, "2026-10-15"))
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["FAIL", "A", "B"])
    assert sources.calls == [["FAIL", "A", "B"], ["A", "B"], ["FAIL"]]
    assert result["updated"] == 3 and result["failed"] == 0
    assert result["retried_symbols"] == ["FAIL"] and result["recovered"] == 1
    assert result["errors"] == {}


def test_interrupted_front_moves_behind_never_attempted_symbols(sources, monkeypatch):
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        if len(sources.calls) == 1:
            return reply(queue, {}, attempted=["PART"], dates={"PART": "2026-10-05"})
        return reply(queue, dict.fromkeys(queue, "updated"), attempted=queue,
                     dates=dict.fromkeys(queue, "2026-10-15"))
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["PART", "A", "B"])
    assert sources.calls == [["PART", "A", "B"], ["A", "B", "PART"]]
    assert result["updated"] == 3


def test_advancing_partial_backfill_can_take_more_than_two_chunks(sources, monkeypatch):
    dates = iter(["2026-06-30", "2026-07-31", "2026-08-31", "2026-10-15"])
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        date = next(dates)
        return reply(queue, {"A": "updated" if date == "2026-10-15" else "pending"},
                     attempted=["A"], dates={"A": date})
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["A"])
    assert result["batches"] == 4 and result["updated"] == 1
    assert result["pending"] == 0 and result["no_source_progress_symbols"] == []


@pytest.mark.parametrize("attempted", [[], ["A"]])
def test_identical_pending_without_source_progress_stops_after_two_batches(sources, monkeypatch, attempted):
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        return reply(queue, {}, attempted=attempted, dates={"A": "2026-10-02"})
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["A"])
    assert result["batches"] == 2 and result["pending_symbols"] == ["A"]
    assert result["updated"] == 0 and result["asof"] == "2026-10-02"
    assert result["no_source_progress_symbols"] == ["A"]
    assert "未推進" in result["continuation_stop_reason"]


def test_stuck_partial_does_not_prevent_other_symbols_or_transient_retry(sources, monkeypatch):
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        statuses = {s: ("pending" if s == "PART" else "failed" if s == "FAIL" else "updated") for s in queue}
        return reply(queue, statuses, attempted=queue, dates={s: "2026-10-02" for s in queue},
                     errors={s: "timeout" for s in queue if s == "FAIL"})
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["PART", "A", "FAIL"])
    assert sources.calls == [["PART", "A", "FAIL"], ["PART"], ["FAIL"]]
    assert (result["updated"], result["failed"], result["pending"]) == (1, 1, 1)
    assert result["errors"] == {"FAIL": "timeout"}


def test_permanent_failures_never_retry_and_transient_failures_retry_only_once(sources, monkeypatch):
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        return reply(queue, dict.fromkeys(queue, "failed"), attempted=queue,
                     nonretryable_symbols=["QUOTA"], errors=dict.fromkeys(queue, "unavailable"))
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["QUOTA", "TIMEOUT"])
    assert sources.calls == [["QUOTA", "TIMEOUT"], ["TIMEOUT"]]
    assert result["failed"] == 2 and result["retried_symbols"] == ["TIMEOUT"]
    assert result["nonretryable_symbols"] == ["QUOTA"]


def test_busy_lease_does_not_start_another_stock_batch(sources, monkeypatch):
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        return {"busy": True, "pending": len(queue)}
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["A", "B"])
    assert result["busy"] and result["batches"] == 1
    assert result["attempted_symbols"] == [] and result["pending"] == 2
    assert sources.events == ["restore", "SOX"]


def test_sox_failure_and_restore_error_are_visible_without_blocking_stock_work(sources, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(workflow.dmu, "restore_cache", unavailable)
    monkeypatch.setattr(workflow.dmu, "refresh_sox", unavailable)
    monkeypatch.setattr(workflow.dmu, "update_symbols", lambda queue, **kwargs:
                        reply(queue, dict.fromkeys(queue, "updated"), attempted=queue,
                              dates=dict.fromkeys(queue, "2026-10-15")))
    result = workflow.run_update(["A"])
    assert result["updated"] == 1
    assert result["sox_error"] == "unavailable" and result["sox_asof"] is None
    assert result["restoration"] == {"status": "failed", "error": "unavailable"}
    assert result["storage"] == {"persistent": True}


def test_hard_batch_limit_retains_partial_dates_and_clear_reason(sources, monkeypatch):
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        return reply(queue, {}, attempted=queue, dates={"A": f"2026-10-{len(sources.calls):02}"})
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["A"], max_batches=3)
    assert result["batches"] == 3 and result["pending"] == 1
    assert result["asof"] == "2026-10-03" and "上限" in result["continuation_stop_reason"]


def test_save_callback_is_forwarded_and_earlier_errors_are_retained(sources, monkeypatch):
    saved = []
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        symbol = queue[0]
        kwargs["on_saved"](symbol)
        return reply(queue, {symbol: "updated"}, attempted=[symbol], dates={symbol: "2026-10-15"},
                     on_saved_errors={symbol: "asset unavailable"} if symbol == "A" else {},
                     cloud_save_error="cloud unavailable" if symbol == "A" else "")
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["A", "B"], on_saved=saved.append)
    assert saved == ["A"] and result["updated"] == 1
    assert result["pending_symbols"] == ["B"]
    assert result["on_saved_errors"] == {"A": "asset unavailable"}
    assert result["cloud_errors"] == {"batch_1": "cloud unavailable"}


def test_configured_cloud_restore_failure_stops_before_sox_or_stock_download(sources, monkeypatch):
    monkeypatch.setattr(workflow.dmu, "restore_cache", lambda: {
        "configured": True, "durable": False, "error": "shared store unavailable"})
    def forbidden(*args, **kwargs):
        raise AssertionError("An unavailable durable cache must not trigger downloads")
    monkeypatch.setattr(workflow.dmu, "refresh_sox", forbidden)
    monkeypatch.setattr(workflow.dmu, "update_symbols", forbidden)
    result = workflow.run_update(["A", "006208"])
    assert result["pending_symbols"] == ["006208", "A"]
    assert result["batches"] == 0 and result["attempted_symbols"] == []
    assert result["cloud_errors"] == {"restore": "shared store unavailable"}
    assert result["public_message"] and result["continuation_stop_reason"]
    assert result["data_dates"] == {} and result["asof"] is None
    from core.data_pipeline import format_update_status
    assert "None" not in format_update_status(result)
    assert "目標 尚未確認" in format_update_status(result)


def test_successful_cloud_restore_proceeds_and_merges_save_errors(sources, monkeypatch):
    monkeypatch.setattr(workflow.dmu, "restore_cache", lambda: {"configured": True, "durable": True})
    def update(queue, **kwargs):
        sources.calls.append(list(queue))
        symbol = queue[0]
        return reply(queue, {symbol: "updated"}, attempted=[symbol],
                     dates={symbol: "2026-10-15"}, cloud_errors={symbol: "save unavailable"})
    monkeypatch.setattr(workflow.dmu, "update_symbols", update)
    result = workflow.run_update(["A", "B"])
    assert result["updated"] == 1 and result["pending_symbols"] == ["B"]
    assert result["cloud_errors"] == {"A": "save unavailable"}
    assert "雲端" in result["continuation_stop_reason"] and len(sources.calls) == 1
    assert sources.events == ["SOX"]


@pytest.mark.parametrize("settings", [{"max_batches": 0}, {"max_batches": True},
                                      {"time_budget_seconds": 0}, {"time_budget_seconds": 301},
                                      {"time_budget_seconds": float("inf")}])
def test_invalid_limits_fail_before_any_backend_operation(sources, settings):
    with pytest.raises(ValueError):
        workflow.run_update(["A"], **settings)
    assert sources.events == []


def test_import_has_no_storage_or_network_side_effects(sources, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not call a backend")
    for name in ("restore_cache", "storage_status", "refresh_sox", "update_symbols"):
        monkeypatch.setattr(workflow.dmu, name, forbidden, raising=False)
    importlib.reload(workflow)
    assert sources.events == []
