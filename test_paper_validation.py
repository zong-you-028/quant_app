"""Archive invariants use synthetic public caches, never the actual app DB."""
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from core import paper_validation as pv


@pytest.fixture
def clock(monkeypatch):
    now = [datetime(2026, 10, 3, 12, tzinfo=timezone.utc)]
    monkeypatch.setattr(pv, "_utc_now", lambda: now[0])
    return now


def snapshot(date="2026-10-02", price=100):
    rows = [["A", "2026-09-30", 99, 100, 98, 99, 1000],
            ["A", date, price, price, price, price, 1000]]
    return {"schema": 1, "tables": {
        "ohlcv": {"columns": list(pv.TABLES["ohlcv"]), "rows": rows},
        "chip_weekly": {"columns": list(pv.TABLES["chip_weekly"]), "rows": [["A", "2026-09-30", 30, 100]]},
        "stock_info": {"columns": list(pv.TABLES["stock_info"]), "rows": [["A", "Synthetic"]]}},
        "sox": [["2026-09-30", 100], ["2026-10-02", 101]]}


def identity():
    return {"test_only": True, "arms": ["buffered_momentum", "production_common_calendar", "006208_buy_hold"],
            "starting_state": {"cash": 1., "positions": {}, "notional_only": True}}


def calculation(date="2026-10-02", target=None, stale=False, calendar=None):
    calendar = calendar or ["2026-09-30", "2026-10-01", "2026-10-02"]
    return {"data_date": date, "calendar": calendar, "models": {
        mode: {"data_date": date, "target": {"A": .125} if target is None else target, "calendar": calendar,
               "calendar_anchor": calendar[0], "selection_date": calendar[0],
               "data_quality": {"stale": stale, "target_date": date}, "sox": {"ok": True}}
        for mode in ("buffered_momentum", "production_common_calendar")}}


def test_first_signal_is_real_timestamp_cash_and_pending_without_performance(tmp_path, clock):
    output = pv.append_latest(tmp_path, snapshot(), calculation(), identity())
    event = output["event"]
    assert output["status"] == "recorded_signal"
    assert event["recorded_at"] == clock[0].isoformat()
    assert event["starting_state"]["cash"] == 1. and event["starting_state"]["positions"] == {}
    assert event["execution_status"] == "pending_future_open_quotes"
    assert event["execution_date"] is None and event["fills"] == [] and event["performance"] is None
    assert set(event["targets"]) == set(identity()["arms"])
    assert event["targets"]["006208_buy_hold"] == {"006208": 1.}
    assert pv.verify_archive(tmp_path)["events"] == [event]


def test_same_date_does_not_replace_old_targets(tmp_path, clock):
    first = pv.append_latest(tmp_path, snapshot(), calculation(), identity())["event"]
    second = pv.append_latest(tmp_path, snapshot(), calculation(target={"B": .125}), identity())
    assert second["status"] == "already_sealed"
    assert second["event"] == first
    assert len(pv.verify_archive(tmp_path)["events"]) == 1


def test_stale_blocks_and_same_day_recovery_seals_once(tmp_path, clock):
    blocked = pv.append_latest(tmp_path, snapshot(), calculation(stale=True), identity())
    assert blocked["status"] == "recorded_blocked" and blocked["event"]["targets"] == {}
    assert pv.append_latest(tmp_path, snapshot(), calculation(stale=True), identity())["status"] == "unchanged_blocked"
    assert pv.append_latest(tmp_path, snapshot(), calculation(), identity())["status"] == "recorded_signal"
    assert len([e for e in pv.verify_archive(tmp_path)["events"] if e["kind"] == "signal"]) == 1


def test_same_day_recheck_reports_current_staleness_without_replacing_the_sealed_event(tmp_path, clock):
    initial = pv.append_latest(tmp_path, snapshot(), calculation(), identity())["event"]
    output = pv.append_latest(tmp_path, snapshot(), calculation(stale=True), identity())
    assert output["status"] == "already_sealed" and output["event"] == initial
    assert "buffered_momentum:stale_or_missing_source" in output["current_block_reasons"]
    assert not output["current_snapshot_changed"]
    assert len(pv.verify_archive(tmp_path)["events"]) == 1


@pytest.mark.parametrize("target,reason", [
    ({"A": float("nan")}, "nonfinite_or_nonpositive_target_weight"),
    ({"A": float("inf")}, "nonfinite_or_nonpositive_target_weight"),
    ({"A": -.1}, "nonfinite_or_nonpositive_target_weight"),
    ({"A": 0}, "nonfinite_or_nonpositive_target_weight"),
    ({"A": 1.1}, "target_weight_exceeds_cash"),
    ({"UNKNOWN": .125}, "unknown_target_symbol"),
    ({str(i): .1 for i in range(9)}, "more_than_8_target_stocks"),
])
def test_illegal_target_weights_fail_closed(tmp_path, clock, target, reason):
    output = pv.append_latest(tmp_path, snapshot(), calculation(target=target), identity())
    assert output["status"] == "recorded_blocked" and output["event"]["targets"] == {}
    assert "buffered_momentum:" + reason in output["event"]["reasons"]


def test_all_cash_target_is_valid(tmp_path, clock):
    output = pv.append_latest(tmp_path, snapshot(), calculation(target={}), identity())
    assert output["status"] == "recorded_signal"
    assert output["event"]["targets"]["buffered_momentum"] == {}


def test_missing_sox_gate_is_blocked(tmp_path, clock):
    value = calculation()
    value["models"]["buffered_momentum"]["sox"] = {"ok": False}
    out = pv.append_latest(tmp_path, snapshot(), value, identity())
    assert out["event"]["targets"] == {} and "buffered_momentum:SOX_gate_unavailable" in out["event"]["reasons"]


def test_backfill_and_backdated_recording_are_forbidden(tmp_path, clock):
    pv.append_latest(tmp_path, snapshot(), calculation(), identity())
    with pytest.raises(ValueError, match="backfill"):
        pv.append_latest(tmp_path, snapshot("2026-10-01"), calculation("2026-10-01"), identity())
    clock[0] = datetime(2026, 10, 3, 11, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="Clock"):
        pv.append_latest(tmp_path, snapshot(), calculation(), identity())


def test_future_or_intraday_snapshot_does_not_become_executable(tmp_path, clock):
    clock[0] = datetime(2026, 10, 2, 8, tzinfo=timezone.utc)  # Taipei16:00, before data publication guard
    out = pv.append_latest(tmp_path, snapshot(), calculation(), identity())
    assert "future_or_incomplete_data_date" in out["event"]["reasons"]
    assert out["event"]["targets"] == {}


def test_changed_code_or_parameters_requires_new_experiment(tmp_path, clock):
    pv.append_latest(tmp_path, snapshot(), calculation(), identity())
    changed = dict(identity(), different_parameter=42)
    with pytest.raises(pv.ProtocolChanged, match="new experiment"):
        pv.append_latest(tmp_path, snapshot(), calculation(), changed)
    assert len(pv.verify_archive(tmp_path)["events"]) == 1


def test_worker_caller_fingerprint_disagreement_creates_no_protocol(tmp_path, clock):
    value = calculation()
    value["worker_identity"] = {"wrong_code": True}
    with pytest.raises(pv.ProtocolChanged, match="identities"):
        pv.append_latest(tmp_path, snapshot(), value, identity())
    assert not (tmp_path / "protocol.json").exists()


@pytest.mark.parametrize("source", ["ohlcv", "chip_weekly", "sox"])
def test_old_history_revision_blocks_without_replacing_the_original(tmp_path, clock, source):
    initial = pv.append_latest(tmp_path, snapshot(), calculation(), identity())["event"]
    changed = snapshot()
    if source == "sox":
        changed["sox"][0][1] = 50
    else:
        changed["tables"][source]["rows"][0][-1] = 500
    out = pv.append_latest(tmp_path, changed, calculation(), identity())
    assert out["status"] == "recorded_blocked"
    assert out["event"]["history_revisions"][source] == 1
    assert out["event"]["targets"] == {}
    assert pv.verify_archive(tmp_path)["events"][0] == initial
    later = deepcopy(changed)
    later["tables"]["ohlcv"]["rows"].append(["A", "2026-10-05", 101, 101, 101, 101, 1000])
    clock[0] = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    out = pv.append_latest(tmp_path, later, calculation("2026-10-05"), identity())
    assert "experiment_already_blocked_by_history_revision" in out["event"]["reasons"]


def test_new_future_rows_are_append_only_and_skipped_signals_never_backfilled(tmp_path, clock):
    pv.append_latest(tmp_path, snapshot(), calculation(), identity())
    later = snapshot()
    later["tables"]["ohlcv"]["rows"].append(["A", "2026-10-07", 102, 102, 102, 102, 1000])
    later["sox"].append(["2026-10-06", 102])
    clock[0] = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
    dates = ["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"]
    out = pv.append_latest(tmp_path, later, calculation("2026-10-07", calendar=dates), identity())
    assert out["status"] == "recorded_signal" and out["event"]["unrecorded_dates"] == ["2026-10-05", "2026-10-06"]
    assert out["event"]["backfilled_signals"] == [] and out["event"]["fills"] == []
    assert out["event"]["history_revisions"] == {}


@pytest.mark.parametrize("artifact", ["protocol", "event", "snapshot"])
def test_modified_archive_is_detected(tmp_path, clock, artifact):
    event = pv.append_latest(tmp_path, snapshot(), calculation(), identity())["event"]
    if artifact == "protocol":
        path = tmp_path / "protocol.json"
        value = json.loads(path.read_bytes()); value["frozen_at"] = "2020-01-01T00:00:00+00:00"
        path.write_bytes(pv._json(value))
    elif artifact == "event":
        path = tmp_path / "events/000001.json"
        value = json.loads(path.read_bytes()); value["targets"] = {}
        path.write_bytes(pv._json(value))
    else:
        path = tmp_path / "snapshots" / (event["snapshot_hash"] + ".json.gz")
        value = json.loads(gzip.decompress(path.read_bytes())); value["sox"][0][1] = 1
        path.write_bytes(gzip.compress(pv._json(value)))
    with pytest.raises(pv.ArchiveCorrupt, match="hash"):
        pv.verify_archive(tmp_path)


def test_exclusive_lock_prevents_two_recorders(tmp_path, clock):
    (tmp_path / ".archive.lock").write_text("owner")
    with pytest.raises(RuntimeError, match="locked"):
        pv.append_latest(tmp_path, snapshot(), calculation(), identity())
    assert not (tmp_path / "protocol.json").exists()
    assert (tmp_path / ".archive.lock").read_text() == "owner"


def test_readonly_market_export_selects_no_journal_rows(tmp_path):
    path = tmp_path / "public.db"
    value = snapshot()
    with sqlite3.connect(path) as conn:
        for table, item in value["tables"].items():
            conn.execute(f'CREATE TABLE "{table}" (' + ",".join(item["columns"]) + ")")
            conn.executemany(f'INSERT INTO "{table}" VALUES (' + ",".join("?" for _ in item["columns"]) + ")", item["rows"])
        conn.execute("CREATE TABLE journal_positions (secret TEXT)")
        conn.execute("INSERT INTO journal_positions VALUES ('PRIVATE_RECORD_DO_NOT_COPY')")
    sox = tmp_path / "sox.csv"
    pd.Series({pd.Timestamp(d): v for d, v in value["sox"]}, name="close").to_csv(sox)
    saved = path.read_bytes()
    actual = pv.read_public_snapshot(path, sox)
    assert actual == value and path.read_bytes() == saved
    assert b"PRIVATE_RECORD" not in pv._json(actual) and set(actual["tables"]) == set(pv.TABLES)
    with pytest.raises(ValueError, match="formal"):
        pv.read_public_snapshot(pv.ROOT / "data/market.db", sox)


def test_sqlite_formal_guard_normalizes_windows_file_uri():
    formal = (pv.ROOT / "data/market.db").resolve()
    assert pv._database_path(formal.as_uri() + "?mode=ro") == formal


def test_replay_only_compares_frozen_targets_and_appends_nothing(tmp_path, clock, monkeypatch):
    initial = pv.append_latest(tmp_path, snapshot(), calculation(), identity())["event"]
    monkeypatch.setattr(pv, "frozen_identity", identity)
    monkeypatch.setattr(pv, "calculate_app_targets", lambda _: dict(calculation(), worker_identity=identity()))
    result = pv.replay_sealed(tmp_path)
    assert result["status"] == "replay_targets_match"
    assert result["recorded_at"] == initial["recorded_at"]
    assert result["fills"] == [] and result["performance"] is None
    assert len(pv.verify_archive(tmp_path)["events"]) == 1
    monkeypatch.setattr(pv, "calculate_app_targets", lambda _: dict(calculation(target={"A": .25}), worker_identity=identity()))
    with pytest.raises(pv.ArchiveCorrupt, match="does not reproduce"):
        pv.replay_sealed(tmp_path)


def test_real_app_entry_uses_synthetic_cache_common_calendar_and_zero_external_access(tmp_path):
    import config
    from core.data_pipeline import _last_trading_day
    end = _last_trading_day()
    index = pd.bdate_range(end=end, periods=400)
    symbols = [*config.UNIVERSE, config.BENCHMARK_SYMBOL]
    value = snapshot()
    value["tables"]["ohlcv"]["rows"] = [
        [symbol, str(date.date()), float(100 + j + i*.1), float(100 + j + i*.1),
         float(100 + j + i*.1), float(100 + j + i*.1), 1000]
        for j, symbol in enumerate(symbols) for i, date in enumerate(index)]
    value["tables"]["chip_weekly"]["rows"] = [
        [symbol, str(date.date()), float(30 + (i % 7)*.01), 100]
        for symbol in config.UNIVERSE for i, date in enumerate(index)]
    value["tables"]["stock_info"]["rows"] = [[s, "Synthetic"] for s in symbols]
    value["sox"] = [[str(date.date()), float(100 + i*.1)] for i, date in enumerate(index)]
    out = pv.calculate_app_targets(value)
    active = out["models"]["buffered_momentum"]
    baseline = out["models"]["production_common_calendar"]
    assert active["calendar"] == baseline["calendar"] == out["calendar"]
    assert active["calendar_anchor"] == baseline["calendar_anchor"]
    assert len(active["target"]) <= 8 and len(baseline["target"]) <= 8
    assert out["isolation"] == {"formal_database_attempts": 0, "external_network_attempts": 0}
    assert "net_returns" not in active and "cagr" not in active
