"""Initialize or verify the public-seed prospective archive, without real fills.

The archive belongs under ignored data/, never in the real market.db/journal.
This script writes a small public research report and verifies idempotence,
source-only replay, calendar agreement, and all-cash starting state.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core import paper_validation as pv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/paper_validation/forward_2026_10_03_v1")
    args = parser.parse_args()
    counts = {"formal_database_attempts": 0}
    original = sqlite3.connect

    def guarded(database, *positional, **keyword):
        if str(database) != ":memory:" and pv._database_path(database) == (ROOT / "data/market.db").resolve():
            counts["formal_database_attempts"] += 1
            raise AssertionError("Formal app database access forbidden")
        return original(database, *positional, **keyword)

    sqlite3.connect = sqlite3.dbapi2.connect = guarded
    try:
        first = pv.capture(args.output)
        state = pv.verify_archive(args.output)
        events_before = len(state["events"])
        second = pv.capture(args.output)
        assert second["status"] in ("already_sealed", "unchanged_blocked")
        assert len(pv.verify_archive(args.output)["events"]) == events_before
        event = state["events"][-1]
        calendars = state["protocol"]["initial_calendars"]
        assert calendars["buffered_momentum"] == calendars["production_common_calendar"]
        assert event["fills"] == [] and event["performance"] is None and event["execution_date"] is None
        replay = pv.replay_sealed(args.output) if event["kind"] == "signal" else None
        first_signal = next((item for item in state["events"] if item["kind"] == "signal"), None)
        if first_signal:
            assert first_signal["starting_state"]["cash"] == 1. and first_signal["starting_state"]["positions"] == {}
        report = {"report_kind": "prospective_signal_archive_initialization_only",
                  "archive_path": str(args.output.resolve()), "first_status": first["status"],
                  "idempotent_status": second["status"], "frozen_at": state["protocol"]["frozen_at"],
                  "recorded_at": event["recorded_at"], "data_date": event["data_date"],
                  "snapshot_hash": event["snapshot_hash"], "protocol_hash": state["protocol"]["protocol_hash"],
                  "event_hash": event["event_hash"], "event_count": events_before,
                  "common_calendar": calendars["buffered_momentum"], "runtime": state["protocol"]["identity"]["runtime"],
                  "evaluation_protocol": state["protocol"]["identity"]["evaluation_protocol"],
                  "targets": event["targets"], "blocked_reasons": event["reasons"],
                  "parent_isolation": counts, "worker_isolation": event["isolation"],
                  "replay_status": replay["status"] if replay else "blocked_no_signal_to_replay",
                  "execution_status": event["execution_status"], "fills": [], "performance": None,
                  "notes": ["All three arms start in cash; historical app holdings and return paths were discarded.",
                            "ETF quote calendar is not a complete official exchange holiday calendar.",
                            "No broker submission, no real journal reads/writes, no market refresh.",
                            "Future executable quotes, costs and corporate actions still require a separate verified fills engine."]}
        Path(__file__).with_name("initial_verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, indent=2))
    finally:
        sqlite3.connect = sqlite3.dbapi2.connect = original


if __name__ == "__main__":
    main()
