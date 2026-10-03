"""Verify the new initial seal without overwriting it or claiming future fills."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core import paper_validation as pv

ARCHIVE = ROOT / "data/paper_validation/forward_150_2026_10_04_v1"
OLD_ARCHIVE = ROOT / "data/paper_validation/forward_2026_10_03_v1"


if __name__ == "__main__":
    state = pv.verify_archive(ARCHIVE)
    event = state["events"][-1]
    assert len(state["events"]) == 1
    assert state["protocol"]["identity"] == pv.frozen_identity()
    replay = pv.replay_sealed(ARCHIVE)
    assert replay["status"] == "replay_targets_match"
    old_state = pv.verify_archive(OLD_ARCHIVE)
    assert len(old_state["events"]) == 1
    assert old_state["head_hash"] == "95c6f885f329b41175e9142e4dda10e8a299bf1c8dc797e209ac9e208831c5c7"
    report = {"status": "initial_seal_verified", "archive": str(ARCHIVE.relative_to(ROOT)),
              "events": 1, "protocol_hash": state["protocol"]["protocol_hash"],
              "event_hash": state["head_hash"], "snapshot_hash": event["snapshot_hash"],
              "recorded_at": event["recorded_at"], "data_date": event["data_date"],
              "universe_size": len(state["protocol"]["identity"]["parameters"]["UNIVERSE"]),
              "targets": event["targets"], "replay": replay,
              "old50_events": 1, "old50_head_unchanged": True,
              "execution_status": event["execution_status"], "fills": [], "performance": None,
              "initial_seal_is_not_new_future_performance": True}
    Path(__file__).with_name("initial_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "events", "protocol_hash", "event_hash", "snapshot_hash", "recorded_at", "old50_head_unchanged")}))
