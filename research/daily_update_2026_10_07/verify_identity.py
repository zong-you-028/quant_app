"""Read-only archive/source identity check without replaying historical returns."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core import paper_validation as pv

archive = ROOT / "data/paper_validation/forward_150_2026_10_04_v1"
state = pv.verify_archive(archive)
assert state["protocol"]["identity"] == pv.frozen_identity()
old = pv.verify_archive(ROOT / "data/paper_validation/forward_2026_10_03_v1")
assert old["head_hash"] == "95c6f885f329b41175e9142e4dda10e8a299bf1c8dc797e209ac9e208831c5c7"
assert state["head_hash"] == "e36eb2ff22f26840ba946e3577bc6e415ca0bafb29709bf27375c358631757b9"
report = {"status": "unchanged", "new150_identity_matches": True,
          "old50_head": old["head_hash"], "new150_head": state["head_hash"],
          "old50_events": len(old["events"]), "new150_events": len(state["events"]),
          "no_capture_or_backfill": True, "no_database_access": True,
          "app_only_new_module": "core/daily_market_update.py"}
Path(__file__).with_name("identity_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
