"""Verify the original archive without changing its code, protocol or events."""
import json
from pathlib import Path
import shutil
import tempfile

import launch
import capture_latest


def main():
    before = launch.inventory(launch.DEFAULT_ARCHIVE)
    verify = launch.run("verify")
    replay = launch.run("replay")
    capture_copy = launch.run("capture-copy")
    continuation = capture_latest.run()
    after = launch.inventory(launch.DEFAULT_ARCHIVE)
    assert before == after
    assert verify["pinned_launcher"]["universe_size"] == 50
    assert replay["status"] == "replay_targets_match"
    assert capture_copy["status"] == "already_sealed"
    assert capture_copy["event"]["targets"] == replay["targets"]
    assert replay["performance"] is None and replay["fills"] == []
    assert continuation["status"] == "already_sealed"
    assert continuation["event_count_before"] == continuation["event_count_after"] == 1
    # A deliberately empty-event temporary clone verifies the true append path
    # without creating extra or fabricated future signals in the actual ledger.
    with tempfile.TemporaryDirectory(prefix="quant_pinned_append_proof_") as temporary:
        temporary = Path(temporary)
        clone = temporary / "synthetic_empty_event_archive"
        clone.mkdir(); (clone / "events").mkdir(); (clone / "snapshots").mkdir()
        shutil.copyfile(launch.DEFAULT_ARCHIVE / "protocol.json", clone / "protocol.json")
        for path in (launch.DEFAULT_ARCHIVE / "snapshots").glob("*.json.gz"):
            shutil.copyfile(path, clone / "snapshots" / path.name)
        cloned_append = capture_latest.run(archive=clone, cache=temporary / "independent_public_cache")
        assert cloned_append["status"] == "recorded_signal"
        assert cloned_append["event_count_before"] == 0 and cloned_append["event_count_after"] == 1
        cloned_event = json.loads((clone / "events/000001.json").read_bytes())
        assert cloned_event["snapshot_hash"] == replay["snapshot_hash"]
        assert cloned_event["targets"] == replay["targets"]
        assert cloned_event["fills"] == [] and cloned_event["performance"] is None
    report = {"reference_commit": verify["pinned_launcher"]["reference_commit"],
              "source_identity_is_unmodified": True,
              "original_archive_unchanged": before == after,
              "universe_size": verify["pinned_launcher"]["universe_size"],
              "runtime": verify["pinned_launcher"]["runtime"],
              "identity_sha256": verify["pinned_launcher"]["identity_sha256"],
              "archive_inventory": before, "verify_status": verify["status"],
              "replay_status": replay["status"], "capture_copy_status": capture_copy["status"],
              "replay_worker_isolation": replay.get("isolation"),
              "capture_copy_current_block_reasons": capture_copy.get("current_block_reasons", []),
              "capture_copy_current_snapshot_changed": capture_copy.get("current_snapshot_changed"),
              "old50_continuation_offline": continuation,
              "synthetic_empty_event_clone_append": cloned_append,
              "pinned_seed_reproduces_original_public_snapshot": True,
              "launcher_isolation": verify["pinned_launcher"]["isolation"],
              "targets": replay["targets"], "fills": [], "performance": None,
              "notes": ["Git LF/Windows CRLF is restored only when its bytes exactly match the old recorded SHA.",
                        "The original protocol is never edited and identity is computed by the actual unchanged sources.",
                        "Read-only launch capture uses a disposable copy; the separately authorized capture_latest helper only appends to old50.",
                        "The synthetic empty-event clone tests real insertion at the current clock; it is not an actual extra future observation.",
                        "Future new app code/pool changes do not alter this separate f49b4ca source tree."]}
    Path(__file__).with_name("verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
