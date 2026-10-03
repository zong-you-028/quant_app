"""Pinned historic sources retain the real 50-stock identity without mutations."""
import hashlib
import json

import pytest

from research.paper_forward_2026_10_03.pinned_runtime import bootstrap, export_runtime, launch, capture_latest


def manifest():
    return json.loads((launch.HERE / "manifest.json").read_bytes())


def test_exported_sources_match_recorded_byte_hashes_and_keep_line_endings():
    value = manifest()
    assert value["reference_commit"] == export_runtime.COMMIT
    assert value["frozen_universe_size"] == 50
    assert value["permissions"]["original_archive_writes"] is False
    for name, item in value["source_files"].items():
        content = (launch.HERE / "source" / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
        if item["checkout_conversion"] == "restore_recorded_windows_crlf":
            assert b"\r\n" in content
    assert "source/** -text" in (launch.HERE / ".gitattributes").read_text()


def test_original_public_seed_is_independent_and_hash_checked():
    value = json.loads((launch.HERE / "public_seed_manifest.json").read_bytes())
    assert value["reference_commit"] == export_runtime.COMMIT
    for name, item in value["public_seed_files"].items():
        assert hashlib.sha256((launch.HERE / "source" / name).read_bytes()).hexdigest() == item["sha256"]


def test_modified_pinned_source_fails_before_loading_model(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for name in manifest()["source_files"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((launch.HERE / "source" / name).read_bytes())
    (root / "config.py").write_bytes((root / "config.py").read_bytes() + b"\n# modified\n")
    monkeypatch.setattr(bootstrap, "SOURCE", root)
    with pytest.raises(ValueError, match="Pinned source was modified"):
        bootstrap.source_verify(manifest())


def test_launcher_has_no_real_append_mode():
    with pytest.raises(ValueError, match="disposable"):
        launch.run("capture")


def test_continuation_requires_explicit_refresh_and_refuses_formal_cache():
    with pytest.raises(ValueError, match="explicit"):
        capture_latest.run(offline=False)
    with pytest.raises(ValueError, match="Formal"):
        capture_latest.run(cache=launch.ROOT / "data")
    with pytest.raises(ValueError, match="90 seconds"):
        capture_latest.run(time_budget_seconds=91)


def test_append_prefix_check_accepts_only_new_records():
    original = {"protocol.json": "P", "events/000001.json": "E1", "snapshots/first.json.gz": "S1"}
    newer = dict(original, **{"events/000002.json": "E2", "snapshots/second.json.gz": "S2"})
    capture_latest.assert_prefix_unchanged(original, newer)
    for name in original:
        corrupt = dict(newer, **{name: "MODIFIED"})
        with pytest.raises(AssertionError, match="existing"):
            capture_latest.assert_prefix_unchanged(original, corrupt)


def test_formal_market_database_is_rejected_before_any_sql():
    with pytest.raises(ValueError, match="actual app database"):
        launch.run("capture-copy", market_db=launch.ROOT / "data/market.db")


def test_archive_inventory_reads_only_public_sealed_artifacts(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "events").mkdir(); (tmp_path / "events/000001.json").write_text("{}")
    (tmp_path / "snapshots").mkdir(); (tmp_path / "snapshots/public.json.gz").write_bytes(b"PUBLIC")
    (tmp_path / "journal.db").write_bytes(b"PRIVATE_SYNTHETIC_NOT_TO_READ")
    assert set(launch.inventory(tmp_path)) == {"protocol.json", "events/000001.json", "snapshots/public.json.gz"}


@pytest.mark.skipif(not (launch.DEFAULT_ARCHIVE / "protocol.json").exists(), reason="Local original paper archive not present")
def test_actual_identity_verify_ignores_host_import_paths_and_journal_credentials(tmp_path, monkeypatch):
    (tmp_path / "config.py").write_text("raise AssertionError('host config must not load')")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("APP_DATA_DIR", str(launch.ROOT / "data"))
    monkeypatch.setenv("JOURNAL_DATABASE_URL", "postgresql://synthetic-never-connect.invalid/test")
    before = launch.inventory(launch.DEFAULT_ARCHIVE)
    result = launch.run("verify")
    assert result["status"] == "pinned_archive_verified"
    assert result["pinned_launcher"]["universe_size"] == 50
    assert result["pinned_launcher"]["identity_sha256"] == manifest()["identity_sha256"]
    assert result["pinned_launcher"]["isolation"] == {
        "formal_database_attempts": 0, "external_network_attempts": 0, "journal_sql_attempts": 0}
    assert result["pinned_launcher"]["original_archive_unchanged"]
    assert launch.inventory(launch.DEFAULT_ARCHIVE) == before


@pytest.mark.skipif(not (launch.DEFAULT_ARCHIVE / "protocol.json").exists(), reason="Local original paper archive not present")
def test_export_is_idempotent_and_changed_existing_copy_is_never_overwritten(tmp_path):
    expected = export_runtime.export(output=tmp_path)
    assert export_runtime.export(output=tmp_path) == expected
    copied = tmp_path / "source/config.py"
    copied.write_bytes(copied.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        export_runtime.export(output=tmp_path)
