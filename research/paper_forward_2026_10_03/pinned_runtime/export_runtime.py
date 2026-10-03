"""Export the genuine f49b4ca sources matching the original 50-stock protocol.

Git stores some files with LF while the recorded Windows source was CRLF.
Only that documented checkout conversion is allowed, and the actual bytes must
match every archived SHA256. Neither the protocol nor model configuration is
edited. Existing exported files can never be overwritten with changed content.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMIT = "f49b4cae3c0f5c55acfd609aae5ce7abf6e672a0"
DEFAULT_ARCHIVE = ROOT / "data/paper_validation/forward_2026_10_03_v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(content):
    return hashlib.sha256(content).hexdigest()


def preserve(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to overwrite an existing pinned artifact: " + str(path))
    else:
        with path.open("xb") as file:
            file.write(content)


def export(archive=DEFAULT_ARCHIVE, output=HERE):
    archive, output = Path(archive), Path(output)
    protocol = json.loads((archive / "protocol.json").read_bytes())
    body = dict(protocol)
    expected_protocol = body.pop("protocol_hash")
    if sha(canonical(body)) != expected_protocol:
        raise ValueError("Original protocol hash mismatch")
    reference = json.loads((ROOT / "research/paper_forward_2026_10_03/initial_verification.json").read_bytes())
    if expected_protocol != reference["protocol_hash"]:
        raise ValueError("This exporter is pinned specifically to the original recorded experiment")
    files = {}
    names = [*protocol["identity"]["source_sha256"], "core/__init__.py"]
    for name in sorted(names):
        blob = subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(),
                                        "show", COMMIT + ":" + name], cwd=ROOT)
        expected = protocol["identity"]["source_sha256"].get(name, sha(blob))
        if sha(blob) == expected:
            content, conversion = blob, "git_blob_bytes"
        else:
            content = blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            conversion = "restore_recorded_windows_crlf"
            if sha(content) != expected:
                raise ValueError("Commit cannot reproduce the exact original source bytes: " + name)
        preserve(output / "source" / name, content)
        files[name] = {"sha256": expected, "git_blob_sha256": sha(blob),
                       "checkout_conversion": conversion, "bytes": len(content)}
    manifest = {"schema": 1, "reference_commit": COMMIT,
                "original_protocol_hash": expected_protocol,
                "identity_sha256": sha(canonical(protocol["identity"])),
                "frozen_runtime": protocol["identity"]["runtime"],
                "frozen_universe_size": len(protocol["identity"]["parameters"]["UNIVERSE"]),
                "source_files": files,
                "permissions": {"network": False, "real_journal": False,
                                "original_archive_writes": False},
                "note": "Exact historic source bytes and runtime identity, not a modified/forged protocol."}
    preserve(output / "manifest.json", canonical(manifest) + b"\n")
    preserve(output / ".gitattributes", b"source/** -text\nmanifest.json text eol=lf\n")
    return manifest


def export_seed(output=HERE):
    """Keep original public Git seed independent of the future 150-stock seed."""
    output = Path(output)
    files = {}
    for name in ("data_seed/market.db.gz", "data_seed/sox.csv"):
        blob = subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(),
                                        "show", COMMIT + ":" + name], cwd=ROOT)
        preserve(output / "source" / name, blob)
        files[name] = {"sha256": sha(blob), "bytes": len(blob)}
    manifest = {"schema": 1, "reference_commit": COMMIT, "public_seed_files": files,
                "tables_allowed": ["ohlcv", "chip_weekly", "stock_info"],
                "note": "Public Git seed only. Default replay/capture-copy still prefers the experiment's actual sealed public snapshot."}
    preserve(output / "public_seed_manifest.json", canonical(manifest) + b"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    print(json.dumps({"source": export(args.archive), "public_seed": export_seed()}, ensure_ascii=False, indent=2))
