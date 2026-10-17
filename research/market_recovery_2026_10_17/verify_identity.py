"""Only inspect public forward archives and their frozen source hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((ROOT / "data/paper_validation/forward_150_2026_10_04_v1/protocol.json").read_text(encoding="utf-8"))
    identity = protocol["identity"]
    expected = identity["source_sha256"] if "source_sha256" in identity else identity["sources"]
    different = [name for name, value in expected.items() if digest(ROOT / name) != value]
    if different:
        raise AssertionError("Frozen source changed: " + ", ".join(different))
    archives = {}
    for name in ("forward_2026_10_03_v1", "forward_150_2026_10_04_v1"):
        folder = ROOT / "data/paper_validation" / name
        archives[name] = {str(path.relative_to(folder)): digest(path)
                          for path in sorted(folder.rglob("*")) if path.is_file()}
    report = {"frozen_source_identity_matches": True, "frozen_source_files": len(expected),
              "archive_files_sha256": archives, "database_access": False}
    destination = Path(__file__).with_name("identity_verification.json")
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved != report:
            raise AssertionError("Public forward archive/source changed during verification")
    else:
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen_source_identity_matches": True, "archive_identity_unchanged": True,
                      "database_access": False}))


if __name__ == "__main__":
    main()
