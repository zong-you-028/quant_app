"""Read-only verifier/replayer for the original 50-stock paper experiment.

Offline capture always uses a disposable copy and is never appended to the
actual archive. Future continuation with new public market data needs a
separately authorized workflow; this launcher has no real-append option.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_ARCHIVE = ROOT / "data/paper_validation/forward_2026_10_03_v1"


def inventory(directory):
    directory = Path(directory)
    paths = [directory / "protocol.json", *(directory / "events").glob("*.json"),
             *(directory / "snapshots").glob("*.json.gz")]
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def run(mode="verify", archive=DEFAULT_ARCHIVE, market_db=None, sox_csv=None, timeout=240):
    if mode not in ("verify", "replay", "capture-copy"):
        raise ValueError("Only verify, replay and disposable capture-copy are allowed")
    archive = Path(archive).resolve()
    if (archive / ".archive.lock").exists():
        raise RuntimeError("Original experiment is being recorded; try when its writer is idle")
    if market_db and Path(market_db).resolve() == (ROOT / "data/market.db").resolve():
        raise ValueError("Use a separate market-only cache, never the actual app database")
    if market_db and mode != "capture-copy":
        raise ValueError("An input market cache is supported only for disposable capture-copy")
    before = inventory(archive)
    with tempfile.TemporaryDirectory(prefix="quant_pinned_paper_") as temporary:
        temporary = Path(temporary)
        target = archive
        if mode == "capture-copy":
            target = temporary / "archive_copy"
            for relative in before:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(archive / relative, destination)
            if inventory(target) != before:
                raise ValueError("Archive changed while being copied; no capture was started")
        environment = dict(os.environ, APP_DATA_DIR=str(temporary / "isolated_cache"),
                           JOURNAL_DATABASE_URL="", PYTHONDONTWRITEBYTECODE="1")
        environment.pop("PYTHONPATH", None)
        environment.pop("FINMIND_TOKEN", None)
        if mode == "capture-copy":
            environment["PINNED_CAPTURE_IS_DISPOSABLE_COPY"] = "1"
        command = [sys.executable, str(HERE / "bootstrap.py"), mode,
                   "--archive", str(target), "--host-root", str(ROOT)]
        if market_db:
            command.extend(["--market-db", str(Path(market_db).resolve())])
        if sox_csv:
            command.extend(["--sox-csv", str(Path(sox_csv).resolve())])
        completed = subprocess.run(command, cwd=HERE, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        after = inventory(archive)
        if after != before:
            raise AssertionError("Original archive changed during verification; investigate its separate writer")
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[-3000:])
        result = json.loads(completed.stdout)
        result["pinned_launcher"]["original_archive_unchanged"] = True
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "replay", "capture-copy"), nargs="?", default="verify")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--market-db", type=Path)
    parser.add_argument("--sox-csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.archive, args.market_db, args.sox_csv), ensure_ascii=False, indent=2))
