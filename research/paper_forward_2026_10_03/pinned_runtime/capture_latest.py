"""Continue the authorized original 50-stock experiment using its real code.

Default is offline. --refresh explicitly permits only public market refresh,
with a 90-second request budget, into the independent old50 market cache.
Existing archived signals/protocol/snapshots are never replaced. No orders.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from . import launch
except ImportError:  # Direct CLI invocation
    import launch

HERE = Path(__file__).resolve().parent
ROOT = launch.ROOT
DEFAULT_CACHE = ROOT / "data/paper_validation/market_cache"


def assert_prefix_unchanged(before, after):
    if any(after.get(name) != digest for name, digest in before.items()):
        raise AssertionError("An existing protocol/event/snapshot changed; stop the experiment and investigate")


def run(offline=True, *, refresh=False, archive=launch.DEFAULT_ARCHIVE,
        cache=DEFAULT_CACHE, time_budget_seconds=90, timeout=300):
    if not 0 < time_budget_seconds <= 90:
        raise ValueError("Public refresh request budget must be positive and at most 90 seconds")
    if not offline and not refresh:
        raise ValueError("Network refresh needs explicit refresh=True/--refresh")
    archive, cache = Path(archive).resolve(), Path(cache).resolve()
    if cache == (ROOT / "data").resolve() or (cache / "market.db").resolve() == (ROOT / "data/market.db").resolve():
        raise ValueError("Formal app database/cache is forbidden")
    if (archive / ".archive.lock").exists() or (cache / ".archive.lock").exists():
        raise RuntimeError("Another recorder owns the archive/cache lock; no lock is deleted")
    before = launch.inventory(archive)
    environment = dict(os.environ, APP_DATA_DIR=str(cache), JOURNAL_DATABASE_URL="",
                       PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    command = [sys.executable, str(HERE / "capture_bootstrap.py"),
               "--archive", str(archive), "--cache", str(cache),
               "--host-root", str(ROOT), "--time-budget", str(time_budget_seconds)]
    if refresh:
        command.append("--refresh")
    completed = subprocess.run(command, cwd=HERE, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    after = launch.inventory(archive)
    assert_prefix_unchanged(before, after)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[-3500:])
    result = json.loads(completed.stdout)
    result["existing_archive_prefix_unchanged"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--offline", action="store_true", help="Default: no network, inspect independent cache")
    network.add_argument("--refresh", action="store_true", help="Explicit public market refresh before sealing the latest date")
    parser.add_argument("--archive", type=Path, default=launch.DEFAULT_ARCHIVE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--time-budget", type=float, default=90)
    args = parser.parse_args()
    print(json.dumps(run(refresh=args.refresh, archive=args.archive, cache=args.cache,
                         time_budget_seconds=args.time_budget), ensure_ascii=False, indent=2))
