"""Parallel prospective 150-pool signal archive, separate from frozen old50.

Initialization is explicit; scheduled runs cannot create a new experiment.
No fills or performance are produced. Only a separate public cache is opened.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data/paper_validation/market_cache_150"
ARCHIVE = ROOT / "data/paper_validation/forward_150_2026_10_04_v1"


def run(offline=False, initialize=False):
    # Import config only after setting the independent cache, in a fresh process.
    os.environ["APP_DATA_DIR"] = str(CACHE)
    os.environ["JOURNAL_DATABASE_URL"] = ""
    import config
    from core import paper_validation as pv
    if Path(config.DB_PATH).resolve() != (CACHE / "market.db").resolve():
        raise AssertionError("Use a fresh process for the independent 150 market cache")
    if not (ARCHIVE / "protocol.json").exists() and not initialize:
        raise pv.ProtocolChanged("150 protocol missing; explicit initialization required, do not start it in the schedule")
    if (ARCHIVE / "protocol.json").exists():
        if pv.verify_archive(ARCHIVE)["protocol"]["identity"] != pv.frozen_identity():
            raise pv.ProtocolChanged("150 frozen source/runtime changed; keep the archive unchanged")
    original = sqlite3.connect
    counts = {"formal_database_attempts": 0, "outside_database_attempts": 0, "offline_network_attempts": 0}
    def guarded(database, *args, **kwargs):
        path = pv._database_path(database) if str(database) != ":memory:" else None
        if path == (ROOT / "data/market.db").resolve():
            counts["formal_database_attempts"] += 1
            raise AssertionError("Formal DB forbidden")
        if path is not None and not path.is_relative_to(CACHE.resolve()):
            counts["outside_database_attempts"] += 1
            raise AssertionError("Only the independent 150 public cache allowed")
        return original(database, *args, **kwargs)
    import requests
    original_request = requests.sessions.Session.request
    def deny(*args, **kwargs):
        counts["offline_network_attempts"] += 1
        raise AssertionError("Network forbidden in offline capture")
    sqlite3.connect = sqlite3.dbapi2.connect = guarded
    if offline:
        requests.sessions.Session.request = deny
    try:
        with pv._lock(CACHE):
            if not (CACHE / "market.db").exists():
                with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as source, (CACHE / "market.db").open("xb") as target:
                    shutil.copyfileobj(source, target)
            if not (CACHE / "sox.csv").exists():
                shutil.copyfile(ROOT / "data_seed/sox.csv", CACHE / "sox.csv")
            from core import data_pipeline as dp, market_regime
            update = None
            if not offline:
                update = dp.update_symbols([*config.UNIVERSE, config.BENCHMARK_SYMBOL], time_budget_seconds=90, max_attempts=1)
                if update.get("busy"):
                    return {"status": "market_cache_busy", "isolation": counts}
                sox = market_regime.load_sox()
                if sox is None or sox.index[-1] + pd.Timedelta(days=3) < dp._last_trading_day():
                    market_regime.refresh_sox()
            result = pv.capture(ARCHIVE, CACHE / "market.db", CACHE / "sox.csv")
            event = result["event"]
            return {"status": result["status"], "experiment": str(ARCHIVE.relative_to(ROOT)),
                    "universe_size": len(config.UNIVERSE), "data_date": event["data_date"],
                    "recorded_at": event["recorded_at"], "event_hash": event["event_hash"],
                    "targets": event["targets"], "reasons": event["reasons"],
                    "current_block_reasons": result.get("current_block_reasons", []),
                    "execution_status": event["execution_status"], "fills": [], "performance": None,
                    "update": update, "isolation": counts}
    finally:
        sqlite3.connect = sqlite3.dbapi2.connect = original
        requests.sessions.Session.request = original_request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--initialize", action="store_true", help="Explicit first-time seal only, not a scheduled-run flag")
    args = parser.parse_args()
    print(json.dumps(run(args.offline, args.initialize), ensure_ascii=False, indent=2))
