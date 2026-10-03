"""Refresh an independent PUBLIC market cache, then seal today's target.

No actual app database, journal, brokerage or deployment is used. Default
archive is the existing frozen experiment. Repeat runs never replace signals.
Use --offline to verify an existing public cache without HTTP refresh.
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


def _run_locked(offline=False):
    cache = ROOT / "data/paper_validation/market_cache"
    archive = ROOT / "data/paper_validation/forward_2026_10_03_v1"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["APP_DATA_DIR"] = str(cache)
    os.environ["JOURNAL_DATABASE_URL"] = ""
    from core import paper_validation as pv
    import config
    if Path(config.DB_PATH).resolve() != (cache / "market.db").resolve():
        raise AssertionError("Run this helper in a fresh Python process with its independent market cache")
    state = pv.verify_archive(archive)
    if state["protocol"]["identity"] != pv.frozen_identity():
        raise pv.ProtocolChanged("Frozen sources/runtime changed; do not refresh or overwrite the old experiment")
    original = sqlite3.connect
    counts = {"formal_database_attempts": 0, "outside_database_attempts": 0}

    def guarded(database, *args, **kwargs):
        candidate = pv._database_path(database)
        if candidate == (ROOT / "data/market.db").resolve():
            counts["formal_database_attempts"] += 1
            raise AssertionError("Formal database is forbidden")
        if candidate is not None and not candidate.is_relative_to(cache.resolve()):
            counts["outside_database_attempts"] += 1
            raise AssertionError("Only the independent public market cache may be opened")
        return original(database, *args, **kwargs)

    sqlite3.connect = sqlite3.dbapi2.connect = guarded
    try:
        if not (cache / "market.db").exists():
            with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as source:
                with (cache / "market.db").open("xb") as target:
                    shutil.copyfileobj(source, target)
        if not (cache / "sox.csv").exists():
            shutil.copyfile(ROOT / "data_seed/sox.csv", cache / "sox.csv")
        from core import data_pipeline as dp, market_regime
        update = None
        if not offline:
            symbols = list(dict.fromkeys([*config.UNIVERSE, config.BENCHMARK_SYMBOL]))
            update = dp.update_symbols(symbols, time_budget_seconds=90, max_attempts=1)
            if update.get("busy"):
                return {"status": "market_cache_busy", "isolation": counts}
            sox = market_regime.load_sox()
            expected = dp._last_trading_day()
            if sox is None or sox.index[-1] + pd.Timedelta(days=3) < expected:
                market_regime.refresh_sox()
        result = pv.capture(archive, cache / "market.db", cache / "sox.csv")
        return {"status": result["status"], "data_date": result["event"]["data_date"],
                "recorded_at": result["event"]["recorded_at"],
                "reasons": result["event"]["reasons"],
                "current_block_reasons": result.get("current_block_reasons", []),
                "unrecorded_dates": result["event"]["unrecorded_dates"],
                "execution_status": result["event"]["execution_status"],
                "fills": [], "performance": None, "update": update, "isolation": counts}
    finally:
        sqlite3.connect = sqlite3.dbapi2.connect = original


def run(offline=False):
    from core import paper_validation as pv
    # The pipeline's process-local lock cannot coordinate two CLI processes.
    with pv._lock(ROOT / "data/paper_validation/market_cache"):
        return _run_locked(offline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    result = run(parser.parse_args().offline)
    print(json.dumps(result, ensure_ascii=False, indent=2))
