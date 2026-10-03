"""Refresh only a temporary copy of the three-table public market seed."""
import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix="quant_update_seed_") as directory:
        os.environ["APP_DATA_DIR"] = directory
        import config
        from core import data_pipeline as dp
        database = Path(config.DB_PATH)
        with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as src, database.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        real_connect = sqlite3.connect

        def guarded(path, *args, **kwargs):
            if str(path) != ":memory:" and Path(path).resolve() == (ROOT / "data/market.db").resolve():
                raise AssertionError("formal database access forbidden")
            return real_connect(path, *args, **kwargs)

        sqlite3.connect = guarded
        try:
            symbols = list(dict.fromkeys(config.UNIVERSE + config.WATCHLIST + [config.BENCHMARK_SYMBOL]))
            started = time.monotonic()
            result = dp.update_symbols(symbols, ignore_throttle=True, time_budget_seconds=90, max_attempts=1)
            report = {"database_isolated": True, "formal_database_attempts": 0,
                      "source": "TWSE official monthly prices / FinMind shareholding",
                      "symbols": symbols, "elapsed_seconds": round(time.monotonic()-started, 2),
                      "result": result}
            # Second invocation demonstrates a no-download current-cache check.
            if not any(result[k] for k in ("pending", "failed", "stale")):
                report["second_check"] = dp.update_symbols(symbols, time_budget_seconds=90, max_attempts=1)
                source_db = guarded(database)
                export_path = Path(directory) / "export.db"
                export_db = guarded(export_path)
                for table in ("ohlcv", "chip_weekly", "stock_info"):
                    schema = source_db.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
                    export_db.execute(schema)
                    rows = source_db.execute(f"SELECT * FROM {table}").fetchall()
                    placeholders = ",".join("?" for _ in rows[0])
                    export_db.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                export_db.commit()
                report["seed_tables"] = [r[0] for r in export_db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                assert set(report["seed_tables"]) == {"ohlcv", "chip_weekly", "stock_info"}
                assert export_db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                export_db.close()
                source_db.close()
                with export_path.open("rb") as src, gzip.open(ROOT / "data_seed/market.db.gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                # This is a public index cache; no journal data is copied.
                shutil.copyfile(ROOT / "data/sox.csv", ROOT / "data_seed/sox.csv")
                report["seed_exported"] = True
            else:
                report["seed_exported"] = False
            (HERE / "live_update_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"seconds": report["elapsed_seconds"], "exported": report["seed_exported"],
                              "updated": result["updated"], "current": result["current"],
                              "failed": result["failed"], "stale": result["stale"], "pending": result["pending"],
                              "errors": result["errors"]}, ensure_ascii=True))
        finally:
            sqlite3.connect = real_connect


if __name__ == "__main__":
    main()
