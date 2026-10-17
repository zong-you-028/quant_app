"""Round-trip the whole public seed through fake cloud bundles, never a ledger."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.robustness_audit import isolated_public_seed


class Cloud:
    def __init__(self):
        self.rows = {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def save_bundle(self, key, value):
        from core import market_cache_storage as storage
        self.rows[key] = storage._encode(value)
    def read_all(self):
        return list(self.rows.items())


def counts():
    from core import data_pipeline as dp
    with closing(dp.get_conn()) as conn:
        return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("ohlcv", "chip_weekly", "stock_info", "market_index_prices")}


def main():
    with isolated_public_seed() as (folder, guard):
        import config
        import pandas as pd
        from core import data_pipeline as dp, market_regime as mr, market_cache_storage as storage
        cloud = Cloud()
        with patch.dict("os.environ", {"MARKET_DATABASE_URL": "postgresql://synthetic.invalid/public"}), \
                patch.object(storage, "_open_cloud", lambda: cloud), \
                patch.object(mr, "SOX_CSV", str(folder / "restored_sox.csv")):
            dp.ensure_db()
            with closing(dp.get_conn()) as conn:
                seed_symbols = [row[0] for row in conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol")]
            symbols = list(dict.fromkeys([*config.UNIVERSE, config.BENCHMARK_SYMBOL, *seed_symbols]))
            # These are the exact already-bundled public index observations.
            sox = pd.read_csv(folder / "sox.csv", index_col=0).iloc[:, 0]
            with closing(dp.get_conn()) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS market_index_prices(symbol TEXT,date TEXT,close REAL,PRIMARY KEY(symbol,date))")
                conn.executemany("INSERT OR REPLACE INTO market_index_prices VALUES (?,?,?)",
                                 [("^SOX", str(pd.Timestamp(day).date()), float(value)) for day, value in sox.items()])
                conn.commit()
            before = counts()
            stocks = storage.save_symbols(symbols)
            if not stocks["durable"]:
                raise AssertionError("Public stock serialization failed: " + str(stocks["errors"]))
            assert storage.save_sox()["durable"]
            with patch.object(config, "DB_PATH", str(folder / "cold.db")), \
                    patch.object(dp, "_hydrate_market_seed", lambda conn: None):
                result = storage.restore_market_cache()
                if not result["durable"]:
                    raise AssertionError("Public seed round-trip failed: " + str(result["errors"]))
                after = counts()
                assert before == after, {"before": before, "after": after}
                assert len(result["restored_symbols"]) == len(symbols)
            report = {"passed": True, "stocks_round_tripped": len(symbols),
                      "resources": len(cloud.rows), "row_counts": before,
                      "compressed_bundle_bytes": sum(len(value) for value in cloud.rows.values()),
                      "isolation": dict(guard), "real_cloud_access": False, "journal_access": False}
    if any(guard.get(key, 0) for key in ("formal_database_attempts", "outside_database_attempts", "external_download_attempts")):
        raise AssertionError("Forbidden access during public seed QA")
    Path(__file__).with_name("public_seed_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
