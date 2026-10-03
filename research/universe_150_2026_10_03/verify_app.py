"""Exercise the actual 150-stock app path, using only an isolated public seed."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run():
    from core.robustness_audit import database_path
    source_hash = hashlib.sha256((ROOT / "data_seed/market.db.gz").read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="quant_150_app_") as directory:
        directory = Path(directory)
        os.environ["APP_DATA_DIR"] = str(directory)
        os.environ["JOURNAL_DATABASE_URL"] = ""
        # config may have been imported by robustness_audit; explicitly confine
        # all actual app readers to the disposable market-only copy.
        import config
        config.DATA_DIR, config.DB_PATH = str(directory), str(directory / "market.db")
        with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as source, (directory / "market.db").open("wb") as target:
            shutil.copyfileobj(source, target)
        shutil.copyfile(ROOT / "data_seed/sox.csv", directory / "sox.csv")
        original = sqlite3.connect
        counts = {"formal_database_attempts": 0, "outside_database_attempts": 0, "network_attempts": 0}

        def guard(database, *args, **kwargs):
            path = database_path(database)
            if path == (ROOT / "data/market.db").resolve():
                counts["formal_database_attempts"] += 1
                raise AssertionError("Formal DB forbidden")
            if path is not None and not path.is_relative_to(directory):
                counts["outside_database_attempts"] += 1
                raise AssertionError("Only isolated public cache allowed")
            return original(database, *args, **kwargs)

        import requests
        request = requests.sessions.Session.request
        def deny(*args, **kwargs):
            counts["network_attempts"] += 1
            raise AssertionError("Network forbidden during actual app calculation")
        sqlite3.connect = sqlite3.dbapi2.connect = guard
        requests.sessions.Session.request = deny
        try:
            from core import data_pipeline as dp, rotation
            started = monotonic()
            update = dp.update_symbols([*config.UNIVERSE, config.BENCHMARK_SYMBOL], time_budget_seconds=90, max_attempts=1)
            result = rotation.run_rotation()
            from main import _universe_text, make_model_status_card
            make_model_status_card(result)
            # Exact comparable-return interval from the original comparison.
            returns = result["net_returns"].loc["2018-09-18":"2026-10-02"]
            cagr = (1 + returns).prod() ** (252 / len(returns)) - 1
            report = {"universe_size": len(config.UNIVERSE), "latest_ranking_count": len(result["ranking"]),
                      "seed_sha256": source_hash, "period": [str(returns.index.min().date()), str(returns.index.max().date())],
                      "observations": len(returns), "comparison_cagr": float(cagr),
                      "model_current_holdings": result["model_current_holdings"],
                      "target_holdings": result["holdings"], "model_current_weights": result["model_current_weights"],
                      "selection_date": result["selection_date"], "data_quality": result["data_quality"],
                      "pool_label": _universe_text(result), "validation_status": result["validation_status"],
                      "update": update, "elapsed_seconds": round(monotonic() - started, 2), "isolation": counts}
            assert len(result["model_current_holdings"]) <= 8
            assert update["current"] == 151 and not any(update[key] for key in ("failed", "pending", "stale"))
            assert not result["data_quality"]["stale"]
            assert all(value == 0 for value in counts.values())
            assert abs(cagr - .6003) < .0001
        finally:
            sqlite3.connect = sqlite3.dbapi2.connect = original
            requests.sessions.Session.request = request
    output = Path(__file__).with_name("app_verification.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("universe_size", "latest_ranking_count", "comparison_cagr", "elapsed_seconds", "isolation")}))


if __name__ == "__main__":
    run()
