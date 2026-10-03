"""Public seed only: real model state, <=8 stocks and unchanged return math."""
import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
with tempfile.TemporaryDirectory(prefix="quant_reconcile_model_") as directory:
    os.environ["APP_DATA_DIR"] = directory
    os.environ["JOURNAL_DATABASE_URL"] = ""
    with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as src, open(Path(directory)/"market.db", "wb") as dst:
        shutil.copyfileobj(src, dst)
    shutil.copyfile(ROOT / "data_seed/sox.csv", Path(directory) / "sox.csv")
    import config
    from core import data_pipeline as dp, rotation
    from core.portfolio_reconcile import reconcile_portfolio
    formal_attempts, http_attempts = [0], [0]
    real_connect = sqlite3.connect

    def guarded(database, *args, **kwargs):
        if str(database) != ":memory:" and Path(database).resolve() == (ROOT / "data/market.db").resolve():
            formal_attempts[0] += 1
            raise AssertionError("formal DB access forbidden")
        return real_connect(database, *args, **kwargs)

    def no_network(*args, **kwargs):
        http_attempts[0] += 1
        raise AssertionError("external market download forbidden")

    sqlite3.connect = sqlite3.dbapi2.connect = guarded
    dp.requests.get = no_network
    model = rotation.run_rotation()
    sample_positions = ([{"symbol": model["model_current_holdings"][0], "shares": 10.}]
                        if model["model_current_holdings"] else [])
    sample_positions.append({"symbol": "TEST_OUT", "name": "隔離測試持倉", "shares": 5.})
    view = reconcile_portfolio(model, sample_positions)
    assert view["verified"] and len(view["current"]) <= 8 and len(view["next_target"]) <= 8
    assert "TEST_OUT" in view["actionable_sells"]
    old = {}
    reference = "e841827c2e791c902082cf72263c7dd0e3a1ea77"
    source = subprocess.check_output(["git", "-c", "safe.directory=D:/quant_app", "show", reference + ":core/execution.py"], cwd=ROOT, text=True, encoding="utf-8")
    exec(compile(source, "previous_execution.py", "exec"), old)
    targets = model["target_weights"]
    prices = {s: dp.load_ohlcv(s) for s in targets.columns}
    opens = pd.DataFrame({s: p["open"] for s, p in prices.items()})
    closes = pd.DataFrame({s: p["close"] for s, p in prices.items()})
    old_returns, _ = old["backtest_open_execution"](
        targets, opens, closes, lag=model["execution_lag"], cost=model["cost_per_turnover"],
        rebalance=pd.Series(targets.index.isin(targets.index[::model["rebal_days"]]), index=targets.index))
    equal = np.array_equal(old_returns.to_numpy(), model["net_returns"].to_numpy())
    assert equal, "metadata addition changed return values"
    report = {"formal_database_attempts": formal_attempts[0], "external_download_attempts": http_attempts[0],
              "asof": view["asof"], "model_current": model["model_current_holdings"],
              "latest_target": model["holdings"], "pending_change": view["pending_change"],
              "model_cash_weight": view["model_cash_weight"], "returns_identical_to_previous_commit": bool(equal),
              "comparison_positions_are_synthetic": True, "sample_exit": view["actionable_sells"], "reference_commit": reference}
    Path(__file__).with_name("model_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    sqlite3.connect = sqlite3.dbapi2.connect = real_connect
