"""Local browser QA: temporary market data, empty journal, no external requests."""
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
assert Path(os.environ["APP_DATA_DIR"]).resolve() != (ROOT / "data").resolve()
os.environ["JOURNAL_DATABASE_URL"] = ""
os.environ.pop("QUANT_APP_OFFLINE", None)
import config
from core import data_pipeline as dp
import main as app
import flet as ft
import uvicorn

original = sqlite3.connect


def guarded(database, *args, **kwargs):
    if str(database) != ":memory:" and Path(database).resolve() == (ROOT / "data/market.db").resolve():
        raise AssertionError("QA cannot access the formal database")
    return original(database, *args, **kwargs)


sqlite3.connect = sqlite3.dbapi2.connect = guarded
dp.requests.get = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("QA external downloads forbidden"))
calls = 0


def partial_update(symbols, **kwargs):
    global calls
    calls += 1
    kwargs["on_start"](0, len(symbols), "2330", "更新行情（隔離情境）")
    time.sleep(3)
    if calls == 1:
        return {"updated": 1, "current": 0, "failed": 0, "stale": 0,
                "pending": len(symbols)-1, "time_budget_reached": True,
                "asof": "2026-10-02", "latest_asof": "2026-10-02", "errors": {}}
    if calls == 2:
        return {"updated": 0, "current": len(symbols)-1, "failed": 1, "stale": 0,
                "pending": 0, "failed_symbols": ["2330"], "errors": {"2330": "測試來源逾時"}}
    return {"updated": 0, "current": len(symbols), "failed": 0, "stale": 0,
            "pending": 0, "errors": {}, "asof": "2026-10-02", "latest_asof": "2026-10-02"}


if os.environ.get("UPDATE_QA_SCENARIO") == "partial":
    app.update_symbols = partial_update

web = ft.run(app._build_app, export_asgi_app=True)
uvicorn.run(web, host="127.0.0.1", port=int(os.environ["UPDATE_QA_PORT"]), log_level="warning")
