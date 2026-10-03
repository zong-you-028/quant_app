"""Local-only UI QA: public market seed plus synthetic in-memory positions."""
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
assert Path(os.environ["APP_DATA_DIR"]).resolve() != (ROOT / "data").resolve()
os.environ["JOURNAL_DATABASE_URL"] = ""
os.environ["QUANT_APP_OFFLINE"] = "1"
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
dp.requests.get = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("external market downloads forbidden"))
model = app.monthly_holdings()
scenario = os.environ["RECONCILE_QA_SCENARIO"]
current = model["model_current_holdings"]
assert current
added = next(s for s in config.UNIVERSE if s not in current)
if scenario == "pending":
    model["holdings"] = model["held"] = current[1:] + [added]
    model["target_execution_date"] = None
    model["target_execution_pending"] = True
if scenario == "stale":
    model["data_quality"]["stale"] = True


def position(symbol):
    return {"symbol": symbol, "name": model["names"].get(symbol, "隔離測試持倉"),
            "shares": 10., "cost": 1000., "market_value": 1000., "average_cost": 100.,
            "pnl": 0., "return": 0., "lots": 1}


scan_calls, book_calls = 0, 0


def scan():
    global scan_calls
    scan_calls += 1
    return model


def book():
    global book_calls
    book_calls += 1
    values = [position(current[0]), position("TEST_OUT")]
    if scenario == "real" and book_calls >= 3:
        values = [position("TEST_OUT")]
    elif scenario == "pending":
        values.append(position(added))
    return values


app.monthly_holdings = scan
app.journal.positions = book


def main(page):
    app._build_app(page)
    page.title = "持倉核對｜隔離測試，帳本為模擬資料"
    page.update()
    Path(os.environ["RECONCILE_QA_META"]).write_text(json.dumps({
        "current": current, "pending_added": added, "scenario": scenario,
        "positions_are_synthetic": True}, ensure_ascii=False), encoding="utf-8")


web = ft.run(main, export_asgi_app=True)
uvicorn.run(web, host="127.0.0.1", port=int(os.environ["RECONCILE_QA_PORT"]), log_level="warning")
