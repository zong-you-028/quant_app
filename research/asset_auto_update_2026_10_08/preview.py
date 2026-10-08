"""Disposable 150-stock UI preview; no formal database or external network."""
import ipaddress
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CACHE = Path(os.environ["APP_DATA_DIR"]).resolve()
assert CACHE != (ROOT / "data").resolve()
os.environ["JOURNAL_DATABASE_URL"] = ""
os.environ["APP_PASSWORD"] = ""
os.environ.pop("QUANT_APP_OFFLINE", None)
META = Path(os.environ["UNIVERSE_UI_META"])
counts = {"formal_database_attempts": 0, "outside_database_attempts": 0,
          "external_network_attempts": 0, "isolated_database_connections": 0}
report = {"isolation": counts, "market_and_journal_cache": str(CACHE),
          "journal_is_synthetic_test_data": True}


def publish():
    target = META.with_suffix(".tmp")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(target, META)


original_connect = sqlite3.connect


def guarded(database, *args, **kwargs):
    value = os.fspath(database) if not isinstance(database, int) else str(database)
    if value != ":memory:":
        if value.startswith("file:"):
            value = unquote(urlsplit(value).path)
            if value.startswith("/") and len(value) > 2 and value[2] == ":":
                value = value[1:]
        path = Path(value).resolve()
        if path == (ROOT / "data/market.db").resolve():
            counts["formal_database_attempts"] += 1
            publish()
            raise AssertionError("Formal database forbidden in preview")
        if not path.is_relative_to(CACHE):
            counts["outside_database_attempts"] += 1
            publish()
            raise AssertionError("Preview only opens disposable public cache")
        counts["isolated_database_connections"] += 1
    return original_connect(database, *args, **kwargs)


sqlite3.connect = sqlite3.dbapi2.connect = guarded
original_socket_connect = socket.socket.connect


def guarded_socket(self, address):
    if isinstance(address, tuple) and self.family in (socket.AF_INET, socket.AF_INET6):
        host = address[0]
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = host == "localhost"
        if not allowed:
            counts["external_network_attempts"] += 1
            publish()
            raise AssertionError("Preview external socket forbidden")
    return original_socket_connect(self, address)


socket.socket.connect = guarded_socket
import requests


def deny_request(*args, **kwargs):
    counts["external_network_attempts"] += 1
    publish()
    raise AssertionError("Preview external HTTP forbidden")


requests.sessions.Session.request = deny_request
# yfinance uses curl_cffi rather than requests; guard that route as well.
from curl_cffi import requests as curl_requests
curl_requests.Session.request = deny_request
publish()

import config
import datetime as dt
import time
import pandas as pd
from core import data_pipeline as dp, daily_market_update as daily, journal

dp._last_trading_day = lambda *a: pd.Timestamp("2026-10-02")
daily._now = lambda: dt.datetime(2026, 10, 8, 19, tzinfo=daily.TAIPEI)
dp.ensure_db()
assert Path(config.DB_PATH).resolve() == CACHE / "market.db"
conn = dp.get_conn()
for symbol, price in (("2330", 100.), ("2317", 50.)):
    conn.execute("DELETE FROM ohlcv WHERE symbol=? AND date='2026-10-02'", (symbol,))
    conn.execute("UPDATE ohlcv SET close=? WHERE symbol=? AND date='2026-10-01'", (price, symbol))
conn.commit()
conn.close()
journal.add_current_asset("2330", 10, 100, asof="2026-10-01")
journal.add_current_asset("2317", 20, 50, asof="2026-10-01")
journal.set_cash_balance(1000)
actual_snapshot = journal.snapshot_assets_daily
actual_snapshot(now=dt.datetime(2026, 10, 7, 19, tzinfo=daily.TAIPEI), expected_asof="2026-10-01")
report["source_calls"] = []
report["asset_saves"] = []

def restore_quote(symbol):
    assert symbol in ("2330", "2317"), "Only synthetic held quotes may refresh"
    report["source_calls"].append(symbol)
    publish()
    # Make the intermediate valuation observable in a real browser.
    time.sleep(3 if len(report["source_calls"]) == 1 else 9)
    price = 110. if symbol == "2330" else 55.
    frame = pd.DataFrame([dict(symbol=symbol, date="2026-10-02", open=price,
                               high=price, low=price, close=price, volume=1000.)])
    dp._merge_twse_prices(symbol, frame)

dp._refresh_market_data = restore_quote

def snapshot(**kwargs):
    result = actual_snapshot(now=dt.datetime(2026, 10, 8, 19, tzinfo=daily.TAIPEI), **kwargs)
    report["asset_saves"].append({"outcome": result, "history": journal.list_asset_history()})
    publish()
    return result

journal.snapshot_assets_daily = snapshot
import main as app
import flet as ft
import uvicorn
actual_update = app.update_symbols

def update(symbols, **kwargs):
    # No model is recalculated; put synthetic held quotes first in this preview.
    ordered = [s for s in ("2330", "2317") if s in symbols] + [s for s in symbols if s not in ("2330", "2317")]
    result = actual_update(ordered, **kwargs)
    report.setdefault("update_runs", []).append(result)
    publish()
    return result

app.update_symbols = update
web = ft.run(app._build_app, export_asgi_app=True)
uvicorn.run(web, host="127.0.0.1", port=int(os.environ["UNIVERSE_UI_PORT"]), log_level="warning")
