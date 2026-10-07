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
          "journal_is_empty_test_data": True}


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
import pandas as pd
from core import data_pipeline as dp, daily_market_update as daily
# Controlled historical preview: restore one removed public bar through the
# actual merger, then prove that subsequent UI sessions do not fetch it again.
dp._last_trading_day = lambda *args: pd.Timestamp("2026-10-02")
daily._now = lambda: dt.datetime(2026, 10, 7, 19, tzinfo=daily.TAIPEI)
dp.ensure_db()
conn = dp.get_conn()
bar = pd.read_sql_query("SELECT * FROM ohlcv WHERE symbol='2330' AND date='2026-10-02'", conn)
assert len(bar) == 1
conn.execute("DELETE FROM ohlcv WHERE symbol='2330' AND date='2026-10-02'")
conn.commit()
conn.close()
report["controlled_historical_gap"] = {"symbol": "2330", "date": "2026-10-02", "source_calls": []}
def restore_quote(symbol):
    assert symbol == "2330", "Only the controlled public-bar gap may refresh"
    report["controlled_historical_gap"]["source_calls"].append(symbol)
    dp._merge_twse_prices(symbol, bar)
    publish()
dp._refresh_market_data = restore_quote
import main as app
import flet as ft
import uvicorn

assert Path(config.DB_PATH).resolve() == CACHE / "market.db"
actual_update = app.update_symbols
actual_monthly = app.monthly_holdings


def update(*args, **kwargs):
    result = actual_update(*args, **kwargs)
    report["startup_update"] = result
    report.setdefault("update_runs", []).append(result)
    publish()
    return result


def monthly(*args, **kwargs):
    result = actual_monthly(*args, **kwargs)
    report["model"] = {"universe_size": len(config.UNIVERSE),
                       "latest_ranking_count": len(result["ranking"]),
                       "current_holdings": result["model_current_holdings"],
                       "selection_date": result["selection_date"],
                       "validation_status": result["validation_status"],
                       "pool_label": app._universe_text(result),
                       "data_quality": result["data_quality"]}
    publish()
    return result


app.update_symbols = update
app.monthly_holdings = monthly
web = ft.run(app._build_app, export_asgi_app=True)
uvicorn.run(web, host="127.0.0.1", port=int(os.environ["UNIVERSE_UI_PORT"]), log_level="warning")
