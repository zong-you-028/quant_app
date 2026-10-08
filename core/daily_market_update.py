"""App-only daily download policy; frozen research pipelines stay unchanged.

Completed downloads are recorded in the same SQLite DB as the quotes, using
Taipei calendar days. Failed/incomplete work can resume without refetching
completed symbols. An expiring DB lease also covers separate app processes.
"""
from contextlib import closing, contextmanager
import datetime as dt
import json
import time
import uuid

import numpy as np
import pandas as pd

from core import data_pipeline as dp, market_regime as mr

TAIPEI = dt.timezone(dt.timedelta(hours=8))
PREFIX = "app_daily_update:"


def _now():
    return dt.datetime.now(TAIPEI)


def _read(conn, key):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (PREFIX + key,)).fetchone()
    return json.loads(row[0]) if row else {}


def _write(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                 (PREFIX + key, json.dumps(value)))


@contextmanager
def _lease(resource):
    dp.ensure_db()
    owner = uuid.uuid4().hex
    key = "lease:" + resource
    with closing(dp.get_conn()) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)")
        conn.execute("BEGIN IMMEDIATE")
        value = _read(conn, key)
        acquired = value.get("expires", 0) <= time.time()
        if acquired:
            _write(conn, key, {"owner": owner, "expires": time.time() + 600})
        conn.commit()
    try:
        yield acquired
    finally:
        if acquired:
            with closing(dp.get_conn()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                if _read(conn, key).get("owner") == owner:
                    conn.execute("DELETE FROM meta WHERE key=?", (PREFIX + key,))
                conn.commit()


def update_symbols(symbols, force=False, ignore_throttle=False, **kwargs):
    """At most one successful download per symbol per Taipei day.

    Existing current rows require no download and consume no daily allowance.
    The low-level updater still commits each response before returning; all
    freshness/status displays are reconstructed from the actual stored rows.
    """
    syms = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    on_saved = kwargs.pop("on_saved", None)
    progress = kwargs.get("progress")
    saved_errors = {}

    def after_commit(done, total, symbol, status):
        # The pipeline reports completion after its DB commit. Valuation errors
        # must not turn a successful quote into a retry or another download.
        if on_saved and status in ("updated", "current", "stale"):
            try:
                on_saved(symbol)
            except Exception as exc:
                saved_errors[symbol] = str(exc)
        if progress:
            progress(done, total, symbol, status)

    kwargs["progress"] = after_commit
    if kwargs.get("time_budget_seconds") is None:
        kwargs["time_budget_seconds"] = 90
    if kwargs.get("time_budget_seconds") is not None and kwargs["time_budget_seconds"] > 300:
        raise ValueError("Daily UI batches must stay below the DB lease duration")
    day = _now().date().isoformat()
    expected = dp._last_trading_day()
    with _lease("stocks") as acquired:
        if not acquired:
            return {"busy": True, "updated": 0, "current": 0, "stale": 0, "failed": 0,
                    "pending": len(syms), "errors": {}, "stale_chip_symbols": []}
        with closing(dp.get_conn()) as conn:
            completed = {s for s in syms if _read(conn, "stock:" + s).get("day") == day}
        eligible = [s for s in syms if s not in completed and (force or dp.needs_update(s))]
        result = dp.update_symbols(eligible, force=force, ignore_throttle=True, **kwargs)
        if result.get("busy"):
            return result
        excluded = set(result.get("failed_symbols", []) + result.get("stale_symbols", [])
                       + result.get("pending_symbols", []))
        # Only a confirmed, committed success uses today's allowance.
        with closing(dp.get_conn()) as conn:
            for symbol in eligible:
                if symbol not in excluded and not dp.needs_update(symbol):
                    _write(conn, "stock:" + symbol, {"day": day,
                           "saved_at": _now().isoformat(), "target": result["expected_asof"]})
            conn.commit()
        for symbol in syms:
            price_date = dp.last_ohlcv_date(symbol)
            chip_date = dp.last_chip_date(symbol) if dp._chip_required(symbol) else None
            result["data_dates"][symbol] = price_date.strftime("%Y-%m-%d") if price_date is not None else None
            result["chip_dates"][symbol] = chip_date.strftime("%Y-%m-%d") if chip_date is not None else None
            if symbol in eligible:
                continue
            stale = price_date is None or price_date.normalize() < expected or dp.chip_needs_update(symbol)
            status = "stale" if stale else "current"
            result[status] += 1
            if stale:
                result["stale_symbols"].append(symbol)
            if dp.chip_needs_update(symbol):
                result["stale_chip_symbols"].append(symbol)
            for name in ("on_start", "progress"):
                callback = progress if name == "progress" else kwargs.get(name)
                if callback:
                    callback(len(syms), len(syms), symbol, "讀取今日已保存資料" if symbol in completed else "current")
        dates = [value for value in result["data_dates"].values() if value]
        result["asof"] = min(dates) if dates and len(dates) == len(syms) else None
        result["latest_asof"] = max(dates) if dates else None
        result["daily_skipped_symbols"] = sorted(completed)
        result["daily_saved_on"] = day
        result["database_saved"] = True
        result["total_symbols"] = len(syms)
        result["on_saved_errors"] = saved_errors
        return result


def refresh_sox(raise_errors=True):
    """Save SOX closes to SQLite and keep the existing strategy CSV in sync."""
    day = _now().date().isoformat()
    with _lease("sox") as acquired:
        if not acquired:
            raise RuntimeError("另一個畫面正在更新費半，未重複下載")
        with closing(dp.get_conn()) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS market_index_prices "
                         "(symbol TEXT NOT NULL,date TEXT NOT NULL,close REAL NOT NULL,PRIMARY KEY(symbol,date))")
            done = _read(conn, "sox").get("day") == day
            rows = conn.execute("SELECT date,close FROM market_index_prices WHERE symbol=? ORDER BY date",
                                ("^SOX",)).fetchall() if done else []
            conn.commit()
        if done and rows:
            stored = pd.Series([r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows]), name="close")
            cached = mr._cached_sox()
            if cached is None or not cached.equals(stored):
                mr._write_sox(stored)
            return stored
        series = mr.refresh_sox(raise_errors=raise_errors)
        if (series is None or series.empty or not np.isfinite(series.values).all()
                or not series.gt(0).all()):
            raise RuntimeError("費半無有效資料，未標記今日更新完成")
        if series.index.max() + pd.Timedelta(days=3) < dp._last_trading_day():
            raise RuntimeError("費半資料仍過期，保留舊資料並允許續抓")
        with closing(dp.get_conn()) as conn:
            conn.executemany("INSERT OR REPLACE INTO market_index_prices(symbol,date,close) VALUES (?,?,?)",
                             [("^SOX", str(date.date()), float(value)) for date, value in series.items()])
            _write(conn, "sox", {"day": day, "saved_at": _now().isoformat(),
                                "data_date": str(series.index.max().date())})
            conn.commit()
        mr._write_sox(series)
        return series
