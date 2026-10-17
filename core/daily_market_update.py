"""App-only target-aware updates and durable public market-cache checkpoints.

Completed downloads are recorded in the same SQLite DB as the quotes, using
Taipei calendar days. Failed/incomplete work can resume without refetching
completed symbols. An expiring DB lease also covers separate app processes.
"""
from contextlib import closing, contextmanager
import datetime as dt
import hashlib
import json
import os
import threading
import time
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd

from core import data_pipeline as dp, market_regime as mr

TAIPEI = dt.timezone(dt.timedelta(hours=8))
PREFIX = "app_daily_update:"
_BASE_REFRESH = dp._refresh_market_data
_BASE_SOX_REFRESH = mr.refresh_sox
_RESTORE_LOCK = threading.Lock()
_RESTORED = set()
_RESTORE_RESULTS = {}
_PUBLISHED_SYMBOLS = {}
_STORAGE_STATE = {}


def _cache_key():
    url = (os.environ.get("MARKET_DATABASE_URL", "").strip()
           or os.environ.get("JOURNAL_DATABASE_URL", "").strip())
    return (os.path.abspath(dp.config.DB_PATH), hashlib.sha256(url.encode()).hexdigest())


def storage_status():
    from core import market_cache_storage as storage
    if not storage.configured():
        return {"configured": False, "durable": False, "backend": "local_sqlite"}
    return {"configured": True, "durable": False, **_STORAGE_STATE}


def restore_cache():
    """Restore the durable public cache before the app checks stale seed rows."""
    from core import market_cache_storage as storage
    if not storage.configured():
        return storage_status()
    key = _cache_key()
    with _RESTORE_LOCK:
        if key in _RESTORED:
            # A later failed upload must not poison successful restoration and
            # block the next upload-only retry before it can start.
            return {**_RESTORE_RESULTS[key], "already_restored": True}
        outcome = storage.restore_market_cache()
        _STORAGE_STATE.clear()
        _STORAGE_STATE.update(outcome)
        if outcome.get("durable"):
            _RESTORED.add(key)
            _RESTORE_RESULTS[key] = dict(outcome)
            _PUBLISHED_SYMBOLS[key] = set(outcome.get("restored_symbols", []))
        return outcome


def _mark_dirty(symbol):
    with closing(dp.get_conn()) as conn:
        _write(conn, "dirty:" + symbol, {"pending": True})
        conn.commit()


def _save_symbols(symbols):
    """A cloud failure keeps local quotes and a retry marker, never a false ack."""
    from core import market_cache_storage as storage
    if not storage.configured():
        return storage_status()
    outcome = storage.save_symbols(symbols)
    _STORAGE_STATE.clear()
    _STORAGE_STATE.update(outcome)
    with closing(dp.get_conn()) as conn:
        for symbol in outcome.get("saved_symbols", []):
            conn.execute("DELETE FROM meta WHERE key=?", (PREFIX + "dirty:" + symbol,))
            _PUBLISHED_SYMBOLS.setdefault(_cache_key(), set()).add(symbol)
        conn.commit()
    return outcome


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
    """Deduplicate confirmed downloads for the same Taipei day AND data target.

    Existing current rows require no download and consume no daily allowance.
    The low-level updater still commits each response before returning; all
    freshness/status displays are reconstructed from the actual stored rows.
    """
    syms = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    on_saved = kwargs.pop("on_saved", None)
    progress = kwargs.get("progress")
    saved_errors = {}
    statuses, attempted, cloud_errors = {}, [], {}
    cloud_saved_symbols, saved_after_failure = set(), set()
    storage_unavailable = False
    from core import market_backfill, market_cache_storage as storage
    source = None

    def persist(symbols):
        nonlocal storage_unavailable
        outcome = _save_symbols(symbols)
        cloud_saved_symbols.update(outcome.get("saved_symbols", []))
        for symbol in outcome.get("saved_symbols", []):
            cloud_errors.pop(symbol, None)
        if not outcome.get("durable"):
            for symbol in outcome.get("failed_symbols", symbols):
                cloud_errors[symbol] = outcome.get("error") or "雲端行情保存未完成；本機資料已保留"
        storage_unavailable = bool(cloud_errors)
        return outcome

    def checkpoint(symbol):
        if storage.configured():
            _mark_dirty(symbol)
            outcome = persist([symbol])
            if not outcome.get("durable"):
                cloud_errors[symbol] = outcome.get("error") or "雲端行情保存未完成；本機資料已保留"
                raise RuntimeError(cloud_errors[symbol])

    def refresh(symbol):
        if storage_unavailable:
            raise dp.UpdateBudgetExceeded("雲端保存未完成，暫停後續下載並保留進度")
        if symbol not in attempted:
            attempted.append(symbol)
        before = (dp.last_ohlcv_date(symbol),
                  dp.last_chip_date(symbol) if dp._chip_required(symbol) else None)
        successful = False
        try:
            if source is _BASE_REFRESH:
                market_backfill.refresh_symbol(symbol, checkpoint=checkpoint)
            else:
                source(symbol)  # Explicit source injection for isolated app tests.
            successful = True
        finally:
            # A source may commit prices and then fail while loading chips.
            # Preserve that partial progress in the cloud as well.
            after = (dp.last_ohlcv_date(symbol),
                     dp.last_chip_date(symbol) if dp._chip_required(symbol) else None)
            unpublished = symbol not in _PUBLISHED_SYMBOLS.get(_cache_key(), set())
            if storage.configured() and after[0] is not None and (successful or after != before or unpublished):
                _mark_dirty(symbol)
                if not successful:
                    # The low-level budget exception skips progress callbacks.
                    # Save an already committed quote before returning pending.
                    persist([symbol])
                    saved_after_failure.add(symbol)

    def after_commit(done, total, symbol, status):
        # The pipeline reports completion after its DB commit. Valuation errors
        # must not turn a successful quote into a retry or another download.
        statuses[symbol] = status
        if symbol in attempted:
            with closing(dp.get_conn()) as conn:
                if status in ("updated", "current") and not dp.needs_update(symbol):
                    _write(conn, "stock:" + symbol, {"day": day,
                           "saved_at": _now().isoformat(), "target": expected.strftime("%Y-%m-%d")})
                conn.commit()
            with closing(dp.get_conn()) as conn:
                dirty = _read(conn, "dirty:" + symbol).get("pending")
            if storage.configured() and dirty and symbol not in saved_after_failure:
                outcome = persist([symbol])
                if not outcome.get("durable"):
                    cloud_errors[symbol] = outcome.get("error") or "雲端行情保存未完成；本機資料已保留"
                    with closing(dp.get_conn()) as conn:
                        conn.execute("DELETE FROM meta WHERE key=?", (PREFIX + "stock:" + symbol,))
                        conn.commit()
                else:
                    cloud_errors.pop(symbol, None)
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
        source = dp._refresh_market_data
        # Adopt already-current legacy local data into an initially empty cloud
        # without manufacturing another market download.
        if storage.configured():
            published = _PUBLISHED_SYMBOLS.get(_cache_key(), set())
            for symbol in syms:
                if symbol not in published and not dp.needs_update(symbol) and dp.has_symbol(symbol):
                    _mark_dirty(symbol)
        with closing(dp.get_conn()) as conn:
            completed = {s for s in syms if _read(conn, "stock:" + s).get("day") == day
                         and _read(conn, "stock:" + s).get("target") == expected.strftime("%Y-%m-%d")
                         and not dp.needs_update(s)}
            dirty = [s for s in syms if _read(conn, "dirty:" + s).get("pending")]
        if dirty and storage.configured():
            persist(dirty)
        eligible = [s for s in syms if s not in completed and (force or dp.needs_update(s))]
        with patch.object(dp, "_refresh_market_data", refresh):
            result = dp.update_symbols(eligible, force=force, ignore_throttle=True, **kwargs)
        if result.get("busy"):
            return result
        for symbol in syms:
            price_date = dp.last_ohlcv_date(symbol)
            chip_date = dp.last_chip_date(symbol) if dp._chip_required(symbol) else None
            result["data_dates"][symbol] = price_date.strftime("%Y-%m-%d") if price_date is not None else None
            result["chip_dates"][symbol] = chip_date.strftime("%Y-%m-%d") if chip_date is not None else None
            if symbol in eligible:
                continue
            stale = price_date is None or price_date.normalize() < expected or dp.chip_needs_update(symbol)
            status = "stale" if stale else "current"
            statuses[symbol] = status
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
        statuses.update(dict.fromkeys(result.get("pending_symbols", []), "pending"))
        result["symbol_statuses"] = statuses
        result["attempted_symbols"] = attempted
        result["cloud_errors"] = cloud_errors
        result["cloud_saved_symbols"] = sorted(cloud_saved_symbols)
        result["storage"] = storage_status()
        result["cloud_saved"] = storage.configured() and not cloud_errors and storage_status().get("durable", False)
        return result


def _sox_target():
    """Conservative completed US session date, independent of Taiwan's 18:00.

    Six Taipei AM covers the regular US close in both DST seasons. Exchange
    holidays are reported as unavailable dates rather than fabricated bars.
    """
    now = _now()
    date = now.date() - dt.timedelta(days=1 if now.hour >= 6 else 2)
    while date.weekday() >= 5:
        date -= dt.timedelta(days=1)
    return pd.Timestamp(date)


def _download_sox():
    """App-only refresh: don't let the strategy reader's 3-day cache skip HTTP."""
    cached = mr._cached_sox()
    if cached is None:
        cached = mr._cached_sox(os.path.join(dp.config.BASE_DIR, "data_seed", "sox.csv"))
    start = ((cached.index.max() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
             if cached is not None else "2014-06-01")
    import yfinance as yf
    frame = yf.download(getattr(dp.config, "ROTATION_SOX_SYMBOL", "^SOX"),
                        start=start, progress=False, auto_adjust=True, threads=False, timeout=8)
    if frame is None or frame.empty:
        raise RuntimeError("費半來源未回傳新資料")
    series = frame["Close"]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = series.astype(float).rename("close")
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.dropna().sort_index().loc[:_sox_target()]
    if series.empty:
        raise RuntimeError("費半來源尚無已完成交易日的資料")
    if cached is not None:
        series = pd.concat([cached.loc[:_sox_target()], series])
        series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _save_sox_cloud():
    from core import market_cache_storage as storage
    if not storage.configured():
        return
    outcome = storage.save_sox()
    _STORAGE_STATE.clear()
    _STORAGE_STATE.update(outcome)
    if not outcome.get("durable"):
        raise RuntimeError(outcome.get("error") or "費半已存本機，但雲端行情保存未完成")
    with closing(dp.get_conn()) as conn:
        conn.execute("DELETE FROM meta WHERE key=?", (PREFIX + "dirty:sox",))
        conn.commit()


def refresh_sox(raise_errors=True):
    """Save completed US sessions and restore CSV without consuming a download."""
    day = _now().date().isoformat()
    target = _sox_target()
    with _lease("sox") as acquired:
        if not acquired:
            raise RuntimeError("另一個畫面正在更新費半，未重複下載")
        with closing(dp.get_conn()) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS market_index_prices "
                         "(symbol TEXT NOT NULL,date TEXT NOT NULL,close REAL NOT NULL,PRIMARY KEY(symbol,date))")
            rows = conn.execute("SELECT date,close FROM market_index_prices WHERE symbol=? ORDER BY date",
                                ("^SOX",)).fetchall()
            dirty = _read(conn, "dirty:sox").get("pending")
            conn.commit()
        if rows:
            stored = pd.Series([r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows]), name="close")
            if stored.index.max() >= target:
                cached = mr._cached_sox()
                if cached is None or not cached.equals(stored):
                    mr._write_sox(stored)
                if dirty:
                    _save_sox_cloud()
                return stored
        cached = mr._cached_sox()
        # A cache read is not marked as a completed download. It may be the
        # restored cloud cache or a public seed already covering the target.
        if cached is not None and cached.index.max() >= target:
            series = cached
            downloaded = False
        else:
            series = (_download_sox() if mr.refresh_sox is _BASE_SOX_REFRESH
                      else mr.refresh_sox(raise_errors=raise_errors))
            downloaded = True
        if (series is None or series.empty or not np.isfinite(series.values).all()
                or not series.gt(0).all()):
            raise RuntimeError("費半無有效資料，未標記今日更新完成")
        if series.index.max().normalize() < target:
            raise RuntimeError("費半資料仍過期，保留舊資料並允許續抓")
        with closing(dp.get_conn()) as conn:
            conn.executemany("INSERT OR REPLACE INTO market_index_prices(symbol,date,close) VALUES (?,?,?)",
                             [("^SOX", str(date.date()), float(value)) for date, value in series.items()])
            if downloaded:
                _write(conn, "sox", {"day": day, "saved_at": _now().isoformat(),
                                    "target": str(target.date()), "data_date": str(series.index.max().date())})
            from core import market_cache_storage as storage
            if storage.configured():
                _write(conn, "dirty:sox", {"pending": True})
            conn.commit()
        mr._write_sox(series)
        _save_sox_cloud()
        return series
