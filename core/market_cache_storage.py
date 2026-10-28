"""Durable public-market bundles; never serialize the application database.

The app keeps SQLite as its read cache. Only explicitly listed public quote,
shareholding, name, SOX, and updater-checkpoint fields cross this boundary.
Importing this module neither loads the app configuration nor connects anywhere.
"""
from contextlib import closing, contextmanager
import calendar
import datetime as dt
import gzip
import io
import json
import math
import os
import re


TABLE = "quant_market_cache_v1"
MAX_PAYLOAD = 8 * 1024 * 1024
STOCK_FIELDS = {"schema", "kind", "symbol", "ohlcv", "chip_weekly", "name", "completion", "backfill"}
SOX_FIELDS = {"schema", "kind", "symbol", "market_index_prices", "completion"}
STOCK_DONE = "app_daily_update:stock:"
SOX_DONE = "app_daily_update:sox"
BACKFILL = "app_backfill:"


class CachePayloadError(ValueError):
    pass


def _url():
    return (os.environ.get("MARKET_DATABASE_URL", "").strip()
            or os.environ.get("JOURNAL_DATABASE_URL", "").strip())


def configured():
    """Configuration presence only; does not test or expose the connection URL."""
    return bool(_url())


def _symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z0-9]{2,16}", value):
        raise CachePayloadError("invalid public symbol")
    return value


def _date(value):
    if not isinstance(value, str) or dt.date.fromisoformat(value).isoformat() != value:
        raise CachePayloadError("invalid public date")
    return value


def _number(value, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CachePayloadError("invalid public number")
    return value


def _completion(value, kind, strict=True):
    fields = {"day", "saved_at", "target" if kind == "stock" else "data_date"}
    if not value:
        return {}
    if not isinstance(value, dict) or (strict and set(value) != fields):
        raise CachePayloadError("invalid public completion")
    result = {key: value[key] for key in fields if key in value}
    if set(result) != fields:
        raise CachePayloadError("incomplete public completion")
    _date(result["day"])
    _date(result["target" if kind == "stock" else "data_date"])
    if not isinstance(result["saved_at"], str) or len(result["saved_at"]) > 64:
        raise CachePayloadError("invalid completion timestamp")
    dt.datetime.fromisoformat(result["saved_at"])
    return result


def _backfill(value, strict=True, symbol=None):
    if not value:
        return {}
    fields = {"version", "symbol", "completed_through", "stored_last"}
    if not isinstance(value, dict) or (strict and set(value) != fields):
        raise CachePayloadError("unrecognized public backfill checkpoint")
    result = {key: value[key] for key in fields if key in value}
    if set(result) != fields or type(result["version"]) is not int or result["version"] != 1:
        raise CachePayloadError("invalid public backfill version")
    _symbol(result["symbol"])
    if symbol is not None and result["symbol"] != symbol:
        raise CachePayloadError("public backfill symbol mismatch")
    through = dt.date.fromisoformat(_date(result["completed_through"]))
    last = dt.date.fromisoformat(_date(result["stored_last"]))
    if through < last or (through.year, through.month) != (last.year, last.month):
        raise CachePayloadError("invalid public backfill interval")
    if through > last and through.day != calendar.monthrange(through.year, through.month)[1]:
        raise CachePayloadError("only a completed month may have an empty tail")
    return result


def _valid_backfill(checkpoint, last, expected):
    if not checkpoint or checkpoint["stored_last"] != last or checkpoint["completed_through"] > expected:
        return False
    return (checkpoint["completed_through"] == last
            or checkpoint["completed_through"][:7] < expected[:7])


def _validate(bundle, expected_key=None):
    if not isinstance(bundle, dict) or type(bundle.get("schema")) is not int or bundle.get("schema") != 1:
        raise CachePayloadError("invalid public bundle schema")
    kind = bundle.get("kind")
    if kind == "stock":
        if set(bundle) != STOCK_FIELDS:
            raise CachePayloadError("unrecognized public stock fields")
        symbol = _symbol(bundle.get("symbol"))
        key = "stock:" + symbol
        specifications = (("ohlcv", 6), ("chip_weekly", 3))
        if bundle["name"] is not None and (not isinstance(bundle["name"], str) or len(bundle["name"]) > 200):
            raise CachePayloadError("invalid public stock name")
        _backfill(bundle["backfill"], symbol=symbol)
    elif kind == "sox":
        if set(bundle) != SOX_FIELDS or bundle.get("symbol") != "^SOX":
            raise CachePayloadError("unrecognized public SOX fields")
        key = "index:^SOX"
        specifications = (("market_index_prices", 2),)
    else:
        raise CachePayloadError("unknown public bundle kind")
    if expected_key is not None and key != expected_key:
        raise CachePayloadError("public cache key mismatch")
    for table, width in specifications:
        rows = bundle[table]
        if not isinstance(rows, list):
            raise CachePayloadError("invalid public rows")
        seen = set()
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != width:
                raise CachePayloadError("invalid public row fields")
            day = _date(row[0])
            if day in seen:
                raise CachePayloadError("duplicate public dates")
            seen.add(day)
            for number in row[1:]:
                _number(number, nullable=table == "ohlcv")
            if table == "ohlcv" and (row[4] is None or row[4] <= 0):
                raise CachePayloadError("invalid public close")
            if table == "chip_weekly" and (row[1] < 0 or row[2] <= 0 or row[1] > row[2]):
                raise CachePayloadError("invalid public shareholding")
            if table == "market_index_prices" and row[1] <= 0:
                raise CachePayloadError("invalid public index close")
    _completion(bundle["completion"], kind)
    return bundle


def _encode(bundle):
    _validate(bundle)
    raw = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_PAYLOAD:
        raise CachePayloadError("public bundle too large")
    return gzip.compress(raw, mtime=0)


def _decode(payload, key):
    try:
        if len(payload) > MAX_PAYLOAD:
            raise CachePayloadError("public bundle too large")
        with gzip.GzipFile(fileobj=io.BytesIO(bytes(payload))) as source:
            raw = source.read(MAX_PAYLOAD + 1)
        if len(raw) > MAX_PAYLOAD:
            raise CachePayloadError("public bundle too large")
        return _validate(json.loads(raw.decode("utf-8")), key)
    except Exception as exc:
        raise CachePayloadError("invalid public cache payload") from exc


def _merge_rows(existing, incoming):
    """Newer dataset wins overlaps; union retains disjoint older observations."""
    old_max = max((row[0] for row in existing), default="")
    new_max = max((row[0] for row in incoming), default="")
    earlier, later = (existing, incoming) if new_max >= old_max else (incoming, existing)
    result = {row[0]: list(row) for row in earlier}
    result.update({row[0]: list(row) for row in later})
    return [result[day] for day in sorted(result)]


def _newer_completion(*values):
    nonempty = [value for value in values if value]
    return max(nonempty, key=lambda value: (value["day"], value.get("target", value.get("data_date", "")),
                                             value["saved_at"]), default={})


def _merge_bundles(existing, incoming):
    _validate(existing)
    _validate(incoming)
    if (existing["kind"], existing["symbol"]) != (incoming["kind"], incoming["symbol"]):
        raise CachePayloadError("different public resources")
    result = dict(incoming)
    datasets = ("ohlcv", "chip_weekly") if incoming["kind"] == "stock" else ("market_index_prices",)
    for dataset in datasets:
        result[dataset] = _merge_rows(existing[dataset], incoming[dataset])
    result["completion"] = _newer_completion(existing["completion"], incoming["completion"])
    if incoming["kind"] == "stock":
        old_max = max((row[0] for row in existing["ohlcv"]), default="")
        new_max = max((row[0] for row in incoming["ohlcv"]), default="")
        preferred, other = (incoming, existing) if new_max >= old_max else (existing, incoming)
        result["name"] = preferred["name"] or other["name"]
        last = max((row[0] for row in result["ohlcv"]), default="")
        checkpoints = [item["backfill"] for item in (existing, incoming)
                       if item["backfill"] and item["backfill"]["stored_last"] == last]
        result["backfill"] = max(checkpoints, key=lambda item: item["completed_through"], default={})
    return _validate(result)


def _connect(url):
    import psycopg
    # Neon/PgBouncer rejects statement_timeout/lock_timeout in startup options.
    # Bound each transaction instead; session SETs also cannot survive pooling.
    return psycopg.connect(url, connect_timeout=8, autocommit=True,
                           prepare_threshold=None)


@contextmanager
def _bounded_transaction(conn, *, read_only=False):
    with conn.transaction():
        if read_only:
            conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SET LOCAL statement_timeout = '15s'")
        conn.execute("SET LOCAL lock_timeout = '8s'")
        yield


class _PostgresGateway:
    def __init__(self, url):
        self._url = url
        self._conn = None

    def __enter__(self):
        try:
            self._conn = _connect(self._url)
            with _bounded_transaction(self._conn):
                self._conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ("
                                   "cache_key TEXT PRIMARY KEY,payload BYTEA NOT NULL,"
                                   "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            return self
        except Exception:
            if self._conn is not None:
                self._conn.close()
            raise

    def __exit__(self, *args):
        self._conn.close()

    def read_all(self):
        with _bounded_transaction(self._conn, read_only=True):
            return self._conn.execute(f"SELECT cache_key,payload FROM {TABLE} ORDER BY cache_key").fetchall()

    def save_bundle(self, key, bundle):
        # INSERT also serializes the first writers, before a row exists to lock.
        with _bounded_transaction(self._conn):
            self._conn.execute(f"INSERT INTO {TABLE}(cache_key,payload) VALUES (%s,%s) "
                               "ON CONFLICT(cache_key) DO NOTHING", (key, _encode(bundle)))
            row = self._conn.execute(f"SELECT payload FROM {TABLE} WHERE cache_key=%s FOR UPDATE", (key,)).fetchone()
            merged = _merge_bundles(_decode(row[0], key), bundle)
            self._conn.execute(f"UPDATE {TABLE} SET payload=%s,updated_at=CURRENT_TIMESTAMP WHERE cache_key=%s",
                               (_encode(merged), key))


def _open_cloud():
    return _PostgresGateway(_url())


def _error_code(exc):
    if isinstance(exc, CachePayloadError):
        return "invalid_payload"
    # Inspect locally, but never return or log driver text (URLs may be in it).
    message = str(exc).lower()
    state = getattr(exc, "sqlstate", None)
    if "unsupported startup parameter" in message:
        return "pooler_startup_parameter"
    if state == "42501":
        return "permission_denied"
    if state == "28P01" or "password authentication failed" in message:
        return "authentication_failed"
    if any(term in message for term in ("quota", "compute time", "suspended")):
        return "database_suspended"
    if any(term in message for term in ("certificate verify", "ssl error", "tls error")):
        return "tls_verification_failed"
    if "timeout" in message or "timed out" in message or state in ("57014", "55P03"):
        return "database_timeout"
    if state and state.startswith("08"):
        return "connection_failed"
    return "storage_failed"


def log_cloud_connectivity():
    """Read-only startup probe using the service's existing environment only."""
    report = {"event": "market_cache_connectivity", "configured": configured(),
              "connected": False}
    if report["configured"]:
        try:
            with closing(_connect(_url())) as conn, _bounded_transaction(conn, read_only=True):
                assert conn.execute("SELECT 1").fetchone()[0] == 1
                report["connected"] = True
                exists = conn.execute("SELECT to_regclass(%s)", (TABLE,)).fetchone()[0]
                report["cache_table_exists"] = bool(exists)
                if exists:
                    permissions = conn.execute(
                        "SELECT has_table_privilege(%s, 'SELECT'),"
                        "has_table_privilege(%s, 'INSERT'),has_table_privilege(%s, 'UPDATE')",
                        (TABLE, TABLE, TABLE)).fetchone()
                    report["cache_readable"] = bool(permissions[0])
                    report["cache_writable"] = all(permissions[1:])
                    if report["cache_readable"]:
                        report["public_resources"] = conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
                        sample = conn.execute(f"SELECT cache_key,payload FROM {TABLE} ORDER BY cache_key LIMIT 1").fetchone()
                        if sample:
                            _decode(sample[1], sample[0])
                            report["sample_payload_valid"] = True
        except Exception as exc:
            report["error_code"] = _error_code(exc)
    print(json.dumps(report), flush=True)
    return report


def _has_table(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _meta(conn, key):
    if not _has_table(conn, "meta"):
        return {}
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else {}


def _valid_done(bundle, completion, require_chips=False):
    if not completion:
        return False
    target = completion.get("target", completion.get("data_date"))
    dataset = "ohlcv" if bundle["kind"] == "stock" else "market_index_prices"
    latest = max((row[0] for row in bundle[dataset]), default="")
    if latest < target:
        return False
    if require_chips:
        chip_latest = max((row[0] for row in bundle["chip_weekly"]), default="")
        return bool(chip_latest and dt.date.fromisoformat(chip_latest) + dt.timedelta(days=7)
                    >= dt.date.fromisoformat(target))
    return True


def _read_stock(conn, symbol):
    from core import data_pipeline as dp
    bundle = {"schema": 1, "kind": "stock", "symbol": symbol,
              "ohlcv": [list(row) for row in conn.execute(
                  "SELECT date,open,high,low,close,volume FROM ohlcv WHERE symbol=? ORDER BY date", (symbol,))],
              "chip_weekly": [list(row) for row in conn.execute(
                  "SELECT date,big_shares,total_shares FROM chip_weekly WHERE symbol=? ORDER BY date", (symbol,))],
              "name": None, "completion": {}, "backfill": {}}
    row = conn.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,)).fetchone()
    bundle["name"] = row[0] if row else None
    try:
        done = _completion(_meta(conn, STOCK_DONE + symbol), "stock", strict=False)
        if _valid_done(bundle, done, dp._chip_required(symbol)):
            bundle["completion"] = done
    except (ValueError, TypeError, KeyError):
        pass
    try:
        checkpoint = _backfill(_meta(conn, BACKFILL + symbol), strict=False, symbol=symbol)
        last = max((row[0] for row in bundle["ohlcv"]), default="")
        if _valid_backfill(checkpoint, last, dp._last_trading_day().date().isoformat()):
            bundle["backfill"] = checkpoint
    except (ValueError, TypeError, KeyError):
        pass
    return _validate(bundle)


def _read_sox(conn):
    rows = conn.execute("SELECT date,close FROM market_index_prices WHERE symbol=? ORDER BY date", ("^SOX",)).fetchall() if _has_table(conn, "market_index_prices") else []
    bundle = {"schema": 1, "kind": "sox", "symbol": "^SOX",
              "market_index_prices": [list(row) for row in rows], "completion": {}}
    try:
        done = _completion(_meta(conn, SOX_DONE), "sox", strict=False)
        if _valid_done(bundle, done):
            bundle["completion"] = done
    except (ValueError, TypeError, KeyError):
        pass
    return _validate(bundle)


def _write_meta(conn, key, value):
    if value:
        conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)")
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", (key, json.dumps(value)))


def _restore_stock(conn, remote):
    from core import data_pipeline as dp
    symbol = remote["symbol"]
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        # Local incoming wins equal-date ties; a remote later dataset still wins.
        merged = _merge_bundles(remote, _read_stock(conn, symbol))
        conn.executemany("INSERT OR REPLACE INTO ohlcv(symbol,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",
                         [(symbol, *row) for row in merged["ohlcv"]])
        conn.executemany("INSERT OR REPLACE INTO chip_weekly(symbol,date,big_shares,total_shares) VALUES (?,?,?,?)",
                         [(symbol, *row) for row in merged["chip_weekly"]])
        if merged["name"]:
            conn.execute("INSERT OR REPLACE INTO stock_info(symbol,name) VALUES (?,?)", (symbol, merged["name"]))
        if _valid_done(merged, merged["completion"], dp._chip_required(symbol)):
            _write_meta(conn, STOCK_DONE + symbol, merged["completion"])
        checkpoint = merged["backfill"]
        last = max((row[0] for row in merged["ohlcv"]), default="")
        if _valid_backfill(checkpoint, last, dp._last_trading_day().date().isoformat()):
            _write_meta(conn, BACKFILL + symbol, checkpoint)


def _restore_sox(conn, remote):
    import pandas as pd
    from core import market_regime as mr
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        merged = _merge_bundles(remote, _read_sox(conn))
        conn.execute("CREATE TABLE IF NOT EXISTS market_index_prices (symbol TEXT NOT NULL,date TEXT NOT NULL,close REAL NOT NULL,PRIMARY KEY(symbol,date))")
        conn.executemany("INSERT OR REPLACE INTO market_index_prices(symbol,date,close) VALUES (?,?,?)",
                         [("^SOX", *row) for row in merged["market_index_prices"]])
        if _valid_done(merged, merged["completion"]):
            _write_meta(conn, SOX_DONE, merged["completion"])
    rows = merged["market_index_prices"]
    if rows:
        mr._write_sox(pd.Series([row[1] for row in rows], index=pd.to_datetime([row[0] for row in rows]), name="close"))


def _result():
    return {"configured": configured(), "durable": False, "saved": 0, "saved_symbols": [],
            "failed_symbols": [], "restored": 0, "restored_symbols": [], "restored_sox": False,
            "errors": {}, "error": None}


def _error(result, key, exc):
    # Never copy driver exceptions: they can contain credentials or source rows.
    code = _error_code(exc)
    message = {
        "invalid_payload": "雲端行情快取格式無效，未套用此筆資料",
        "pooler_startup_parameter": "雲端資料庫連線參數不相容，請更新部署版本",
        "permission_denied": "雲端行情資料庫權限不足，請檢查快取表讀寫權限",
        "authentication_failed": "雲端資料庫驗證失敗，請檢查服務的連線設定",
        "database_suspended": "雲端資料庫暫停或額度不足，請檢查資料庫服務狀態",
        "tls_verification_failed": "雲端資料庫安全連線驗證失敗，請檢查憑證設定",
        "database_timeout": "雲端資料庫連線或查詢逾時，本機資料仍保留，可稍後重試",
    }.get(code, "雲端行情保存或還原失敗；本機資料仍保留，可稍後重試")
    result["errors"][key] = message
    result["error"] = message
    safe_key = key if re.fullmatch(r"(?:stock:)?[A-Z0-9]{2,16}|index:\^SOX|\^SOX|cloud", key) else "resource"
    print(json.dumps({"event": "market_cache_error", "resource": safe_key,
                      "error_code": code}), flush=True)


def save_symbols(symbols):
    """Persist each committed symbol independently; return explicit safe status."""
    result = _result()
    if not result["configured"]:
        result["error"] = "未設定雲端行情儲存，目前僅保存於本機"
        return result
    syms = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if symbol))
    try:
        from core import data_pipeline as dp
        dp.ensure_db()
        with closing(dp.get_conn()) as local, _open_cloud() as cloud:
            for symbol in syms:
                try:
                    _symbol(symbol)
                    bundle = _read_stock(local, symbol)
                    if not bundle["ohlcv"]:
                        raise CachePayloadError("no public quotes")
                    cloud.save_bundle("stock:" + symbol, bundle)
                    result["saved_symbols"].append(symbol)
                except Exception as exc:
                    result["failed_symbols"].append(symbol)
                    _error(result, symbol, exc)
    except Exception as exc:
        result["failed_symbols"] = [symbol for symbol in syms if symbol not in result["saved_symbols"]]
        _error(result, "cloud", exc)
    result["saved"] = len(result["saved_symbols"])
    result["durable"] = not result["errors"]
    return result


def save_sox():
    result = _result()
    if not result["configured"]:
        result["error"] = "未設定雲端行情儲存，目前僅保存於本機"
        return result
    try:
        from core import data_pipeline as dp
        dp.ensure_db()
        with closing(dp.get_conn()) as local, _open_cloud() as cloud:
            bundle = _read_sox(local)
            if not bundle["market_index_prices"]:
                raise CachePayloadError("no public SOX rows")
            cloud.save_bundle("index:^SOX", bundle)
        result["saved"] = 1
        result["durable"] = True
    except Exception as exc:
        _error(result, "^SOX", exc)
    return result


def restore_market_cache():
    """Merge allowed cloud resources into SQLite before app freshness checks."""
    result = _result()
    if not result["configured"]:
        result["error"] = "未設定雲端行情儲存，目前僅保存於本機"
        return result
    try:
        from core import data_pipeline as dp
        dp.ensure_db()
        with closing(dp.get_conn()) as local, _open_cloud() as cloud:
            for key, payload in cloud.read_all():
                try:
                    bundle = _decode(payload, key)
                    if bundle["kind"] == "stock":
                        _restore_stock(local, bundle)
                        result["restored_symbols"].append(bundle["symbol"])
                    else:
                        _restore_sox(local, bundle)
                        result["restored_sox"] = True
                    result["restored"] += 1
                except Exception as exc:
                    safe_key = key if isinstance(key, str) and (re.fullmatch(r"stock:[A-Z0-9]{2,16}", key)
                                                               or key == "index:^SOX") else "payload"
                    _error(result, safe_key, exc)
        result["durable"] = not result["errors"]
    except Exception as exc:
        _error(result, "cloud", exc)
    return result
