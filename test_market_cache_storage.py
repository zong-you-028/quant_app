"""Public-only market persistence against temporary SQLite and a fake cloud."""
from contextlib import contextmanager
import gzip
import json
import sqlite3
import sys
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from core import data_pipeline as dp, market_regime as mr
from core import market_cache_storage as storage


class FakeCloud:
    def __init__(self):
        self.payloads = {}
        self.fail = set()
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read_all(self):
        return list(self.payloads.items())

    def save_bundle(self, key, bundle):
        if key in self.fail:
            raise RuntimeError("postgresql://private-password@secret-host/db")
        with self.lock:
            if key in self.payloads:
                bundle = storage._merge_bundles(storage._decode(self.payloads[key], key), bundle)
            self.payloads[key] = storage._encode(bundle)


def initialize(path):
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv(symbol TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,PRIMARY KEY(symbol,date));
            CREATE TABLE IF NOT EXISTS chip_weekly(symbol TEXT,date TEXT,big_shares REAL,total_shares REAL,PRIMARY KEY(symbol,date));
            CREATE TABLE IF NOT EXISTS stock_info(symbol TEXT PRIMARY KEY,name TEXT);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
        """)


@pytest.fixture
def state(tmp_path, monkeypatch):
    state = SimpleNamespace(path=tmp_path / "local.db", cloud=FakeCloud(), sox=[])
    monkeypatch.setenv("MARKET_DATABASE_URL", "postgresql://fake-market.invalid/cache")
    monkeypatch.setenv("JOURNAL_DATABASE_URL", "postgresql://fake-journal.invalid/journal")
    monkeypatch.setattr(dp, "ensure_db", lambda: initialize(state.path))
    monkeypatch.setattr(dp, "get_conn", lambda: sqlite3.connect(state.path))
    monkeypatch.setattr(dp, "_chip_required", lambda symbol: symbol != "006208")
    monkeypatch.setattr(dp, "_last_trading_day", lambda *args: pd.Timestamp("2026-10-16"))
    monkeypatch.setattr(mr, "_write_sox", lambda series: state.sox.append(series.copy()))
    monkeypatch.setattr(storage, "_open_cloud", lambda: state.cloud)
    monkeypatch.setattr(dp.requests.sessions.Session, "request", lambda *a, **k: pytest.fail("network forbidden"))
    initialize(state.path)
    return state


def put_stock(state, symbol="2330", day="2026-10-15", close=100., chip="2026-10-15"):
    with sqlite3.connect(state.path) as conn:
        conn.execute("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?)", (symbol, day, close, close, close, close, 1000.))
        if chip:
            conn.execute("INSERT OR REPLACE INTO chip_weekly VALUES (?,?,?,?)", (symbol, chip, 20., 100.))
        conn.execute("INSERT OR REPLACE INTO stock_info VALUES (?,?)", (symbol, "公開股票"))


def put_meta(state, key, value):
    with sqlite3.connect(state.path) as conn:
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))


def done(target="2026-10-15"):
    return {"day": "2026-10-16", "target": target, "saved_at": "2026-10-16T10:00:00+08:00"}


def maximum(state, symbol="2330"):
    with sqlite3.connect(state.path) as conn:
        return conn.execute("SELECT MAX(date) FROM ohlcv WHERE symbol=?", (symbol,)).fetchone()[0]


def bundle(symbol="2330", day="2026-10-15", close=100., chip_day="2026-10-15", chip_value=20.):
    return {"schema": 1, "kind": "stock", "symbol": symbol,
            "ohlcv": [[day, close, close, close, close, 1000.]],
            "chip_weekly": [[chip_day, chip_value, 100.]] if chip_day else [],
            "name": "公開股票", "completion": {}, "backfill": {}}


def test_explicit_url_priority_and_journal_fallback_without_connecting(state, monkeypatch):
    assert storage.configured()
    assert storage._url() == "postgresql://fake-market.invalid/cache"
    monkeypatch.setenv("MARKET_DATABASE_URL", "")
    assert storage._url() == "postgresql://fake-journal.invalid/journal"
    monkeypatch.setenv("JOURNAL_DATABASE_URL", "")
    assert not storage.configured()
    monkeypatch.setattr(storage, "_open_cloud", lambda: pytest.fail("unconfigured must not connect"))
    assert not storage.save_symbols(["2330"])["durable"]
    assert not storage.restore_market_cache()["configured"]


def test_restart_new_local_directory_restores_oct15_instead_of_oct2_seed(state, tmp_path):
    put_stock(state, day="2026-10-02", close=90.)
    put_stock(state)
    put_meta(state, storage.STOCK_DONE + "2330", done())
    saved = storage.save_symbols(["2330", "2330"])
    assert saved["durable"] and saved["saved_symbols"] == ["2330"]
    state.path = tmp_path / "new-instance.db"
    initialize(state.path)
    put_stock(state, day="2026-10-02", close=90., chip="2026-10-02")
    result = storage.restore_market_cache()
    assert result["durable"] and result["restored_symbols"] == ["2330"]
    assert maximum(state) == "2026-10-15"
    with sqlite3.connect(state.path) as conn:
        restored = json.loads(conn.execute("SELECT value FROM meta WHERE key=?", (storage.STOCK_DONE + "2330",)).fetchone()[0])
    assert restored == done()


def test_stale_writer_cannot_rollback_remote_adjusted_history_and_chips_merge_independently(state):
    newer = bundle(day="2026-10-16", close=110., chip_day="2026-10-14", chip_value=20.)
    newer["ohlcv"].insert(0, ["2026-10-02", 80., 80., 80., 80., 1000.])
    state.cloud.save_bundle("stock:2330", newer)
    older = bundle(day="2026-10-15", close=100., chip_day="2026-10-15", chip_value=30.)
    older["ohlcv"].insert(0, ["2026-10-02", 90., 90., 90., 90., 1000.])
    state.cloud.save_bundle("stock:2330", older)
    remote = storage._decode(state.cloud.payloads["stock:2330"], "stock:2330")
    assert [row[0] for row in remote["ohlcv"]] == ["2026-10-02", "2026-10-15", "2026-10-16"]
    assert remote["ohlcv"][0][4] == 80.
    assert remote["chip_weekly"][-1] == ["2026-10-15", 30., 100.]


def test_equal_asof_later_save_updates_overlap_and_concurrent_disjoints_survive(state):
    a, b = bundle(day="2026-10-14", close=100.), bundle(day="2026-10-15", close=110.)
    threads = [threading.Thread(target=state.cloud.save_bundle, args=("stock:2330", item)) for item in (a, b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state.cloud.save_bundle("stock:2330", bundle(day="2026-10-15", close=120.))
    remote = storage._decode(state.cloud.payloads["stock:2330"], "stock:2330")
    assert [row[0] for row in remote["ohlcv"]] == ["2026-10-14", "2026-10-15"]
    assert remote["ohlcv"][-1][4] == 120.


def test_restore_preserves_newer_local_dataset_and_does_not_adjust_again(state):
    state.cloud.save_bundle("stock:2330", bundle(day="2026-10-15", close=90.))
    put_stock(state, day="2026-10-15", close=100.)
    put_stock(state, day="2026-10-16", close=120.)
    assert storage.restore_market_cache()["durable"]
    with sqlite3.connect(state.path) as conn:
        assert conn.execute("SELECT close FROM ohlcv WHERE symbol='2330' AND date='2026-10-15'").fetchone()[0] == 100.
    assert maximum(state) == "2026-10-16"


def test_private_tables_leases_and_extra_completion_fields_are_not_serialized(state):
    put_stock(state)
    sentinel = "PRIVATE_JOURNAL_SENTINEL"
    with sqlite3.connect(state.path) as conn:
        conn.execute("CREATE TABLE trades(secret TEXT)")
        conn.execute("INSERT INTO trades VALUES (?)", (sentinel,))
    put_meta(state, "app_daily_update:lease:stocks", {"owner": sentinel})
    put_meta(state, "unrelated_private_key", sentinel)
    put_meta(state, storage.STOCK_DONE + "2330", {**done(), "private": sentinel})
    assert storage.save_symbols(["2330"])["durable"]
    raw = gzip.decompress(state.cloud.payloads["stock:2330"]).decode("utf-8")
    assert sentinel not in raw and "trades" not in raw and "lease" not in raw
    assert set(json.loads(raw)) == storage.STOCK_FIELDS


def test_partial_cloud_failure_is_explicit_and_already_committed_resources_survive(state):
    put_stock(state)
    put_stock(state, symbol="2317")
    state.cloud.fail.add("stock:2317")
    result = storage.save_symbols(["2330", "2317"])
    assert not result["durable"] and result["saved_symbols"] == ["2330"]
    assert result["failed_symbols"] == ["2317"] and result["errors"]
    assert "private-password" not in json.dumps(result) and "secret-host" not in json.dumps(result)
    assert "stock:2330" in state.cloud.payloads and maximum(state) == "2026-10-15"


def test_connection_failure_does_not_expose_url_or_claim_durability(state, monkeypatch):
    put_stock(state)
    def unavailable():
        raise RuntimeError("postgresql://private-password@secret-host/db")
    monkeypatch.setattr(storage, "_open_cloud", unavailable)
    for result in (storage.save_symbols(["2330"]), storage.restore_market_cache(), storage.save_sox()):
        assert not result["durable"] and result["error"]
        assert "private-password" not in json.dumps(result) and "secret-host" not in json.dumps(result)


def test_corrupt_payload_is_explicit_and_does_not_block_other_public_resources(state):
    state.cloud.payloads["stock:2330"] = b"not gzip"
    state.cloud.save_bundle("stock:2317", bundle(symbol="2317"))
    result = storage.restore_market_cache()
    assert not result["durable"] and "stock:2330" in result["errors"]
    assert result["restored_symbols"] == ["2317"] and maximum(state, "2317") == "2026-10-15"


def test_payload_key_and_schema_whitelist_prevent_private_rows(state):
    invalid = bundle()
    invalid["trades"] = [{"secret": "never serialize"}]
    with pytest.raises(storage.CachePayloadError):
        storage._encode(invalid)
    state.cloud.payloads["stock:2317"] = storage._encode(bundle(symbol="2330"))
    assert not storage.restore_market_cache()["durable"]


def test_completion_is_not_restored_when_actual_prices_or_chips_are_stale(state):
    incomplete = bundle(day="2026-10-15", chip_day="2026-10-02")
    incomplete["completion"] = done()
    state.cloud.save_bundle("stock:2330", incomplete)
    result = storage.restore_market_cache()
    assert result["durable"]
    with sqlite3.connect(state.path) as conn:
        assert conn.execute("SELECT 1 FROM meta WHERE key=?", (storage.STOCK_DONE + "2330",)).fetchone() is None


def test_backfill_checkpoint_is_public_validated_and_restored_with_anchor(state, tmp_path):
    put_stock(state, day="2026-09-25", chip="2026-09-25")
    checkpoint = {"version": 1, "symbol": "2330", "stored_last": "2026-09-25", "completed_through": "2026-09-30"}
    put_meta(state, storage.BACKFILL + "2330", {**checkpoint, "private": "never export"})
    assert storage.save_symbols(["2330"])["durable"]
    remote = storage._decode(state.cloud.payloads["stock:2330"], "stock:2330")
    assert remote["backfill"] == checkpoint
    state.path = tmp_path / "new-backfill.db"
    initialize(state.path)
    assert storage.restore_market_cache()["durable"]
    with sqlite3.connect(state.path) as conn:
        restored = json.loads(conn.execute("SELECT value FROM meta WHERE key=?", (storage.BACKFILL + "2330",)).fetchone()[0])
    assert restored == checkpoint
    put_stock(state, day="2026-10-15")
    assert storage.save_symbols(["2330"])["durable"]
    assert not storage._decode(state.cloud.payloads["stock:2330"], "stock:2330")["backfill"]


@pytest.mark.parametrize("change", [
    {"symbol": "2317"}, {"version": True}, {"completed_through": "2026-10-31"},
    {"completed_through": "2026-09-29"}, {"completed_through": "2026-09-20"},
])
def test_invalid_backfill_metadata_is_not_serialized(state, change):
    put_stock(state, day="2026-09-25", chip="2026-09-25")
    checkpoint = {"version": 1, "symbol": "2330", "stored_last": "2026-09-25", "completed_through": "2026-09-30"}
    put_meta(state, storage.BACKFILL + "2330", {**checkpoint, **change})
    assert storage.save_symbols(["2330"])["durable"]
    assert not storage._decode(state.cloud.payloads["stock:2330"], "stock:2330")["backfill"]


def test_unknown_corrupt_key_never_appears_as_a_secret_in_status(state):
    state.cloud.payloads["postgres://secret@private"] = b"bad"
    result = storage.restore_market_cache()
    assert result["errors"] and "secret" not in json.dumps(result)


def test_sox_restore_rewrites_only_public_index_and_atomic_csv_hook(state, tmp_path):
    with sqlite3.connect(state.path) as conn:
        conn.execute("CREATE TABLE market_index_prices(symbol TEXT,date TEXT,close REAL,PRIMARY KEY(symbol,date))")
        conn.execute("INSERT INTO market_index_prices VALUES ('^SOX','2026-10-15',5000.)")
        conn.execute("INSERT INTO market_index_prices VALUES ('PRIVATE','2026-10-15',999.)")
    marker = {"day": "2026-10-16", "data_date": "2026-10-15", "saved_at": "2026-10-16T10:00:00+08:00"}
    put_meta(state, storage.SOX_DONE, marker)
    assert storage.save_sox()["durable"]
    raw = gzip.decompress(state.cloud.payloads["index:^SOX"]).decode("utf-8")
    assert "PRIVATE" not in raw
    state.path = tmp_path / "new-sox.db"
    initialize(state.path)
    restored = storage.restore_market_cache()
    assert restored["durable"] and restored["restored_sox"]
    assert state.sox[-1].loc[pd.Timestamp("2026-10-15")] == 5000.


def test_production_gateway_uses_conflict_insert_row_lock_and_per_resource_transaction(monkeypatch):
    class Connection:
        def __init__(self):
            self.payload = None
            self.sql = []
            self.commits = 0
        def execute(self, sql, params=()):
            self.sql.append(sql)
            if sql.startswith("INSERT") and self.payload is None:
                self.payload = params[1]
            if sql.startswith("UPDATE"):
                self.payload = params[0]
            return SimpleNamespace(fetchone=lambda: (self.payload,))
        def commit(self):
            self.commits += 1
        def close(self):
            pass
        @contextmanager
        def transaction(self):
            yield
            self.commits += 1
    conn = Connection()
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *a, **k: conn))
    with storage._PostgresGateway("postgresql://synthetic.invalid/test") as gateway:
        gateway.save_bundle("stock:2330", bundle())
        gateway.save_bundle("stock:2330", bundle(day="2026-10-16"))
    assert any("ON CONFLICT" in sql for sql in conn.sql)
    assert sum("FOR UPDATE" in sql for sql in conn.sql) == 2
    assert conn.commits == 3
    assert storage._decode(conn.payload, "stock:2330")["ohlcv"][-1][0] == "2026-10-16"


def test_neon_pooler_rejects_startup_options_but_gateway_reads_and_saves(monkeypatch):
    class PoolerConnection:
        def __init__(self):
            self.payload = None
            self.active = False
            self.local_settings = set()
            self.transactions = 0
            self.closed = False

        @contextmanager
        def transaction(self):
            assert not self.active
            self.active = True
            self.transactions += 1
            try:
                yield
            finally:
                self.active = False
                self.local_settings.clear()  # PgBouncer may switch servers now.

        def execute(self, sql, params=()):
            assert self.active, "All cache work must remain in an explicit transaction"
            if sql.startswith("SET LOCAL"):
                self.local_settings.add(sql)
            elif not sql.startswith("SET TRANSACTION"):
                assert self.local_settings == {
                    "SET LOCAL statement_timeout = '15s'", "SET LOCAL lock_timeout = '8s'"}
            if sql.startswith("INSERT") and self.payload is None:
                self.payload = params[1]
            if sql.startswith("UPDATE"):
                self.payload = params[0]
            return SimpleNamespace(fetchone=lambda: (self.payload,),
                                   fetchall=lambda: [("stock:2330", self.payload)])

        def close(self):
            self.closed = True

    conn = PoolerConnection()
    def connect(url, **kwargs):
        if "options" in kwargs:
            raise RuntimeError("unsupported startup parameter in options: statement_timeout")
        assert kwargs == {"connect_timeout": 8, "autocommit": True, "prepare_threshold": None}
        return conn
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    with storage._PostgresGateway("postgresql://synthetic-pooler.invalid/cache") as gateway:
        gateway.save_bundle("stock:2330", bundle())
        gateway.save_bundle("stock:2330", bundle(day="2026-10-16"))
        key, payload = gateway.read_all()[0]
        assert storage._decode(payload, key)["ohlcv"][-1][0] == "2026-10-16"
        assert not conn.active and not conn.local_settings
    assert conn.transactions == 4 and conn.closed


@pytest.mark.parametrize("error, code", [
    ("unsupported startup parameter in options: statement_timeout", "pooler_startup_parameter"),
    ("connection timeout expired", "database_timeout"),
    ("password authentication failed", "authentication_failed"),
    ("endpoint is suspended", "database_suspended"),
    ("certificate verify failed", "tls_verification_failed"),
])
def test_cloud_error_classification_never_logs_credentials(state, monkeypatch, capsys, error, code):
    def unavailable():
        raise RuntimeError(error + " postgresql://private-password@secret-host/db")
    monkeypatch.setattr(storage, "_open_cloud", unavailable)
    result = storage.restore_market_cache()
    output = capsys.readouterr().out
    assert json.loads(output)["error_code"] == code
    assert not result["durable"] and result["error"]
    assert "private-password" not in output + json.dumps(result)
    assert "secret-host" not in output + json.dumps(result)


def test_startup_probe_is_read_only_and_reports_cache_access_without_secrets(monkeypatch, capsys):
    payload = storage._encode(bundle())
    calls = []
    class Connection:
        @contextmanager
        def transaction(self):
            yield
        def execute(self, sql, params=()):
            calls.append(sql)
            if sql == "SELECT 1":
                row = (1,)
            elif "to_regclass" in sql:
                row = (storage.TABLE,)
            elif "has_table_privilege" in sql:
                row = (True, True, True)
            elif "count(*)" in sql:
                row = (1,)
            else:
                row = ("stock:2330", payload)
            return SimpleNamespace(fetchone=lambda: row)
        def close(self):
            calls.append("CLOSE")
    monkeypatch.setenv("JOURNAL_DATABASE_URL", "postgresql://private-password@secret-host/db")
    monkeypatch.setattr(storage, "_connect", lambda url: Connection())
    result = storage.log_cloud_connectivity()
    assert result["connected"] and result["cache_writable"] and result["sample_payload_valid"]
    assert result["public_resources"] == 1
    assert calls[0] == "SET TRANSACTION READ ONLY"
    assert all(sql.startswith(("SET ", "SELECT ", "CLOSE")) for sql in calls)
    output = capsys.readouterr().out
    assert "private-password" not in output and "secret-host" not in output


def test_startup_probe_reports_connection_failure_without_raising(monkeypatch, capsys):
    monkeypatch.setenv("JOURNAL_DATABASE_URL", "postgresql://private-password@secret-host/db")
    def unavailable(url):
        raise RuntimeError("connection timeout: " + url)
    monkeypatch.setattr(storage, "_connect", unavailable)
    result = storage.log_cloud_connectivity()
    assert not result["connected"] and result["error_code"] == "database_timeout"
    output = capsys.readouterr().out
    assert "private-password" not in output and "secret-host" not in output
