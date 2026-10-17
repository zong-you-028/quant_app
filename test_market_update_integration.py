"""Cold Render instances recover public updates without a live remote service."""
from contextlib import closing
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pandas as pd
import pytest

from core import data_pipeline as dp, daily_market_update as daily, market_regime as mr
from core import market_cache_storage as storage, market_update_workflow as workflow
from test_market_cache_storage import FakeCloud


def quote(symbol, date="2026-10-15", chips=True):
    dp._merge_twse_prices(symbol, pd.DataFrame([{
        "symbol": symbol, "date": date, "open": 100., "high": 101.,
        "low": 99., "close": 100., "volume": 1000.}]))
    if chips and symbol != dp.config.BENCHMARK_SYMBOL:
        with closing(dp.get_conn()) as conn:
            conn.execute("INSERT OR REPLACE INTO chip_weekly VALUES (?,?,?,?)", (symbol, date, 40., 100.))
            conn.commit()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    base = tmp_path / "app"
    seed_folder = base / "data_seed"
    seed_folder.mkdir(parents=True)
    seed_db = tmp_path / "seed.db"
    with sqlite3.connect(seed_db) as conn:
        conn.executescript("""
            CREATE TABLE ohlcv(symbol TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,PRIMARY KEY(symbol,date));
            CREATE TABLE chip_weekly(symbol TEXT,date TEXT,big_shares REAL,total_shares REAL,PRIMARY KEY(symbol,date));
            CREATE TABLE stock_info(symbol TEXT PRIMARY KEY,name TEXT);
        """)
        for symbol in ("2330", "2317", "006208"):
            conn.execute("INSERT INTO ohlcv VALUES (?,?,?,?,?,?,?)", (symbol, "2026-10-02", 100., 101., 99., 100., 1000.))
            if symbol != "006208":
                conn.execute("INSERT INTO chip_weekly VALUES (?,?,?,?)", (symbol, "2026-10-02", 40., 100.))
    with gzip.open(seed_folder / "market.db.gz", "wb") as output:
        output.write(seed_db.read_bytes())
    local = tmp_path / "instance_a"
    local.mkdir()
    cloud = FakeCloud()
    monkeypatch.setenv("MARKET_DATABASE_URL", "postgresql://synthetic.invalid/public")
    monkeypatch.setenv("JOURNAL_DATABASE_URL", "")
    monkeypatch.setattr(dp.config, "BASE_DIR", str(base))
    monkeypatch.setattr(dp.config, "DATA_DIR", str(local))
    monkeypatch.setattr(dp.config, "DB_PATH", str(local / "market.db"))
    monkeypatch.setattr(dp.config, "DATA_SOURCE", "finmind")
    monkeypatch.setattr(dp.config, "ROTATION_FASTSELL_GATE", True)
    monkeypatch.setattr(dp, "_last_trading_day", lambda *args: pd.Timestamp("2026-10-15"))
    monkeypatch.setattr(daily, "_now", lambda: dt.datetime(2026, 10, 16, 19, tzinfo=daily.TAIPEI))
    monkeypatch.setattr(mr, "SOX_CSV", str(local / "sox.csv"))
    monkeypatch.setattr(mr, "refresh_sox", lambda **kwargs: pd.Series([5100.], index=pd.to_datetime(["2026-10-15"]), name="close"))
    monkeypatch.setattr(storage, "_open_cloud", lambda: cloud)
    monkeypatch.setattr(dp.time, "sleep", lambda *args: None)
    def denied(*args, **kwargs):
        raise AssertionError("Integration tests forbid live services")
    monkeypatch.setattr(dp.requests.sessions.Session, "request", denied)
    dp.ensure_db()  # Real seed hydration, not a mocked initializer.
    return {"cloud": cloud, "folder": tmp_path, "base": base}


def test_new_instance_restores_real_seed_updates_and_completions(cache, monkeypatch):
    downloaded = []
    monkeypatch.setattr(dp, "_refresh_market_data", lambda symbol: (downloaded.append(symbol), quote(symbol)))
    first = workflow.run_update(["2330", "2317", "006208"])
    assert first["updated"] == 3 and first["storage"]["durable"]
    assert downloaded == ["006208", "2330", "2317"]
    newer = cache["folder"] / "instance_b"
    newer.mkdir()
    monkeypatch.setattr(dp.config, "DB_PATH", str(newer / "market.db"))
    monkeypatch.setattr(mr, "SOX_CSV", str(newer / "sox.csv"))
    dp.ensure_db()
    assert dp.last_ohlcv_date("2330") == pd.Timestamp("2026-10-02")
    downloaded.clear()
    def no_download(symbol):
        pytest.fail("Restored current data must not be downloaded")
    monkeypatch.setattr(dp, "_refresh_market_data", no_download)
    second = workflow.run_update(["2330", "2317", "006208"])
    assert second["current"] == 3 and second["failed"] == second["pending"] == 0
    assert second["asof"] == second["sox_asof"] == "2026-10-15"
    assert second["restoration"]["restored"] == 4
    assert not downloaded and mr._cached_sox().index.max() == pd.Timestamp("2026-10-15")
    assert dp.last_chip_date("2330") == pd.Timestamp("2026-10-15")


def test_partial_price_commit_survives_budget_exception_before_progress_callback(cache, monkeypatch):
    def interrupted(symbol):
        quote(symbol, chips=False)
        raise dp.UpdateBudgetExceeded("chip request reached batch budget")
    monkeypatch.setattr(dp, "_refresh_market_data", interrupted)
    result = daily.update_symbols(["2330"], max_attempts=1)
    assert result["pending"] == 1
    saved = storage._decode(cache["cloud"].payloads["stock:2330"], "stock:2330")
    assert saved["ohlcv"][-1][0] == "2026-10-15"
    assert saved["chip_weekly"][-1][0] == "2026-10-02"
    assert not saved["completion"]


def test_failed_cloud_upload_retries_without_redownloading_current_prices(cache, monkeypatch):
    downloads = []
    monkeypatch.setattr(dp, "_refresh_market_data", lambda symbol: (downloads.append(symbol), quote(symbol)))
    cache["cloud"].fail.add("stock:2330")
    first = workflow.run_update(["2330"])
    assert first["updated"] == 1 and first["cloud_errors"]
    cache["cloud"].fail.clear()
    second = workflow.run_update(["2330"])
    assert second["current"] == 1 and not second["cloud_errors"]
    assert second["storage"]["durable"] and downloads == ["2330"]
    assert "stock:2330" in cache["cloud"].payloads


def test_cloud_failure_stops_new_downloads_but_next_run_resumes(cache, monkeypatch):
    downloads = []
    monkeypatch.setattr(dp, "_refresh_market_data", lambda symbol: (downloads.append(symbol), quote(symbol)))
    cache["cloud"].fail.add("stock:2330")
    first = workflow.run_update(["2330", "2317"])
    assert downloads == ["2330"] and first["pending_symbols"] == ["2317"]
    assert first["cloud_errors"]
    cache["cloud"].fail.clear()
    second = workflow.run_update(["2330", "2317"])
    assert downloads == ["2330", "2317"]
    assert second["current"] == second["updated"] == 1
    assert second["pending"] == 0 and not second["cloud_errors"]


def test_already_current_legacy_cache_is_adopted_without_new_http(cache, monkeypatch):
    quote("2330")
    monkeypatch.setattr(dp, "_refresh_market_data", lambda symbol: pytest.fail("No duplicate download"))
    result = workflow.run_update(["2330"])
    assert result["current"] == 1 and result["storage"]["durable"]
    assert storage._decode(cache["cloud"].payloads["stock:2330"], "stock:2330")["ohlcv"][-1][0] == "2026-10-15"


def test_failed_chip_only_request_does_not_republish_unchanged_history(cache, monkeypatch):
    quote("2330", chips=False)
    assert storage.save_symbols(["2330"])["durable"]
    calls = []
    original_save = cache["cloud"].save_bundle
    cache["cloud"].save_bundle = lambda key, bundle: (calls.append(key), original_save(key, bundle))[-1]
    def unavailable(symbol):
        raise dp.SourceUnavailableError("quota exhausted")
    monkeypatch.setattr(dp, "_refresh_market_data", unavailable)
    result = workflow.run_update(["2330"])
    assert result["failed"] == 1 and not result["cloud_errors"]
    assert "stock:2330" not in calls


def test_fresh_python_process_restores_public_cloud_without_actual_network(cache, monkeypatch):
    quote("2330")
    assert storage.save_symbols(["2330"])["durable"]
    cloud_file = cache["folder"] / "fake_cloud.json"
    cloud_file.write_text(json.dumps({key: value.hex() for key, value in cache["cloud"].payloads.items()}), encoding="utf-8")
    new_dir = cache["folder"] / "subprocess_instance"
    new_dir.mkdir()
    code = '''import json,sys,datetime as dt,pandas as pd
from core import data_pipeline as dp,daily_market_update as daily,market_cache_storage as storage
dp.config.BASE_DIR=sys.argv[1]
dp._last_trading_day=lambda *a:pd.Timestamp("2026-10-15")
daily._now=lambda:dt.datetime(2026,10,16,19,tzinfo=daily.TAIPEI)
def deny(*a,**k):raise AssertionError("no actual download or service")
dp._refresh_market_data=deny
dp.requests.sessions.Session.request=deny
class Cloud:
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def read_all(self):return [(k,bytes.fromhex(v)) for k,v in json.load(open(sys.argv[2])).items()]
 def save_bundle(self,*a):deny()
storage._open_cloud=lambda:Cloud()
restored=daily.restore_cache()
result=daily.update_symbols(["2330"],max_attempts=1)
assert restored["durable"] and result["current"]==1 and result["asof"]=="2026-10-15"
print(json.dumps({"restored":restored["restored"],"asof":result["asof"]}))
'''
    environment = {**os.environ, "APP_DATA_DIR": str(new_dir), "JOURNAL_DATABASE_URL": ""}
    process = subprocess.run([sys.executable, "-c", code, str(cache["base"]), str(cloud_file)],
                             env=environment, capture_output=True, text=True, check=True)
    assert json.loads(process.stdout) == {"restored": 1, "asof": "2026-10-15"}
