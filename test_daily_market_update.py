"""Daily policy tests use disposable market databases and simulated sources."""
from contextlib import closing
import datetime as dt
import json
import subprocess
import sys

import pandas as pd
import pytest

from core import daily_market_update as daily
from core import data_pipeline as dp, market_regime as mr


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = str(tmp_path / "market.db")
    monkeypatch.setattr(dp.config, "DB_PATH", path)
    monkeypatch.setattr(dp.config, "DATA_SOURCE", "finmind")
    monkeypatch.setattr(dp.config, "ROTATION_FASTSELL_GATE", True)
    monkeypatch.setattr(dp, "_hydrate_market_seed", lambda conn: None)
    monkeypatch.setattr(dp, "_last_trading_day", lambda *a: pd.Timestamp("2026-10-06"))
    monkeypatch.setattr(dp.time, "sleep", lambda *a: None)
    monkeypatch.setattr(daily, "_now", lambda: dt.datetime(2026, 10, 7, 10, tzinfo=daily.TAIPEI))
    monkeypatch.setattr(mr, "SOX_CSV", str(tmp_path / "sox.csv"))
    def deny(*a, **k):
        raise AssertionError("No real HTTP in daily update tests")
    monkeypatch.setattr(dp.requests.sessions.Session, "request", deny)
    dp.ensure_db()
    return path


def commit_quote(symbol, date="2026-10-06"):
    frame = pd.DataFrame([dict(symbol=symbol, date=date, open=100., high=101.,
                               low=99., close=100., volume=1000.)])
    dp._merge_twse_prices(symbol, frame)
    with closing(dp.get_conn()) as conn:
        conn.execute("INSERT OR REPLACE INTO chip_weekly VALUES (?,?,?,?)", (symbol, date, 40., 100.))
        conn.commit()


def install_source(monkeypatch):
    calls = []
    def refresh(symbol):
        calls.append(symbol)
        commit_quote(symbol)
    monkeypatch.setattr(dp, "_refresh_market_data", refresh)
    return calls


def test_latest_quotes_and_chips_are_committed_and_same_day_force_cannot_redownload(store, monkeypatch):
    calls = install_source(monkeypatch)
    first = daily.update_symbols(["2330", "2330"], max_attempts=1)
    second = daily.update_symbols(["2330"], force=True, ignore_throttle=True, max_attempts=1)
    assert calls == ["2330"]
    assert first["updated"] == 1 and first["asof"] == "2026-10-06"
    assert second["current"] == 1 and second["daily_skipped_symbols"] == ["2330"]
    with closing(dp.get_conn()) as conn:
        assert conn.execute("SELECT date,close FROM ohlcv WHERE symbol='2330'").fetchall() == [("2026-10-06", 100.)]
        assert conn.execute("SELECT date,big_shares FROM chip_weekly WHERE symbol='2330'").fetchall() == [("2026-10-06", 40.)]


def test_daily_completion_survives_fresh_python_process(store, monkeypatch):
    install_source(monkeypatch)
    daily.update_symbols(["2330"], max_attempts=1)
    code = '''import sys,json,datetime as dt,pandas as pd
from core import data_pipeline as dp,daily_market_update as daily
dp.config.DB_PATH=sys.argv[1]
dp.config.DATA_SOURCE="finmind"
dp._hydrate_market_seed=lambda conn:None
dp._last_trading_day=lambda *a:pd.Timestamp("2026-10-06")
daily._now=lambda:dt.datetime(2026,10,7,10,tzinfo=daily.TAIPEI)
def deny(*a,**k):raise AssertionError("restart must not download")
dp._refresh_market_data=deny
dp.requests.sessions.Session.request=deny
print(json.dumps(daily.update_symbols(["2330"],force=True,max_attempts=1)))
'''
    proc = subprocess.run([sys.executable, "-c", code, store], capture_output=True, text=True, check=True)
    result = json.loads(proc.stdout)
    assert result["current"] == 1 and result["daily_skipped_symbols"] == ["2330"]


def test_failure_can_resume_without_touching_successes(store, monkeypatch):
    calls = []
    failed = [True]
    def source(symbol):
        calls.append(symbol)
        if symbol == "2317" and failed[0]:
            raise dp.SourceUnavailableError("source down")
        commit_quote(symbol)
    monkeypatch.setattr(dp, "_refresh_market_data", source)
    first = daily.update_symbols(["2330", "2317"], max_attempts=1)
    failed[0] = False
    second = daily.update_symbols(["2330", "2317"], max_attempts=1)
    assert first["updated"] == first["failed"] == 1
    assert second["updated"] == second["current"] == 1 and second["failed"] == 0
    assert calls == ["2330", "2317", "2317"]


def test_next_taipei_day_allows_new_data(store, monkeypatch):
    calls = install_source(monkeypatch)
    daily.update_symbols(["2330"], max_attempts=1)
    monkeypatch.setattr(daily, "_now", lambda: dt.datetime(2026, 10, 8, 19, tzinfo=daily.TAIPEI))
    monkeypatch.setattr(dp, "_last_trading_day", lambda *a: pd.Timestamp("2026-10-07"))
    monkeypatch.setattr(dp, "_refresh_market_data", lambda s: (calls.append(s), commit_quote(s, "2026-10-07")))
    result = daily.update_symbols(["2330"], max_attempts=1)
    assert calls == ["2330", "2330"] and result["asof"] == "2026-10-07"


def test_same_day_new_target_stays_truthfully_stale_until_next_day(store, monkeypatch):
    calls = install_source(monkeypatch)
    daily.update_symbols(["2330"], max_attempts=1)
    monkeypatch.setattr(dp, "_last_trading_day", lambda *a: pd.Timestamp("2026-10-07"))
    result = daily.update_symbols(["2330"], force=True, max_attempts=1)
    assert calls == ["2330"] and result["stale"] == 1
    assert result["asof"] == "2026-10-06" and result["expected_asof"] == "2026-10-07"


def test_reading_current_cache_does_not_consume_daily_download(store, monkeypatch):
    commit_quote("2330")
    calls = install_source(monkeypatch)
    result = daily.update_symbols(["2330"], max_attempts=1)
    assert result["current"] == 1 and not calls
    monkeypatch.setattr(dp, "_last_trading_day", lambda *a: pd.Timestamp("2026-10-07"))
    monkeypatch.setattr(dp, "_refresh_market_data", lambda s: (calls.append(s), commit_quote(s, "2026-10-07")))
    assert daily.update_symbols(["2330"], max_attempts=1)["updated"] == 1


def test_db_lease_blocks_another_session_and_is_released(store, monkeypatch):
    calls = install_source(monkeypatch)
    with daily._lease("stocks") as acquired:
        assert acquired
        assert daily.update_symbols(["2330"])["busy"]
        assert not calls
    assert daily.update_symbols(["2330"], max_attempts=1)["updated"] == 1


def test_stale_response_does_not_mark_success(store, monkeypatch):
    calls = []
    monkeypatch.setattr(dp, "_refresh_market_data", lambda s: (calls.append(s), commit_quote(s, "2026-10-05")))
    assert daily.update_symbols(["2330"], max_attempts=1)["stale"] == 1
    monkeypatch.setattr(dp, "_refresh_market_data", lambda s: (calls.append(s), commit_quote(s)))
    assert daily.update_symbols(["2330"], max_attempts=1)["updated"] == 1
    assert calls == ["2330", "2330"]


def test_sox_persists_in_sqlite_and_is_not_refetched_after_csv_loss(store, monkeypatch):
    calls = []
    series = pd.Series([5000., 5100.], index=pd.to_datetime(["2026-10-05", "2026-10-06"]), name="close")
    monkeypatch.setattr(mr, "refresh_sox", lambda **k: calls.append(1) or series)
    first = daily.refresh_sox()
    second = daily.refresh_sox()
    assert calls == [1] and first.equals(second)
    assert mr._cached_sox().equals(series)
    with closing(dp.get_conn()) as conn:
        assert conn.execute("SELECT date,close FROM market_index_prices ORDER BY date").fetchall() == [("2026-10-05", 5000.), ("2026-10-06", 5100.)]


def test_sox_failure_is_not_daily_success(store, monkeypatch):
    monkeypatch.setattr(mr, "refresh_sox", lambda **k: None)
    with pytest.raises(RuntimeError, match="無有效"):
        daily.refresh_sox()
    series = pd.Series([5100.], index=pd.to_datetime(["2026-10-06"]), name="close")
    monkeypatch.setattr(mr, "refresh_sox", lambda **k: series)
    assert daily.refresh_sox().equals(series)


def test_sox_expired_response_is_not_daily_success(store, monkeypatch):
    monkeypatch.setattr(mr, "refresh_sox", lambda **k: pd.Series([5000.], index=pd.to_datetime(["2026-10-01"])))
    with pytest.raises(RuntimeError, match="仍過期"):
        daily.refresh_sox()


def test_sox_busy_session_does_not_download(store, monkeypatch):
    monkeypatch.setattr(mr, "refresh_sox", lambda **k: pytest.fail("must not download"))
    with daily._lease("sox"):
        with pytest.raises(RuntimeError, match="未重複"):
            daily.refresh_sox()
