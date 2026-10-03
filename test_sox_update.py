"""SOX failures keep usable data and never recursively redownload."""
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from core import data_pipeline as dp, market_regime as mr


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "sox.csv"
    monkeypatch.setattr(mr, "SOX_CSV", str(path))
    monkeypatch.setattr(mr.config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(dp, "_last_trading_day", lambda: pd.Timestamp("2026-10-02"))
    return path


def test_missing_cache_failed_download_attempts_once(source, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **kw: calls.append(kw) or pd.DataFrame()))
    assert mr.load_sox() is None
    assert len(calls) == 1
    assert not source.exists()
    with pytest.raises(RuntimeError, match="未回傳"):
        mr.refresh_sox(raise_errors=True)
    assert len(calls) == 2


def test_failed_download_preserves_old_cache(source, monkeypatch):
    pd.Series([100.], index=pd.to_datetime(["2026-09-01"]), name="close").to_csv(source)
    before = source.read_bytes()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **kw: pd.DataFrame()))
    assert mr.refresh_sox().iloc[0] == 100.
    assert source.read_bytes() == before


def test_fresh_cache_does_not_download(source, monkeypatch):
    pd.Series([100.], index=pd.to_datetime(["2026-10-01"]), name="close").to_csv(source)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **kw: pytest.fail("fresh cache")))
    assert mr.refresh_sox(raise_errors=True).iloc[0] == 100.


def test_incremental_single_row_download_merges_atomically(source, monkeypatch):
    pd.Series([100.], index=pd.to_datetime(["2026-09-01"]), name="close").to_csv(source)
    calls = []
    fresh = pd.DataFrame({("Close", "^SOX"): [120.]}, index=pd.to_datetime(["2026-10-02"]))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **kw: calls.append(kw) or fresh))
    result = mr.refresh_sox(raise_errors=True)
    assert len(result) == 2 and result.iloc[-1] == 120.
    assert calls[0]["start"] == "2026-08-27"
    assert calls[0]["timeout"] == 8 and calls[0]["threads"] is False
    assert len(pd.read_csv(source)) == 2
    assert list(source.parent.glob("*.csv")) == [source]


def test_atomic_write_failure_keeps_old_file(source, monkeypatch):
    pd.Series([100.], index=pd.to_datetime(["2026-09-01"]), name="close").to_csv(source)
    before = source.read_bytes()
    fresh = pd.DataFrame({"Close": [120.]}, index=pd.to_datetime(["2026-10-02"]))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **kw: fresh))
    monkeypatch.setattr(mr.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk failed")))
    with pytest.raises(OSError, match="disk failed"):
        mr.refresh_sox(raise_errors=True)
    assert source.read_bytes() == before
    assert list(source.parent.glob("*.csv")) == [source]


def test_new_data_directory_uses_bundled_sox_without_download(source, monkeypatch):
    seed_dir = source.parent / "data_seed"
    seed_dir.mkdir()
    pd.Series([100.], index=pd.to_datetime(["2026-10-02"]), name="close").to_csv(seed_dir / "sox.csv")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **kw: pytest.fail("seed is current")))
    assert mr.load_sox().iloc[0] == 100.
    assert pd.read_csv(source).iloc[0, 1] == 100.
