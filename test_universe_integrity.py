"""Real new listings stay unranked without falsely blocking a fresh pool."""
from contextlib import closing
import json
from pathlib import Path
import runpy
import sqlite3

import numpy as np
import pandas as pd
import pytest

import config
from core import data_pipeline as dp, rotation, paper_validation as pv


def test_current_short_listing_is_fresh_but_cannot_be_selected(monkeypatch):
    monkeypatch.setattr(dp, "_last_trading_day", lambda *args: pd.Timestamp("2026-10-02"))
    calendar = pd.bdate_range(end="2026-10-02", periods=350)
    values = 100 * np.exp(np.arange(len(calendar)) * .001)
    old = pd.DataFrame({"open": values, "high": values, "low": values, "close": values, "volume": 1000.}, index=calendar)
    frames = {"OLD": old, "NEW": old.iloc[-6:], config.BENCHMARK_SYMBOL: old}
    monkeypatch.setattr(rotation, "ensure_data", lambda symbol: None)
    monkeypatch.setattr(rotation, "load_ohlcv", lambda symbol: frames[symbol])
    monkeypatch.setattr(rotation, "get_stock_name", lambda symbol: symbol)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", False)
    monkeypatch.setattr(config, "ROTATION_SOX_GATE", False)
    result = rotation.run_rotation(["OLD", "NEW"])
    assert result["data_quality"]["coverage"] == 1
    assert not result["data_quality"]["stale"]
    assert [symbol for symbol, _, _ in result["ranking"]] == ["OLD"]
    assert "NEW" not in result["holdings"]


def test_reading_old_cache_never_reintroduces_prelisting_prices(tmp_path, monkeypatch):
    database = tmp_path / "public-market.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE ohlcv(symbol TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL)")
        connection.executemany("INSERT INTO ohlcv VALUES(?,?,?,?,?,?,?)", [
            ("NEW", "2025-01-01", 100., 100., 100., 100., 100.),
            ("NEW", "2025-09-17", 101., 101., 101., 101., 100.),
        ])
        connection.commit()
    monkeypatch.setattr(config, "DB_PATH", str(database))
    monkeypatch.setattr(config, "UNIVERSE_LISTING_DATES", {"NEW": "2025-09-17"})
    monkeypatch.setattr(dp, "ensure_db", lambda: None)
    assert dp.load_ohlcv("NEW").index.tolist() == [pd.Timestamp("2025-09-17")]


def test_same_symbols_with_changed_listing_date_change_frozen_identity(monkeypatch):
    before = pv.frozen_identity()
    changed = dict(config.UNIVERSE_LISTING_DATES)
    changed[config.UNIVERSE[0]] = "2000-01-01"
    monkeypatch.setattr(config, "UNIVERSE_LISTING_DATES", changed)
    after = pv.frozen_identity()
    assert before != after
    assert "data_seed/universe_150.json" in before["source_sha256"]
    assert "core/tpex_prices.py" in before["source_sha256"]


@pytest.mark.parametrize("corruption", ["missing_metadata", "wrong_market", "future_listing"])
def test_bad_metadata_cannot_silently_remove_otc_or_listing_guards(tmp_path, corruption):
    root = Path(config.BASE_DIR)
    snapshot = json.loads((root / "data_seed/universe_150.json").read_text(encoding="utf-8"))
    if corruption == "missing_metadata": snapshot["top150"].pop()
    if corruption == "wrong_market": snapshot["top150"][0]["market"] = "unknown"
    if corruption == "future_listing": snapshot["top150"][0]["listed_date"] = "2099-01-01"
    (tmp_path / "data_seed").mkdir()
    (tmp_path / "data_seed/universe_150.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (tmp_path / "config.py").write_bytes((root / "config.py").read_bytes())
    with pytest.raises(ValueError):
        runpy.run_path(str(tmp_path / "config.py"))
