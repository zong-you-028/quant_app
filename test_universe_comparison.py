"""Isolated tests for current-universe research, never actual trading records."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from core.robustness_audit import MarketSnapshot
from research.universe_150_2026_10_03 import compare_universes as comparison


def ranking_file(tmp_path, symbols=None, asof="2020-01-01"):
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps({"symbols": symbols or [str(i) for i in range(1000, 1150)],
                                "asof": asof, "source": "synthetic regression fixture"}), encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", ["short", "duplicate", "no_asof", "intraday"])
def test_incomplete_or_ambiguous_ranking_is_rejected(tmp_path, kind):
    path = ranking_file(tmp_path)
    spec = json.loads(path.read_text())
    if kind == "short":
        spec["symbols"] = spec["symbols"][:-1]
    elif kind == "duplicate":
        spec["symbols"][-1] = spec["symbols"][0]
    elif kind == "no_asof":
        spec.pop("asof")
    else:
        spec["asof"] += "T12:00:00"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError):
        comparison.read_ranking(path)


def test_original_50_does_not_follow_later_config_changes(monkeypatch):
    import config
    original = comparison.FIXED_50
    monkeypatch.setattr(config, "UNIVERSE", [str(i) for i in range(1000, 1150)])
    assert comparison.FIXED_50 == original
    assert len(original) == len(set(original)) == 50
    assert original[:3] == ("2330", "2317", "2454")


def test_provisional_ranking_is_never_reported_as_official_complete_market_top_150():
    summary = comparison.make_summary({"status": "incomplete_sources_no_performance_claim",
        "ranking_asof": "2026-10-02", "price_asof": "2026-10-02", "coverage": {},
        "ranking_source_metadata": {"ranking_scope": "same-day priced TWSE/TPEx ordinary shares",
        "provisional": True, "source_incomplete": True,
        "excluded_unpriced": [{"symbol": str(i)} for i in range(7)]}})
    assert "名單仍暫定" in summary and "7檔普通股缺同日有效價格" in summary
    assert "不稱官方完整全市場市值榜" in summary


def thin_snapshot():
    calendar = pd.bdate_range("2020-01-01", periods=400)
    full = pd.DataFrame({"open": np.arange(400) + 100., "high": np.arange(400) + 101.,
                         "low": np.arange(400) + 99., "close": np.arange(400) + 100.,
                         "volume": 1000.}, index=calendar)
    chip = pd.DataFrame({"big_shares": 400., "total_shares": 1000.}, index=calendar)
    return MarketSnapshot({"006208": full.copy(), "OLD": full.copy(), "NEW": full.iloc[-30:].copy()},
                          {"OLD": chip.copy(), "NEW": chip.iloc[-30:].copy()},
                          full["close"], ["OLD", "NEW"], "006208")


def test_true_short_listing_history_stays_in_declared_pool_without_imputation():
    snapshot = thin_snapshot()
    before = snapshot.prices["NEW"].copy()
    result = comparison.coverage(snapshot, snapshot.calendar[-1])
    assert result["complete_source_coverage"]
    assert result["declared_symbols"] == 2
    assert result["latest_rank_qualified_count"] == 1
    assert result["latest_rank_qualified_symbols"] == ["OLD"]
    assert result["short_price_history_below_252"] == ["NEW"]
    assert result["chip_reader_warmup_below_61"] == ["NEW"]
    pd.testing.assert_frame_equal(before, snapshot.prices["NEW"])


@pytest.mark.parametrize("source", ["prices", "chips"])
def test_completely_missing_source_does_not_become_short_listing_or_smaller_pool(source):
    snapshot = thin_snapshot()
    getattr(snapshot, source).pop("NEW")
    result = comparison.coverage(snapshot, snapshot.calendar[-1])
    assert not result["complete_source_coverage"]
    assert result["declared_symbols"] == 2
    assert result["missing_" + source] == ["NEW"]


def test_formal_database_rejected_before_file_hash_or_connection(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("no file hashing or connection allowed")
    monkeypatch.setattr(comparison, "digest", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with pytest.raises(ValueError, match="before hashing"):
        comparison.run_comparison(comparison.ROOT / "data/market.db", "unused", "unused", tmp_path)


@pytest.fixture
def public_cache(tmp_path):
    database = tmp_path / "research.db"
    symbols = list(comparison.FIXED_50) + [str(i) for i in range(8100, 8200)]
    calendar = pd.bdate_range("2020-01-01", periods=900)
    rng = np.random.default_rng(59)
    prices, chips = [], []
    for i, symbol in enumerate([*symbols, "006208"]):
        close = 100 * np.exp(np.cumsum(rng.normal(.0005 + i * .000001, .012, len(calendar))))
        ratio = np.clip(.4 + np.cumsum(rng.normal(0, .0004, len(calendar))), .05, .95)
        for date, value, hold in zip(calendar, close, ratio):
            day = str(date.date())
            prices.append((symbol, day, value * .997, value * 1.02, value * .98, value, 100000.))
            if symbol != "006208":
                chips.append((symbol, day, hold * 1000000, 1000000.))
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE ohlcv(symbol TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL)")
        connection.execute("CREATE TABLE chip_weekly(symbol TEXT,date TEXT,big_shares REAL,total_shares REAL)")
        connection.execute("CREATE TABLE stock_info(symbol TEXT,name TEXT)")
        connection.executemany("INSERT INTO ohlcv VALUES(?,?,?,?,?,?,?)", prices)
        connection.executemany("INSERT INTO chip_weekly VALUES(?,?,?,?)", chips)
        connection.commit()
    sox = tmp_path / "sox.csv"
    sox_dates = pd.bdate_range("2018-01-01", calendar[-1])
    pd.Series(100 * np.exp(np.arange(len(sox_dates)) * .0005), index=sox_dates).to_csv(sox)
    rank = ranking_file(tmp_path, symbols, str(calendar[-1].date()))
    return database, sox, rank, symbols


def test_complete_comparison_keeps_original_rules_same_dates_and_all_stress_cases(public_cache, tmp_path, monkeypatch):
    import config
    database, sox, rank, symbols = public_cache
    # Simulate a later app universe change: comparison still freezes old50.
    monkeypatch.setattr(config, "UNIVERSE", symbols)
    monkeypatch.setattr(config, "ROTATION_TOP_K", 8)
    monkeypatch.setattr(config, "ROTATION_REBAL_DAYS", 20)
    before = comparison.digest(database)
    report = comparison.run_comparison(database, sox, rank, tmp_path / "complete", verbose=False)
    assert report["status"] == "retrospective_completed"
    assert report["universes"]["fixed_50"] == list(comparison.FIXED_50)
    assert len(report["universes"]["market_cap_150"]) == 150
    assert report["model_parameters"]["top_k"] == 8
    assert report["model_parameters"]["retention_rank_limit"] == 16
    assert len(report["scenarios"]) == 9
    assert report["evaluation"]["observations"] == 144
    assert set(report["statistics"]["paired_pointwise_bootstrap"]) == {"20", "40", "60"}
    assert not report["app_config_changed"] and not report["strategy_parameters_changed"]
    assert comparison.digest(database) == before
    for group in ("source", "model"):
        for key in ("formal_database_attempts", "outside_database_attempts", "external_download_attempts"):
            assert report["isolation"][group][key] == 0
    for name in ("fixed_50", "market_cap_150"):
        targets = pd.read_csv(tmp_path / "complete" / f"targets_{name}.csv", index_col="date")
        assert targets.gt(0).sum(axis=1).max() <= 8


def test_missing_chip_refuses_performance_and_does_not_silently_reduce_150(public_cache, tmp_path):
    database, sox, rank, symbols = public_cache
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM chip_weekly WHERE symbol=?", (symbols[-1],))
        connection.commit()
    output = tmp_path / "incomplete"
    report = comparison.run_comparison(database, sox, rank, output, verbose=False)
    assert report["status"] == "incomplete_sources_no_performance_claim"
    assert report["coverage"]["market_cap_150"]["declared_symbols"] == 150
    assert report["coverage"]["market_cap_150"]["missing_chips"] == [symbols[-1]]
    assert "scenarios" not in report and "statistics" not in report
    assert not (output / "daily_returns.csv").exists()


def test_private_tables_are_rejected_before_public_seed_backup(public_cache):
    database, sox, rank, symbols = public_cache
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE journal_trades(private_value TEXT)")
        connection.commit()
    with pytest.raises(ValueError, match="private/non-market"):
        with comparison.provided_public_snapshot(database, sox, symbols):
            raise AssertionError("private source should never yield")
