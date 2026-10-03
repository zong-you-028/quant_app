"""Meaningful offline regressions for audit isolation and causal comparisons."""
from contextlib import closing
import gzip
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from core.robustness_audit import (
    MarketSnapshot, compare_past, execution_at, fixed_scenarios,
    isolated_public_seed, load_public_snapshot, replay, yearly_concentration,
    paired_confidence,
)


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    import config
    import requests
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "forbidden.db"))
    monkeypatch.setenv("JOURNAL_DATABASE_URL", "")
    monkeypatch.setattr(config, "ROTATION_ABS_MOM", True)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", True)
    monkeypatch.setattr(config, "ROTATION_SOX_GATE", True)
    monkeypatch.setattr(config, "ROTATION_DEFENSIVE", False)
    monkeypatch.setattr(config, "ROTATION_TOP_K", 8)
    monkeypatch.setattr(config, "ROTATION_REBAL_DAYS", 20)

    def forbidden(*args, **kwargs):
        raise AssertionError("test must not access database or network")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(sqlite3.dbapi2, "connect", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    index = pd.bdate_range("2020-01-01", periods=420)
    rng = np.random.default_rng(37)
    symbols = [f"S{i:02d}" for i in range(16)]
    values = 100 * np.exp(np.cumsum(rng.normal(.001, .012, (420, 16)), axis=0))
    prices, chips = {}, {}
    for i, symbol in enumerate(symbols):
        close = values[:, i]
        prices[symbol] = pd.DataFrame({"open": close * .997, "high": close * 1.02,
                                      "low": close * .98, "close": close,
                                      "volume": 100000. + np.arange(len(index))}, index=index)
        ratio = .4 + np.cumsum(rng.normal(0, .0008, 420))
        chips[symbol] = pd.DataFrame({"big_shares": ratio * 1000000,
                                     "total_shares": 1000000.}, index=index)
    # A real gap in the ETF calendar must affect both active and its baseline.
    benchmark_index = index.delete([38, 61, 119])
    prices[config.BENCHMARK_SYMBOL] = prices[symbols[0]].reindex(benchmark_index).copy()
    sox_index = pd.bdate_range("2018-01-01", index[-1])
    sox = pd.Series(100 * np.exp(np.arange(len(sox_index)) * .0006), index=sox_index)
    return MarketSnapshot(prices, chips, sox, symbols, config.BENCHMARK_SYMBOL)


@pytest.mark.parametrize("source", ["prices", "chips", "sox", "all"])
def test_future_source_pollution_cannot_change_past_execution(snapshot, source):
    cutoff = snapshot.calendar[348]
    original = replay(snapshot)
    changed = snapshot.perturb_future(cutoff, source)
    future = replay(changed)
    differences = compare_past(original, future, cutoff)
    assert all(value == 0 for value in differences.values())
    assert execution_at(original, snapshot, cutoff) == execution_at(future, changed, cutoff)


def test_complete_three_source_prefix_matches_original_model(snapshot):
    cutoff = snapshot.calendar[348]
    original = replay(snapshot)
    prefix = replay(snapshot.prefix(cutoff))
    differences = compare_past(original, prefix, cutoff)
    assert all(value == 0 for value in differences.values())
    assert execution_at(original, snapshot, cutoff)["effective_target_weights"] == prefix["model"]["model_current_weights"]
    assert prefix["model"]["last_date"] == str(cutoff.date())


def test_perturbation_never_changes_original_sources_or_past_rows(snapshot):
    cutoff = snapshot.calendar[348]
    saved = snapshot.prices["S00"].copy()
    changed = snapshot.perturb_future(cutoff)
    pd.testing.assert_frame_equal(saved, snapshot.prices["S00"])
    pd.testing.assert_frame_equal(saved.loc[:cutoff], changed.prices["S00"].loc[:cutoff])
    assert not changed.prices["S00"].loc[cutoff:].equals(saved.loc[cutoff:])
    assert changed.sox.loc[:cutoff].equals(snapshot.sox.loc[:cutoff])


def test_comparator_rejects_real_past_signal_drift(snapshot):
    cutoff = snapshot.calendar[348]
    original = replay(snapshot)
    broken = replay(snapshot)
    broken["model"]["target_weights"].iloc[320, 0] += .01
    with pytest.raises(AssertionError):
        compare_past(original, broken, cutoff)


def test_common_calendar_baseline_has_identical_calendar_and_phase(snapshot):
    active = replay(snapshot)
    baseline = replay(snapshot, mode="production_common_calendar")
    assert active["model"]["target_weights"].index.equals(snapshot.calendar)
    assert baseline["model"]["target_weights"].index.equals(snapshot.calendar)
    assert active["model"]["net_returns"].index.equals(baseline["model"]["net_returns"].index)
    # Eligible rebalance dates are the same phase; legacy starts before the
    # active year's warmup, so compare scheduled dates after both are eligible.
    pd.testing.assert_series_equal(active["rebalance"].iloc[280:], baseline["rebalance"].iloc[280:])


def test_higher_costs_reduce_nav_without_changing_selected_targets(snapshot):
    base = replay(snapshot, cost=.003)
    high = replay(snapshot, cost=.008)
    pd.testing.assert_frame_equal(base["model"]["target_weights"], high["model"]["target_weights"])
    assert high["model"]["full_equity"].iloc[-1] < base["model"]["full_equity"].iloc[-1]


def test_longer_execution_lag_retains_causal_signal_targets(snapshot):
    base = replay(snapshot, lag=2)
    delayed = replay(snapshot, lag=4)
    pd.testing.assert_frame_equal(base["model"]["target_weights"], delayed["model"]["target_weights"])
    assert base["model"]["execution_lag"] == 2
    assert delayed["model"]["execution_lag"] == 4
    assert not base["model"]["net_returns"].equals(delayed["model"]["net_returns"])


def test_external_source_delay_shifts_gate_without_refitting(snapshot):
    # Force a market-gate transition late enough for the active model.
    boundary = snapshot.calendar[333]
    snapshot.sox.loc[boundary:] *= .3
    base = replay(snapshot)
    delayed = replay(snapshot, sox_delay=2)
    target = base["model"]["target_weights"]
    changed = delayed["model"]["target_weights"]
    assert target.loc[boundary].sum() == 0
    assert changed.loc[boundary].sum() > 0
    assert changed.loc[snapshot.calendar[335]].sum() == 0


def test_chip_information_delay_changes_only_available_rebalance_gate(snapshot):
    boundary = snapshot.calendar[320]
    normal = replay(snapshot)
    normal_delayed = replay(snapshot, chip_delay=2)
    eligible = normal["model"]["target_weights"].loc[boundary]
    delayed_eligible = normal_delayed["model"]["target_weights"].loc[boundary]
    symbol = eligible[(eligible > 0) & (delayed_eligible > 0)].index[0]
    snapshot.chips[symbol].loc[boundary:, "big_shares"] *= .3
    immediate = replay(snapshot)
    delayed = replay(snapshot, chip_delay=2)
    assert immediate["model"]["target_weights"].loc[boundary, symbol] == 0
    assert delayed["model"]["target_weights"].loc[boundary, symbol] == .125
    pd.testing.assert_frame_equal(immediate["model"]["target_weights"].loc[:snapshot.calendar[319]],
                                  normal["model"]["target_weights"].loc[:snapshot.calendar[319]])
    pd.testing.assert_frame_equal(delayed["model"]["target_weights"].loc[:snapshot.calendar[319]],
                                  normal_delayed["model"]["target_weights"].loc[:snapshot.calendar[319]])


def test_all_stress_scenarios_are_fixed_and_retained():
    scenarios = fixed_scenarios()
    grid = [(r["cost"], r["lag"]) for r in scenarios[:9]]
    assert grid == [(cost, lag) for cost in (.003, .005, .008) for lag in (2, 3, 4)]
    assert len(scenarios) == 14
    assert len({r["name"] for r in scenarios}) == 14
    assert scenarios[-1]["chip_delay"] == scenarios[-1]["sox_delay"] == 2


def test_year_concentration_discloses_every_exclusion_and_keeps_paired_dates():
    index = pd.bdate_range("2020-01-01", "2022-12-30")
    noise = np.tile([-.01, .01], int(np.ceil(len(index) / 2)))[:len(index)]
    alpha = np.where(index.year == 2021, .003, -.0002)
    paths = pd.DataFrame({"cost_0.003_lag_2__buffered_momentum": noise + alpha,
                          "cost_0.003_lag_2__production_common_calendar": noise,
                          "006208": noise}, index=index)
    report = yearly_concentration(paths)
    assert report["highest_annualized_difference_year"] == 2021
    assert report["all_years_annualized_mean_active_minus_baseline"] > 0
    assert report["excluding_highest_difference_year"]["annualized_mean_active_minus_baseline"] == pytest.approx(-.0002 * 252)
    assert {r["excluded_year"] for r in report["leave_one_calendar_year_out"]} == {2020, 2021, 2022}
    assert report["no_exclusion_used_for_strategy_selection"]
    assert all("annual_turnover" not in row["buffered_momentum"] for row in report["yearly"])


def test_retrospective_confidence_is_single_candidate_paired_and_never_replaces_family_audit():
    index = pd.bdate_range("2020-01-01", periods=180)
    baseline = np.tile([-.01, .012], 90)
    paths = pd.DataFrame({"cost_0.003_lag_2__buffered_momentum": baseline,
                          "cost_0.003_lag_2__production_common_calendar": baseline,
                          "006208": baseline}, index=index)
    report = paired_confidence(paths)
    assert report["status"] == "pointwise_retrospective_not_selection_adjusted"
    assert not report["original_11_candidate_correction_recomputed"]
    for rows in report["comparisons"].values():
        assert set(rows) == {"20", "40", "60"}
        for row in rows.values():
            assert row["rows"] == 180 and row["draws"] == 3000 and row["seed"] == 314159
            assert row["annualized_mean_active"] == row["pointwise_95_lower"] == row["pointwise_95_upper"] == 0
            assert row["single_candidate_one_sided_p"] == 1


def test_public_seed_context_restores_paths_environment_and_forbids_formal_db(tmp_path, monkeypatch):
    import config
    # A minimal public seed is enough to exercise isolation without a model.
    source = tmp_path / "seed"
    source.mkdir()
    db = source / "market.db"
    with closing(sqlite3.connect(db)) as connection:
        connection.execute("CREATE TABLE ohlcv (symbol TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL)")
        connection.execute("CREATE TABLE chip_weekly (symbol TEXT,date TEXT,big_shares REAL,total_shares REAL)")
        connection.execute("INSERT INTO ohlcv VALUES ('006208','2020-01-01',10,10,10,10,100)")
        connection.commit()
    with gzip.open(source / "market.db.gz", "wb") as target:
        target.write(db.read_bytes())
    pd.Series([10.], index=pd.to_datetime(["2020-01-01"])).to_csv(source / "sox.csv")
    previous_path, previous_data = config.DB_PATH, config.DATA_DIR
    monkeypatch.setenv("APP_DATA_DIR", "kept_previous_value")
    monkeypatch.setenv("JOURNAL_DATABASE_URL", "kept_previous_value")
    with isolated_public_seed(source) as (directory, counters):
        assert config.DB_PATH == str(directory / "market.db")
        assert os.environ["JOURNAL_DATABASE_URL"] == ""
        loaded = load_public_snapshot(directory)
        assert loaded.calendar[0] == pd.Timestamp("2020-01-01")
        with pytest.raises(AssertionError, match="formal"):
            sqlite3.connect(Path(__file__).parent / "data/market.db")
        with pytest.raises(AssertionError, match="outside"):
            sqlite3.connect(tmp_path / "outside.db")
        assert counters["formal_database_attempts"] == counters["outside_database_attempts"] == 1
    assert config.DB_PATH == previous_path and config.DATA_DIR == previous_data
    assert os.environ["APP_DATA_DIR"] == "kept_previous_value"
    assert os.environ["JOURNAL_DATABASE_URL"] == "kept_previous_value"


@pytest.mark.parametrize("kwargs", [{"lag": 0}, {"cost": -.01}, {"chip_delay": -1}, {"sox_delay": -1}])
def test_invalid_stress_input_is_rejected_before_any_data_access(snapshot, kwargs):
    with pytest.raises(ValueError, match="invalid"):
        replay(snapshot, **kwargs)
