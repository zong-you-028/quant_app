import numpy as np
import pandas as pd
import pytest

from core.strategy_research import (
    ResearchSettings, choose_candidate, features_and_rules, make_targets,
    portfolio_targets, training_rows, metrics,
    walk_forward_scores,
    staggered_targets,
)


def test_targets_use_executable_opens_and_require_longest_horizon():
    idx = pd.bdate_range("2020-01-01", periods=12)
    opens = pd.DataFrame({"A": [100, 100, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380],
                          "B": [100] * 12}, index=idx)
    settings = ResearchSettings(train_min=8, lag=2, horizons=(2, 4))
    y, ends = make_targets(opens, settings)
    assert y.loc[(idx[0], "A")] == 1.0
    assert y.loc[(idx[0], "B")] == .5
    assert ends.loc[idx[0]] == idx[6]
    # The 2-day label exists here but the 4-day label does not: reject the row.
    assert (idx[6], "A") not in y.index
    opens.loc[idx[2], "A"] = 0
    y, _ = make_targets(opens, settings)
    assert (idx[0], "A") not in y.index


def test_purge_rejects_labels_touching_prediction_boundary():
    idx = pd.bdate_range("2020-01-01", periods=20)
    mi = pd.MultiIndex.from_product([idx, ["A", "B"]], names=["date", "symbol"])
    X = pd.DataFrame({"x": .5}, index=mi)
    y = pd.Series(.7, index=mi, name="target")
    ends = pd.Series(idx, index=idx).shift(-6)
    train = training_rows(X, y, ends, idx[12], idx[0])
    assert train.index.get_level_values("date").max() == idx[5]
    assert ends.reindex(train.index.get_level_values("date")).max() < idx[12]
    changed = y.copy()
    changed.loc[changed.index.get_level_values("date") >= idx[6]] = -100
    pd.testing.assert_frame_equal(train, training_rows(X, changed, ends, idx[12], idx[0]))


def test_future_changes_cannot_change_past_features_or_rules():
    idx = pd.bdate_range("2020-01-01", periods=340)
    rng = np.random.default_rng(7)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (340, 4)), axis=0)),
                         index=idx, columns=list("ABCD"))
    volume = close * 1000
    X, rules, _ = features_and_rules(close, volume)
    changed = close.copy()
    changed.loc[idx[310]:] *= 10
    X2, rules2, _ = features_and_rules(changed, volume)
    mask = X.index.get_level_values("date") < idx[310]
    pd.testing.assert_frame_equal(X.loc[mask], X2.loc[mask])
    for name in rules:
        pd.testing.assert_frame_equal(rules[name].loc[:idx[309]], rules2[name].loc[:idx[309]])


def test_new_stock_needs_observed_history_not_filled_listing_prices():
    idx = pd.bdate_range("2020-01-01", periods=320)
    close = pd.DataFrame({"old": np.arange(320) + 100., "new": np.arange(320) + 100.}, index=idx)
    close.loc[:idx[99], "new"] = np.nan
    X, rules, _ = features_and_rules(close, close * 100)
    assert "new" not in X.index.get_level_values("symbol")
    assert rules["multi_momentum"]["new"].isna().all()


def test_candidates_start_cash_and_keep_failed_slots_in_cash(monkeypatch):
    import config
    monkeypatch.setattr(config, "ROTATION_ABS_MOM", False)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", True)
    idx = pd.bdate_range("2020-01-01", periods=12)
    close = pd.DataFrame(100., index=idx, columns=["A", "B"])
    score = pd.DataFrame({"A": 1., "B": .5}, index=idx)
    fs = pd.DataFrame({"A": -2., "B": 0.}, index=idx)
    snapshot = {"close": close, "fastsell": fs, "sox_gate": pd.Series(1., index=idx)}
    settings = ResearchSettings(train_min=6, rebalance_days=4, top_k=2, horizons=(1,))
    weights, schedule = portfolio_targets(score, snapshot, settings, "lgb_rank")
    assert weights.loc[:idx[7]].eq(0).all().all()
    assert weights.loc[idx[8], "A"] == 0
    assert weights.loc[idx[8], "B"] == .5
    assert not schedule.loc[:idx[7]].any()
    assert schedule.loc[idx[8]]


def test_selection_requires_both_cagr_and_sharpe_improvement():
    baseline = {"cagr": .2, "sharpe": 1., "mdd": -.3, "annual_turnover": 8.}
    development = {"production": baseline, "higher_sharpe_only": {**baseline, "sharpe": 2.}}
    assert choose_candidate(development) == "production"
    development["good"] = {**baseline, "cagr": .25, "sharpe": 1.2}
    development["overtrade"] = {**baseline, "cagr": .4, "sharpe": 3., "annual_turnover": 30.}
    assert choose_candidate(development) == "good"


def test_initial_capital_counts_toward_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.Series([-.1, 0., 0.], index=idx)
    result = metrics(returns, pd.Series(0., index=idx), pd.Series(0., index=idx))
    assert result["mdd"] == pytest.approx(-.1)


def test_invalid_research_schedule_is_rejected():
    with pytest.raises(ValueError):
        ResearchSettings(lag=0).validate()
    with pytest.raises(ValueError):
        ResearchSettings(train_min=20, horizons=(60,)).validate()


def test_actual_models_predictions_are_invariant_to_future_prices():
    idx = pd.bdate_range("2020-01-01", periods=430)
    rng = np.random.default_rng(51)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .015, (430, 16)), axis=0)),
                         index=idx, columns=[str(i) for i in range(16)])
    settings = ResearchSettings(train_min=360, refit_days=35, horizons=(5, 10))
    X, _, _ = features_and_rules(close, close * 100)
    y, ends = make_targets(close, settings)
    predicted, audit = walk_forward_scores(X, y, ends, idx, settings, verbose=False)
    altered = close.copy()
    altered.loc[idx[395]:] *= np.linspace(1, 4, 16)
    X2, _, _ = features_and_rules(altered, close * 100)
    y2, ends2 = make_targets(altered, settings)
    predicted2, _ = walk_forward_scores(X2, y2, ends2, idx, settings, verbose=False)
    for name in predicted:
        pd.testing.assert_frame_equal(predicted[name].loc[:idx[394]], predicted2[name].loc[:idx[394]])
    assert all(row["latest_label_end"] < row["test_start"] for row in audit)


def test_staggered_portfolio_averages_fixed_sleeves_without_leverage(monkeypatch):
    import config
    monkeypatch.setattr(config, "ROTATION_ABS_MOM", False)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", False)
    idx = pd.bdate_range("2020-01-01", periods=50)
    close = pd.DataFrame(100., index=idx, columns=["A", "B"])
    scores = pd.DataFrame({"A": np.tile([2., 0.], 25), "B": 1.}, index=idx)
    snapshot = {"close": close, "fastsell": close * 0, "sox_gate": pd.Series(1., index=idx)}
    settings = ResearchSettings(train_min=8, horizons=(1,), top_k=1, rebalance_days=20)
    target, schedule = staggered_targets(scores, snapshot, settings, "trend_quality")
    phases = [portfolio_targets(scores, snapshot, settings, "trend_quality", offset)
              for offset in (0, 5, 10, 15)]
    pd.testing.assert_frame_equal(target, sum(t for t, _ in phases) / 4)
    assert target.sum(axis=1).le(1.).all()
    assert target.ge(0).all().all()
    assert target.loc[:idx[7]].eq(0).all().all()
    assert all(i % 5 == 0 for i in np.flatnonzero(schedule))
