"""Single active app model, reproducible research variants, and execution."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

import config
from core import data_pipeline, market_regime, rotation, strategy_models
from core.execution import backtest_open_execution
from core.strategy_models import MODEL_SPECS, build_rank_scores, data_quality
from core.strategy_research import ResearchSettings, features_and_rules, portfolio_targets


TAIPEI = dt.timezone(dt.timedelta(hours=8))
ASOF = dt.datetime(2026, 10, 3, 10, 0, tzinfo=TAIPEI)


@pytest.mark.parametrize("legacy_mode", ["production", "multi_momentum", "trend_quality", "invalid_legacy_setting"])
def test_only_active_default_ignores_legacy_model_switch(monkeypatch, legacy_mode):
    monkeypatch.setattr(config, "ROTATION_SCORE_MODE", legacy_mode, raising=False)
    mode, spec = strategy_models.model_spec()
    assert strategy_models.ACTIVE_MODEL_ID == "buffered_momentum"
    assert mode == "buffered_momentum"
    active = strategy_models.active_model_spec()
    assert active["label"] == spec["label"] == MODEL_SPECS["buffered_momentum"]["label"]
    # UI consumers receive metadata copies, so editing a returned value cannot
    # change subsequent active-model identity or presentation.
    active["label"] = "local presentation change"
    assert strategy_models.active_model_spec()["label"] == spec["label"]


def test_default_rank_scores_use_active_model_without_removing_research_variants(monkeypatch):
    monkeypatch.setattr(config, "ROTATION_SCORE_MODE", "production", raising=False)
    close = _closes()
    pd.testing.assert_frame_equal(
        build_rank_scores(close), build_rank_scores(close, "buffered_momentum"))
    pd.testing.assert_frame_equal(
        build_rank_scores(close, "production"),
        close.shift(config.ROTATION_SKIP_DAYS) / close.shift(config.ROTATION_MOM_DAYS) - 1)
    for mode in MODEL_SPECS:
        assert strategy_models.model_spec(mode)[0] == mode


def _closes(periods=340, symbols="ABCD", seed=7):
    index = pd.bdate_range("2020-01-01", periods=periods)
    rng = np.random.default_rng(seed)
    values = 100 * np.exp(np.cumsum(rng.normal(0, .01, (periods, len(symbols))), axis=0))
    return pd.DataFrame(values, index=index, columns=list(symbols))


@pytest.mark.parametrize("mode", ["multi_momentum", "trend_quality", "buffered_momentum"])
def test_app_rank_scores_equal_researched_rules(mode):
    close = _closes()
    _, rules, _ = features_and_rules(close, close * 1000)
    pd.testing.assert_frame_equal(build_rank_scores(close, mode), rules[mode])


@pytest.mark.parametrize("mode", list(MODEL_SPECS))
def test_future_prices_do_not_change_past_app_scores(mode):
    close = _closes()
    changed = close.copy()
    boundary = close.index[310]
    changed.loc[boundary:] *= np.array([.1, 2., 5., 20.])
    pd.testing.assert_frame_equal(
        build_rank_scores(close, mode).loc[:close.index[309]],
        build_rank_scores(changed, mode).loc[:close.index[309]],
    )


@pytest.mark.parametrize("mode", [mode for mode in MODEL_SPECS if mode != "production"])
def test_new_listing_needs_observed_252_day_history(mode):
    close = _closes(periods=380, symbols=["old", "new"])
    close.loc[:close.index[99], "new"] = np.nan
    scores = build_rank_scores(close, mode)
    # A 252-day return needs the current bar plus the bar 252 days earlier.
    assert scores.loc[:close.index[351], "new"].isna().all()
    assert pd.notna(scores.loc[close.index[352], "new"])
    # Filling listing dates would change eligibility and must never occur.
    assert close.loc[:close.index[99], "new"].isna().all()


def _price_frame(last_date):
    return pd.DataFrame({"open": [100.], "close": [101.]},
                        index=pd.DatetimeIndex([last_date]))


def test_data_quality_reports_each_stock_date_in_mixed_snapshot(monkeypatch):
    monkeypatch.setattr(config, "ROTATION_SOX_GATE", False)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", False)
    prices = {"A": _price_frame("2026-10-02"), "B": _price_frame("2026-09-30"),
              config.BENCHMARK_SYMBOL: _price_frame("2026-10-02")}
    quality = data_quality(["A", "B"], prices=prices, asof=ASOF)
    assert quality["target_date"] == "2026-10-02"
    assert quality["asof"] == "2026-10-02"
    assert quality["oldest_asof"] == "2026-09-30"
    assert quality["stale_symbols"] == ["B"]
    assert quality["symbol_dates"]["A"] == "2026-10-02"
    assert quality["symbol_dates"]["B"] == "2026-09-30"
    assert quality["coverage"] == .5
    assert quality["stale"]


def test_fresh_stocks_cannot_hide_stale_benchmark_calendar(monkeypatch):
    monkeypatch.setattr(config, "ROTATION_SOX_GATE", False)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", False)
    prices = {"A": _price_frame("2026-10-02"), "B": _price_frame("2026-10-02"),
              config.BENCHMARK_SYMBOL: _price_frame("2026-09-01")}
    quality = data_quality(["A", "B"], prices=prices, asof=ASOF)
    assert quality["stale_symbols"] == []
    assert quality["coverage"] == 1.
    assert quality["benchmark_asof"] == "2026-09-01"
    assert quality["stale"]
    assert any("基準" in reason for reason in quality["reasons"])


def test_fresh_stocks_and_benchmark_are_current(monkeypatch):
    monkeypatch.setattr(config, "ROTATION_SOX_GATE", False)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", False)
    prices = {"A": _price_frame("2026-10-02"),
              config.BENCHMARK_SYMBOL: _price_frame("2026-10-02")}
    quality = data_quality(["A"], prices=prices, asof=ASOF)
    assert not quality["stale"]
    assert quality["reasons"] == []


def test_current_prices_cannot_hide_stale_or_missing_chip_gate(monkeypatch):
    monkeypatch.setattr(config, "ROTATION_SOX_GATE", False)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", True)
    prices = {s: _price_frame("2026-10-02") for s in ("A", "B", "C", config.BENCHMARK_SYMBOL)}
    quality = data_quality(["A", "B", "C", config.BENCHMARK_SYMBOL], prices=prices,
                           asof=ASOF, chip_dates={"A": "2026-09-25", "B": "2026-09-24"})
    assert quality["stale_symbols"] == []
    assert quality["stale_chip_symbols"] == ["B", "C"]
    assert quality["chip_asof"] == "2026-09-24"
    assert quality["stale"]
    assert any("外資持股" in reason for reason in quality["reasons"])
    # Current gate data clears the warning; ETF foreign holdings are not needed.
    quality = data_quality(["A", "B", "C", config.BENCHMARK_SYMBOL], prices=prices,
                           asof=ASOF, chip_dates={s: "2026-10-02" for s in ("A", "B", "C")})
    assert not quality["stale"]


def _patch_app_prices(monkeypatch, close, scores=None, opens=None):
    """Use in-memory prices while keeping the real portfolio and execution code."""
    opens = close if opens is None else opens
    prices = {symbol: pd.DataFrame({"open": opens[symbol], "close": close[symbol]})
              for symbol in close}
    prices[config.BENCHMARK_SYMBOL] = pd.DataFrame(
        {"open": opens.mean(axis=1), "close": close.mean(axis=1)})
    baseline = pd.DataFrame(.2, index=close.index, columns=close.columns)
    monkeypatch.setattr(rotation, "_load_panel", lambda *args: (
        close.pct_change(fill_method=None), baseline, baseline,
        {symbol: symbol for symbol in close},
    ))
    monkeypatch.setattr(rotation, "load_ohlcv", lambda symbol: prices[symbol])
    monkeypatch.setattr(rotation, "ensure_data", lambda symbol: "cached")
    monkeypatch.setattr(config, "ROTATION_ABS_MOM", False)
    monkeypatch.setattr(config, "ROTATION_FASTSELL_GATE", False)
    monkeypatch.setattr(config, "ROTATION_SOX_GATE", False)
    monkeypatch.setattr(config, "ROTATION_EXECUTION_LAG", 2)
    monkeypatch.setattr(rotation, "validation_status", lambda mode: "test fixture")
    monkeypatch.setattr(rotation, "data_quality", lambda symbols, prices: data_quality(
        symbols, prices=prices, asof=ASOF))
    if scores is not None:
        monkeypatch.setattr(rotation, "build_rank_scores", lambda *args, **kwargs: scores.copy())
    return prices


def test_non_rebalance_sox_exit_has_its_own_pending_execution(monkeypatch):
    index = pd.bdate_range("2026-09-21", periods=6)
    close = pd.DataFrame({"A": np.arange(6) + 100.}, index=index)
    _patch_app_prices(monkeypatch, close)
    observed_lags = []

    def gate(calendar, lag):
        observed_lags.append(lag)
        return pd.Series([1., 1., 1., 1., 1., 0.], index=calendar)

    monkeypatch.setattr(market_regime, "sox_regime_series", gate)
    monkeypatch.setattr(market_regime, "sox_status", lambda asof: {
        "ok": True, "risk_on": False, "asof": str(index[-1].date()),
    })
    result = rotation.run_rotation(
        symbols=["A"], top_k=1, rebal_days=3, cost_per_turnover=0,
        abs_mom=False, defensive=False, sox_gate=True, score_mode="production")
    assert observed_lags == [0]  # The execution engine applies the sole t+2 lag.
    assert result["selection_date"] == str(index[3].date())
    assert result["selection_execution_date"] == str(index[5].date())
    assert not result["selection_execution_pending"]
    assert result["target_change_date"] == str(index[5].date())
    assert result["target_execution_date"] is None
    assert result["target_execution_pending"]
    assert result["holdings"] == []
    assert result["model_current_holdings"] == ["A"]
    assert result["model_current_weights"] == {"A": 1.}
    # The latest risk-off target is pending; the executable path still owns A.
    assert result["net_returns"].iloc[-1] == pytest.approx(105 / 104 - 1)


def test_app_drawdown_includes_first_period_loss():
    index = pd.bdate_range("2026-09-21", periods=3)
    result = rotation._aligned_performance(
        pd.Series([-.1, 0., 0.], index=index), pd.Series(0., index=index))
    assert result["mdd"] == pytest.approx(-.1)
    assert result["benchmark_mdd"] == 0.


def test_buffer_keeps_previous_top_2k_and_exits_beyond_buffer(monkeypatch):
    close = _closes(periods=16, symbols="ABCDE", seed=16)
    opens = close.shift().fillna(100.)
    scores = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    scores.iloc[4:8] = [5., 4., 3., 2., 1.]   # Initial A, B.
    scores.iloc[8:12] = [3., 2., 5., 4., 1.]  # Retain A rank 3, B rank 4.
    scores.iloc[12:] = [1., 3., 5., 4., 2.]   # A rank 5 exits; retain B, add C.
    _patch_app_prices(monkeypatch, close, scores=scores, opens=opens)
    captured = {}

    def capture_execution(targets, open_prices, close_prices, **kwargs):
        captured["targets"] = targets.copy()
        captured["schedule"] = kwargs["rebalance"].copy()
        return backtest_open_execution(targets, open_prices, close_prices, **kwargs)

    monkeypatch.setattr(rotation, "backtest_open_execution", capture_execution)
    result = rotation.run_rotation(
        symbols=list(close), top_k=2, rebal_days=4, cost_per_turnover=.003,
        abs_mom=False, defensive=False, sox_gate=False, score_mode="buffered_momentum")
    target = captured["targets"]
    assert list(target.loc[close.index[4]][lambda row: row.gt(0)].index) == ["A", "B"]
    assert list(target.loc[close.index[8]][lambda row: row.gt(0)].index) == ["A", "B"]
    assert list(target.loc[close.index[12]][lambda row: row.gt(0)].index) == ["B", "C"]
    assert set(result["holdings"]) == {"B", "C"}
    assert result["sells"] == ["A"]
    assert result["buys"] == ["C"]

    settings = ResearchSettings(train_min=1, top_k=2, rebalance_days=4, lag=2, cost=.003)
    snapshot = {"close": close, "fastsell": close * 0,
                "sox_gate": pd.Series(1., index=close.index)}
    expected_targets, expected_schedule = portfolio_targets(
        scores, snapshot, settings, "buffered_momentum")
    pd.testing.assert_frame_equal(target, expected_targets)
    pd.testing.assert_series_equal(captured["schedule"], expected_schedule)
    expected_returns, _ = backtest_open_execution(
        expected_targets, opens, close, lag=2, cost=.003, rebalance=expected_schedule)
    pd.testing.assert_series_equal(result["net_returns"], expected_returns)


def test_default_rotation_and_stock_analysis_share_buffered_membership(monkeypatch):
    close = _closes(periods=16, symbols="ABCDE", seed=16)
    opens = close.shift().fillna(100.)
    scores = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    scores.iloc[4:8] = [5., 4., 3., 2., 1.]   # Initially A and B.
    scores.iloc[8:12] = [3., 2., 5., 4., 1.]  # Buffer retains A and B.
    scores.iloc[12:] = [1., 3., 5., 4., 2.]   # B is rank 3, retained; add C.
    _patch_app_prices(monkeypatch, close, scores=scores, opens=opens)
    monkeypatch.setattr(config, "ROTATION_SCORE_MODE", "production", raising=False)
    monkeypatch.setattr(config, "ROTATION_TOP_K", 2)
    monkeypatch.setattr(config, "ROTATION_REBAL_DAYS", 4)
    monkeypatch.setattr(config, "ROTATION_DEFENSIVE", False)

    def no_live_access(*args, **kwargs):
        raise AssertionError("Default-model consistency uses only in-memory data")

    monkeypatch.setattr(data_pipeline, "get_conn", no_live_access)
    monkeypatch.setattr(data_pipeline.requests, "get", no_live_access)
    result = rotation.run_rotation(symbols=list(close))
    assert result["score_mode"] == "buffered_momentum"
    assert set(result["held"]) == {"B", "C"}
    assert result["selection_date"] == str(close.index[12].date())
    for symbol in ("B", "C", "D"):
        analysis = rotation.analyze_stock(symbol, symbols=list(close))
        assert analysis["score_mode"] == result["score_mode"]
        assert analysis["selected"] is (symbol in result["held"])
    retained = rotation.analyze_stock("B", symbols=list(close))
    assert retained["rank"] == 3 and not retained["in_top_k"]
    assert retained["selected"]  # The buffer, rather than a fresh top-K, decides.


def test_empty_real_data_response_never_creates_synthetic_trading_data(monkeypatch):
    monkeypatch.setattr(config, "DATA_SOURCE", "finmind")
    monkeypatch.setattr(config, "FALLBACK_TO_SYNTHETIC", False)
    monkeypatch.setattr(data_pipeline, "has_symbol", lambda symbol: False)
    fetches = []
    synthetic = []
    monkeypatch.setattr(data_pipeline, "fetch_real_data", lambda symbol: fetches.append(symbol))
    monkeypatch.setattr(data_pipeline, "seed_sample_data", lambda symbol: synthetic.append(symbol))
    with pytest.raises(RuntimeError, match="未取得.*真實行情"):
        data_pipeline.ensure_data("MISSING")
    assert fetches == ["MISSING"]
    assert synthetic == []
