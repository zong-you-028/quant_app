import numpy as np
import pandas as pd
import pytest

from core.execution import backtest_open_execution
from core import rotation


def frame(values):
    return pd.DataFrame({"A": values}, index=pd.bdate_range("2026-09-21", periods=len(values)))


def test_entry_excludes_pre_entry_gap_and_exit_keeps_overnight_loss():
    # Monday signal enters Wednesday at 200, not Tuesday at 100.
    # Wednesday exit signal sells Friday at 99, keeping the 50% gap loss.
    target = frame([1, 1, 0, 0, 0])
    opens = frame([100, 100, 200, 220, 99])
    closes = frame([100, 150, 220, 198, 150])
    returns, turnover = backtest_open_execution(target, opens, closes)
    assert returns.to_numpy() == pytest.approx([0, 0, .1, -.1, -.5])
    assert turnover.to_numpy() == pytest.approx([0, 0, 1, 0, 1])


def test_cost_is_paid_on_execution_date():
    returns, turnover = backtest_open_execution(frame([1]*4), frame([100]*4),
                                               frame([100]*4), cost=.01)
    assert returns.iloc[:2].eq(0).all()
    assert returns.iloc[2] == pytest.approx(1/1.01 - 1)
    assert turnover.iloc[3] == 0


def test_missing_required_open_defers_until_tradable():
    returns, turns = backtest_open_execution(
        frame([1]*4), frame([100, 100, np.nan, 200]), frame([100, 100, 150, 220]))
    assert returns.tolist() == pytest.approx([0, 0, 0, .1])
    assert turns.tolist() == pytest.approx([0, 0, 0, 1])
    assert returns.attrs["deferred_dates"] == ["2026-09-23"]
    assert returns.attrs["effective_target_weights"] == {"A": 1.}
    assert returns.attrs["last_execution_date"] == "2026-09-24"
    assert not returns.attrs["execution_deferred"]


def test_pending_exit_does_not_change_current_model_holdings():
    target = frame([1, 1, 1, 0, 0, 0])
    prices = frame([100]*6)
    opens = frame([100, 100, 100, 100, 100, np.nan])
    result, _ = backtest_open_execution(target, opens, prices)
    assert result.attrs["effective_target_weights"] == {"A": 1.}
    assert result.attrs["execution_deferred"]
    assert result.attrs["last_execution_date"] == "2026-09-23"
    opens.iloc[-1, 0] = 100.
    completed, _ = backtest_open_execution(target, opens, prices)
    assert completed.attrs["effective_target_weights"] == {}
    assert not completed.attrs["execution_deferred"]


def test_weekend_is_not_an_extra_execution_bar():
    idx = pd.to_datetime(["2026-09-24", "2026-09-25", "2026-09-28"])
    prices = frame([100, 100, 110]).set_axis(idx)
    target = frame([1, 1, 1]).set_axis(idx)
    returns, turnover = backtest_open_execution(target, prices, prices)
    assert returns.eq(0).all()
    assert turnover.tolist() == pytest.approx([0, 0, 1])


def test_latest_target_retains_scheduled_selection(monkeypatch):
    idx = pd.bdate_range("2026-09-01", periods=6)
    close = pd.DataFrame({"A": [100]*6, "B": [100]*6}, index=idx)
    mom = pd.DataFrame({"A": [.2, .2, .2, .2, .1, .1],
                        "B": [.1, .1, .1, .1, .3, .3]}, index=idx)
    monkeypatch.setattr(rotation, "_load_panel",
                        lambda *args: (close.pct_change(), mom, mom, {"A":"A", "B":"B"}))
    monkeypatch.setattr(rotation, "load_ohlcv",
                        lambda symbol: pd.DataFrame({"open": 100., "close": 100.}, index=idx))
    monkeypatch.setattr(rotation, "ensure_data", lambda symbol: None)
    monkeypatch.setattr(rotation.config, "ROTATION_FASTSELL_GATE", False)
    result = rotation.run_rotation(symbols=["A", "B"], rebal_days=3, top_k=1,
                                   defensive=False, sox_gate=False, score_mode="production")
    assert result["holdings"] == ["A"]
    assert result["selection_date"] == str(idx[3].date())
    assert result["execution_lag"] == 2


def test_hold_shares_between_rebalances_instead_of_daily_equal_weight():
    idx = pd.bdate_range("2026-09-21", periods=4)
    targets = pd.DataFrame({"A": .5, "B": .5}, index=idx)
    opens = pd.DataFrame({"A": [100, 100, 200, 200], "B": [100]*4}, index=idx)
    closes = pd.DataFrame({"A": [100, 200, 200, 400], "B": [100]*4}, index=idx)
    returns, _ = backtest_open_execution(targets, opens, closes, lag=1)
    # 1 -> 1.5 -> 1.5 -> 2.5; an implicit daily reset would produce only 2.25.
    assert (1 + returns).prod() == pytest.approx(2.5)


def test_sox_status_does_not_read_beyond_signal_date(monkeypatch):
    from core import market_regime
    prices = pd.Series([100., 90., 200.], index=pd.bdate_range("2026-09-21", periods=3))
    monkeypatch.setattr(market_regime, "load_sox", lambda: prices)
    status = market_regime.sox_status(ma=2, asof=prices.index[1])
    assert status["close"] == 90.
    assert not status["risk_on"]
    assert status["asof"] == "2026-09-22"
