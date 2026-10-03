import numpy as np
import pandas as pd
import pytest

from core.overfitting_audit import cscv, shared_block_bootstrap, sharpes


def test_cash_score_is_zero_but_nonzero_constant_sharpe_is_rejected():
    assert sharpes(np.zeros((8, 2))).tolist() == [0., 0.]
    with pytest.raises(ValueError, match="undefined Sharpe"):
        sharpes(np.full((8, 2), .001))


def test_cscv_detects_a_stable_winner_and_preserves_tied_selection_weights():
    noise = np.tile([-.01, .01], 40)
    returns = pd.DataFrame({"weak": noise - .001, "strong": noise + .003, "tie": noise + .003})
    report, details = cscv(returns, blocks=4, active="strong")
    assert report["split_count"] == 6
    assert report["pbo"] == 0
    assert report["active_fixed_below_median_rate"] == 0
    assert report["by_is_winner"]["strong"]["selection_share"] == .5
    assert report["by_is_winner"]["tie"]["selection_share"] == .5
    assert details["weight"].sum() == 6


def test_cscv_flags_a_regime_winner_that_reverses_in_complement():
    noise = np.tile([-.01, .01], 20)
    drift = np.repeat([.004, .004, -.004, -.004], 10)
    returns = pd.DataFrame({"A": noise + drift, "B": noise - drift, "C": noise})
    report, details = cscv(returns, blocks=4, active="A")
    assert report["pbo"] == 1
    assert details["rank_logit"].le(0).all()
    # Ties are symmetric: permuting candidate columns must not change PBO.
    reordered, _ = cscv(returns[["C", "B", "A"]], blocks=4, active="A")
    assert reordered["pbo"] == report["pbo"]


def test_shared_bootstrap_preserves_paired_returns_and_family_adjustment():
    rng = np.random.default_rng(123)
    baseline = rng.normal(.0003, .01, 180)
    alpha = np.tile([-.001, .002, .003], 60)
    returns = pd.DataFrame({"production": baseline, "A": baseline + alpha, "B": baseline + alpha})
    result = shared_block_bootstrap(returns, block=10, draws=300, seed=7)
    again = shared_block_bootstrap(returns, block=10, draws=300, seed=7)
    assert result == again
    assert result["candidates"]["A"] == result["candidates"]["B"]
    a = result["candidates"]["A"]
    assert a["annualized_mean_active"] == pytest.approx(alpha.mean() * 252)
    # Absolute-max intervals are symmetric; percentile intervals can be skewed
    # and need not be nested within them for a finite bootstrap sample.
    assert a["simultaneous_95_lower"] < a["annualized_mean_active"] < a["simultaneous_95_upper"]
    assert a["simultaneous_95_upper"] - a["annualized_mean_active"] == pytest.approx(
        a["annualized_mean_active"] - a["simultaneous_95_lower"])
    assert 0 <= a["max_mean_adjusted_p"] <= 1


def test_identical_baseline_has_no_spurious_active_evidence():
    baseline = np.tile([-.01, .012], 40)
    result = shared_block_bootstrap(pd.DataFrame({"production": baseline, "same": baseline}),
                                    block=10, draws=200)
    candidate = result["candidates"]["same"]
    assert candidate["annualized_mean_active"] == 0
    assert candidate["simultaneous_95_lower"] == candidate["simultaneous_95_upper"] == 0
    assert candidate["max_mean_adjusted_p"] == result["family_max_mean_p"] == 1


@pytest.mark.parametrize("corrupt", ["missing", "infinite", "loss"])
def test_bad_return_data_is_rejected_instead_of_silently_dropping_dates(corrupt):
    returns = pd.DataFrame({"A": [.01, -.01] * 8, "B": [.02, -.01] * 8})
    returns.loc[5, "A"] = {"missing": np.nan, "infinite": np.inf, "loss": -1}[corrupt]
    with pytest.raises(ValueError, match="complete, finite"):
        cscv(returns, blocks=4)


def test_cscv_equal_blocks_report_tail_exclusion_and_reject_invalid_block_count():
    returns = pd.DataFrame({"A": [.01, -.01] * 9, "B": [.02, -.01] * 9})
    report, _ = cscv(returns, blocks=4)
    assert report["rows_used"] == 16
    assert report["tail_rows_excluded"] == 2
    with pytest.raises(ValueError, match="even block"):
        cscv(returns, blocks=5)
