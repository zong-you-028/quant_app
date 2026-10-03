"""Calendar rebalance timing sensitivity for the production rotation strategy."""
import numpy as np
import pandas as pd

import config
from core import evaluation as ev
from core import market_regime
from core.chip_data import fastsell_z_panel
from core.rotation import _load_panel, _select_picks


def _rebalance_dates(index, day):
    dates = []
    for _, group in pd.Series(index=index, data=index).groupby(index.to_period("M")):
        values = pd.DatetimeIndex(group.values)
        if day == "month_end":
            dates.append(values[-1])
        else:
            eligible = values[values.day >= int(day)]
            dates.append(eligible[0] if len(eligible) else values[-1])
    return pd.DatetimeIndex(dates)


def _metrics(returns):
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    return {
        "cagr": ev.annualized_return(returns),
        "sharpe": ev.annualized_sharpe(returns),
        "mdd": ev.max_drawdown(equity),
    }


def run_timing_lab(verbose=True):
    symbols = list(config.UNIVERSE)
    ret_df, mom_df, gate_df, _ = _load_panel(
        symbols, config.ROTATION_MOM_DAYS, config.ROTATION_MIN_OBS
    )
    idx, cols = ret_df.index, ret_df.columns
    try:
        fastsell = fastsell_z_panel(symbols, idx)
    except Exception:
        fastsell = None
    try:
        sox = market_regime.sox_regime_series(idx)
    except Exception:
        sox = pd.Series(1.0, index=idx)

    split = int(len(idx) * 0.6)
    oos_idx = idx[split:]
    rows = []
    for day in list(range(1, 29)) + ["month_end"]:
        weights = pd.DataFrame(np.nan, index=idx, columns=cols)
        for date in _rebalance_dates(idx, day):
            rank_row = mom_df.loc[date].dropna()
            if rank_row.empty:
                continue
            gate_row = gate_df.loc[date]
            fast_row = (fastsell.loc[date] if fastsell is not None
                        and date in fastsell.index else None)
            selected = _select_picks(
                rank_row, gate_row, config.ROTATION_TOP_K,
                config.ROTATION_ABS_MOM, config.ROTATION_ABS_THRESH,
                fastsell_row=fast_row,
                fastsell_z=config.ROTATION_FASTSELL_Z,
            )
            weight = pd.Series(0.0, index=cols)
            for symbol in selected:
                weight[symbol] = 1.0 / config.ROTATION_TOP_K
            weights.loc[date] = weight
        weights = weights.ffill().fillna(0.0)
        gross = (weights.shift(1).fillna(0.0) * ret_df.fillna(0.0)).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1)
        if len(turnover):
            turnover.iloc[0] = weights.iloc[0].abs().sum()
        net = gross - turnover * config.COST_PER_TURNOVER
        switch = sox.diff().abs().fillna(0.0) * config.COST_PER_TURNOVER
        net = net * sox - switch
        full = _metrics(net)
        oos = _metrics(net.reindex(oos_idx))
        rows.append({
            "day": day, "full_cagr": full["cagr"], "full_sharpe": full["sharpe"],
            "full_mdd": full["mdd"], "oos_cagr": oos["cagr"],
            "oos_sharpe": oos["sharpe"], "oos_mdd": oos["mdd"],
        })
    result = pd.DataFrame(rows)
    if verbose:
        shown = result.sort_values(["oos_sharpe", "oos_cagr"], ascending=False)
        print(shown.to_string(
            index=False,
            formatters={c: (lambda value: f"{value*100:.1f}%")
                        for c in ("full_cagr", "full_mdd", "oos_cagr", "oos_mdd")},
        ))
        print("\nOOS CAGR range:",
              f"{result['oos_cagr'].min()*100:.1f}%..{result['oos_cagr'].max()*100:.1f}%")
        print("OOS Sharpe range:",
              f"{result['oos_sharpe'].min():.2f}..{result['oos_sharpe'].max():.2f}")
    return result


if __name__ == "__main__":
    run_timing_lab()
