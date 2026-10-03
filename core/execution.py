"""Daily-bar execution: delayed signals, opening auctions, and held shares."""
import numpy as np
import pandas as pd


def backtest_open_execution(targets, opens, closes, lag=2, cost=0.0, rebalance=None):
    """Execute t's target at t+lag open; unchanged targets retain shares.

    Prices must share the same adjustment basis. Missing held-stock closes use
    the last mark; missing required opens defer the entire rebalance until tradable.
    Deferred dates are returned in returns.attrs['deferred_dates'].
    """
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("execution lag must be a positive integer")
    if not 0 <= cost < 1:
        raise ValueError("cost must be between zero and one")
    targets = targets.fillna(0.0)
    if (targets < 0).any().any() or (targets.sum(axis=1) > 1.000001).any():
        raise ValueError("long-only target weights must sum to at most one")
    opens = opens.reindex(index=targets.index, columns=targets.columns)
    closes = closes.reindex_like(opens).where(lambda x: x > 0).ffill()
    intended = targets.shift(lag).fillna(0.0)
    scheduled = (rebalance.reindex(targets.index).fillna(False).shift(lag, fill_value=False)
                 if rebalance is not None else pd.Series(False, index=targets.index))
    shares = np.zeros(len(targets.columns))
    previous = np.zeros(len(shares))
    cash, nav = 1.0, 1.0
    returns, turns, deferred = [], [], []
    pending = False
    for i in range(len(targets)):
        target = intended.iloc[i].to_numpy(dtype=float)
        opening = opens.iloc[i].to_numpy(dtype=float)
        closing = closes.iloc[i].to_numpy(dtype=float)
        turnover = 0.0
        if pending or scheduled.iloc[i] or not np.allclose(target, previous, rtol=0, atol=1e-12):
            required = (shares != 0) | (target != 0)
            if np.any(required & (~np.isfinite(opening) | (opening <= 0))):
                deferred.append(str(targets.index[i].date()))
                pending = True
            else:
                pending = False
                safe_open = np.where(required, opening, 1.0)
                old_value = shares * safe_open
                wealth = cash + old_value.sum()
                # Pay costs from wealth before allocating; solve the fee equation.
                lo, hi = 0.0, wealth
                for _ in range(45):
                    investable = (lo + hi) / 2
                    fee = np.abs(target * investable - old_value).sum() * cost
                    if investable + fee > wealth:
                        hi = investable
                    else:
                        lo = investable
                investable = (lo + hi) / 2
                traded = np.abs(target * investable - old_value).sum()
                turnover = traded / wealth if wealth else 0.0
                shares = target * investable / safe_open
                cash = wealth - (target * investable).sum() - traded * cost
                previous = target.copy()
        if np.any((shares != 0) & ~np.isfinite(closing)):
            raise ValueError("Missing valuation for a held stock")
        new_nav = cash + np.sum(shares * np.where(shares != 0, closing, 0.0))
        returns.append(new_nav / nav - 1.0)
        turns.append(turnover)
        nav = new_nav
    result = pd.Series(returns, index=targets.index)
    result.attrs["deferred_dates"] = deferred
    return result, pd.Series(turns, index=targets.index)
