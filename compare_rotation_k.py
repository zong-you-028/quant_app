"""Compare concentrated momentum portfolios on identical data and assumptions."""
import pandas as pd

import config
from core import evaluation as ev
from core.signal_lab import DELISTED, backtest_signal, load_closes, sig_skip10


def metrics(returns):
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    return {
        "cagr": ev.annualized_return(returns),
        "sharpe": ev.annualized_sharpe(returns),
        "mdd": ev.max_drawdown(equity),
        "total": float(equity.iloc[-1] - 1.0),
    }


def main():
    symbols = list(config.UNIVERSE) + DELISTED
    close_df, ret_df = load_closes(symbols, config.ROTATION_MIN_OBS)
    signal_df = sig_skip10(close_df, ret_df)
    gate_df = close_df.pct_change(60, fill_method=None)
    market = ret_df.fillna(0.0).mean(axis=1)
    split = int(len(ret_df) * 0.6)
    recent_start = ret_df.index[-1] - pd.Timedelta(days=config.EVAL_LOOKBACK_DAYS)
    windows = {
        "full": ret_df.index,
        "oos40": ret_df.index[split:],
        "recent1y": ret_df.index[ret_df.index >= recent_start],
    }

    series = {"market": market}
    for k in (3, 5, 8):
        series[f"k{k}"] = backtest_signal(
            signal_df, gate_df, ret_df, top_k=k,
            rebal=config.ROTATION_REBAL_DAYS,
            cost=config.COST_PER_TURNOVER,
        )

    print(f"data={ret_df.index[0].date()}..{ret_df.index[-1].date()} symbols={ret_df.shape[1]}")
    print("signal=skip10 gate=raw60>0 rebalance=20 cost=0.003")
    print(f"{'window':<10} {'portfolio':<8} {'CAGR':>9} {'Sharpe':>9} {'MDD':>9} {'Total':>9}")
    for window, idx in windows.items():
        for name, returns in series.items():
            m = metrics(returns.reindex(idx))
            print(f"{window:<10} {name:<8} {m['cagr']*100:>8.1f}% {m['sharpe']:>9.2f} "
                  f"{m['mdd']*100:>8.1f}% {m['total']*100:>8.1f}%")
        print()


if __name__ == "__main__":
    main()
