"""Offline candidate research with purged models and delayed open execution.

Candidate definitions are fixed before evaluation. Development uses prequential
predictions; the final 504 bars are excluded from strategy selection. This is a
retrospective split of an already researched snapshot, not a new live holdout.
No downloads, database writes, model-weight downloads, or production changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config
from core import evaluation as ev
from core.execution import backtest_open_execution


@dataclass(frozen=True)
class ResearchSettings:
    train_min: int = 756
    train_window: int = 1260
    refit_days: int = 126
    holdout_days: int = 504
    horizons: tuple[int, ...] = (20, 60)
    top_k: int = config.ROTATION_TOP_K
    rebalance_days: int = config.ROTATION_REBAL_DAYS
    lag: int = config.ROTATION_EXECUTION_LAG
    cost: float = config.COST_PER_TURNOVER
    seed: int = 42

    def validate(self):
        if min(self.train_min, self.train_window, self.refit_days,
               self.holdout_days, self.top_k, self.rebalance_days, self.lag) < 1:
            raise ValueError("research intervals must be positive")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("positive label horizons required")
        if self.train_min <= self.lag + max(self.horizons):
            raise ValueError("training history must exceed label maturity")
        if not 0 <= self.cost < 1:
            raise ValueError("invalid transaction cost")


def load_snapshot(symbols=None, db_path=None):
    """Use SQLite read-only mode, ETF trading calendar, and cached SOX only."""
    symbols = list(dict.fromkeys(symbols or config.UNIVERSE))
    path = Path(db_path or config.DB_PATH).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        placeholders = ",".join("?" for _ in [*symbols, config.BENCHMARK_SYMBOL])
        bars = pd.read_sql_query(
            f"SELECT symbol,date,open,close,volume FROM ohlcv "
            f"WHERE symbol IN ({placeholders}) ORDER BY date,symbol", conn,
            params=[*symbols, config.BENCHMARK_SYMBOL])
        chips = pd.read_sql_query(
            "SELECT symbol,date,big_shares,total_shares FROM chip_weekly ORDER BY date", conn)
    bars["date"] = pd.to_datetime(bars["date"])
    benchmark = bars[bars.symbol == config.BENCHMARK_SYMBOL].set_index("date")
    if benchmark.empty:
        raise RuntimeError("real benchmark bars required")
    index = pd.DatetimeIndex(benchmark.index).sort_values()
    stock_bars = bars[bars.symbol.isin(symbols)]
    panels = {key: stock_bars.pivot(index="date", columns="symbol", values=key)
              .reindex(index=index, columns=symbols).astype(float)
              for key in ("open", "close", "volume")}
    available = panels["close"].notna().sum() >= config.ROTATION_MIN_OBS
    panels = {key: panel.loc[:, available] for key, panel in panels.items()}
    if panels["close"].empty:
        raise RuntimeError("no stock history available")
    missing_symbols = [s for s in symbols if s not in panels["close"].columns]
    holding = pd.DataFrame(np.nan, index=index, columns=panels["close"].columns)
    if not chips.empty:
        chips["date"] = pd.to_datetime(chips["date"])
        chips["ratio"] = chips.big_shares / chips.total_shares.replace(0, np.nan)
        holding = chips.pivot(index="date", columns="symbol", values="ratio")
        # Union before filling preserves observations reported on non-ETF dates.
        holding = holding.reindex(holding.index.union(index)).sort_index().ffill()
        holding = holding.reindex(index=index, columns=panels["close"].columns)
    change = holding.diff(config.ROTATION_FASTSELL_WIN)
    fastsell = change / change.rolling(250).std().replace(0, np.nan)
    sox_path = Path(config.DATA_DIR) / "sox.csv"
    if config.ROTATION_SOX_GATE and not sox_path.exists():
        raise RuntimeError("SOX cache required; research never silently downloads")
    sox_gate = pd.Series(1.0, index=index)
    if config.ROTATION_SOX_GATE:
        sox = pd.read_csv(sox_path, index_col=0).iloc[:, 0].astype(float)
        sox.index = pd.to_datetime(sox.index)
        sox = sox.sort_index()
        if len(sox) < config.ROTATION_SOX_MA:
            raise RuntimeError("insufficient SOX cache")
        sox_gate = (sox > sox.rolling(config.ROTATION_SOX_MA).mean()).astype(float)
        # Execution applies the single shared lag; do not delay this gate twice.
        sox_gate = sox_gate.reindex(index, method="ffill").fillna(1.0)
    digest = hashlib.sha256(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
    digest.update(pd.util.hash_pandas_object(chips, index=False).values.tobytes())
    if config.ROTATION_SOX_GATE:
        digest.update(sox_path.read_bytes())
    return {**panels, "benchmark": benchmark["close"].astype(float),
            "fastsell": fastsell, "sox_gate": sox_gate,
            "metadata": {"price_asof": str(index[-1].date()),
                         "start": str(index[0].date()), "symbols": list(panels["close"]),
                         "missing_symbols": missing_symbols,
                         "snapshot_sha256": digest.hexdigest(),
                         "sox_asof": str(sox.index[-1].date()) if config.ROTATION_SOX_GATE else None}}


def features_and_rules(close, volume):
    """Causal features, daily cross-sectional ranks, and fixed rule scores."""
    ret = close.pct_change(fill_method=None)
    valid = close.notna() & (close > 0) & close.notna().rolling(252).sum().ge(252)
    raw = {}
    for horizon in (20, 60, 120, 252):
        raw[f"mom_{horizon}"] = close.shift(10) / close.shift(horizon) - 1
    raw["reversal_5"] = -close.pct_change(5, fill_method=None)
    raw["gap_60"] = close / close.rolling(60).mean() - 1
    raw["gap_200"] = close / close.rolling(200).mean() - 1
    raw["low_vol_20"] = -ret.rolling(20).std()
    raw["low_vol_60"] = -ret.rolling(60).std()
    raw["low_downside"] = -ret.clip(upper=0).rolling(60).std()
    raw["high_252"] = close / close.rolling(252).max() - 1
    raw["stability"] = (ret > 0).astype(float).where(ret.notna()).rolling(60).mean()
    raw["volume_ratio"] = volume / volume.rolling(60).mean().replace(0, np.nan)
    ranks = {k: v.where(valid).replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)
             for k, v in raw.items()}
    frames = {s: pd.DataFrame({k: v[s] for k, v in ranks.items()}) for s in close}
    X = pd.concat(frames, names=["symbol", "date"]).swaplevel().sort_index()
    X = X.dropna()
    multi = sum(ranks[f"mom_{h}"] for h in (60, 120, 252)) / 3
    quality = .5 * multi + .25 * ranks["high_252"] + .25 * ranks["stability"]
    production = close.shift(config.ROTATION_SKIP_DAYS) / close.shift(config.ROTATION_MOM_DAYS) - 1
    return X, {"production": production, "multi_momentum": multi,
               "trend_quality": quality, "buffered_momentum": multi,
               "vol_control": multi}, valid


def make_targets(opens, settings):
    """Rank future returns from executable entry open to horizon exit open.

    Require every horizon to be observed. label_end is the latest observation
    used anywhere in a row's target, so purge includes both horizon and lag.
    """
    components = []
    for horizon in settings.horizons:
        future = opens.shift(-(settings.lag + horizon)) / opens.shift(-settings.lag) - 1
        future = future.where((opens.shift(-(settings.lag + horizon)) > 0) &
                              (opens.shift(-settings.lag) > 0))
        components.append(future.rank(axis=1, pct=True))
    target = sum(components) / len(components)
    # Sum propagates NaN instead of silently accepting a shorter last horizon.
    y = target.stack(future_stack=True).dropna().rename("target")
    y.index.names = ["date", "symbol"]
    label_end = pd.Series(opens.index, index=opens.index).shift(
        -(settings.lag + max(settings.horizons)))
    return y, label_end


def training_rows(X, y, label_end, boundary, train_start):
    """No training outcome may touch the model's first prediction date."""
    dates = X.index.get_level_values("date")
    maturity = label_end.reindex(dates).to_numpy()
    mask = (dates >= train_start) & (dates < boundary) & (maturity < np.datetime64(boundary))
    train = X.loc[mask].join(y, how="inner").dropna(subset=["target"])
    return train.sort_index()


def walk_forward_scores(X, y, label_end, index, settings, verbose=True):
    """Fixed shallow models; refit with only matured labels on a fixed schedule."""
    import lightgbm as lgb
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge

    names = ("lgb_rank", "lambda_rank", "xgb_rank", "ridge_rank")
    outputs = {name: [] for name in names}
    audit = []
    common = dict(n_estimators=140, learning_rate=.03, max_depth=3,
                  num_leaves=7, min_child_samples=180, reg_alpha=1.0,
                  reg_lambda=8.0, colsample_bytree=.8, subsample=.8,
                  subsample_freq=1, random_state=settings.seed, n_jobs=4, verbosity=-1)
    dates = X.index.get_level_values("date")
    for start in range(settings.train_min, len(index), settings.refit_days):
        end = min(start + settings.refit_days, len(index))
        boundary = index[start]
        train = training_rows(X, y, label_end, boundary,
                              index[max(0, start - settings.train_window)])
        test = X.loc[(dates >= boundary) & (dates <= index[end - 1])]
        if len(train) < 1000 or test.empty:
            raise RuntimeError(f"insufficient training or prediction rows at {boundary}")
        group = train.groupby(level="date", sort=True).size().to_numpy()
        ages = (boundary - train.index.get_level_values("date")).days.to_numpy()
        recency = np.power(.5, ages / 730.5)
        # Each date has equal aggregate weight; recency decays with a two-year half-life.
        per_date = train.groupby(level="date")["target"].transform("size").to_numpy()
        sample_weight = recency * len(index[:start]) / per_date
        sample_weight /= sample_weight.mean()
        Xt, yt = train[X.columns], train["target"]
        models = {
            "lgb_rank": lgb.LGBMRegressor(objective="regression", **common),
            "lambda_rank": lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1, 2, 3, 4], **common),
            "xgb_rank": XGBRegressor(n_estimators=140, max_depth=3, learning_rate=.03,
                                      min_child_weight=80, reg_alpha=1, reg_lambda=8,
                                      subsample=.8, colsample_bytree=.8, tree_method="hist",
                                      random_state=settings.seed, n_jobs=4),
            "ridge_rank": Ridge(alpha=100.0),
        }
        for name, model in models.items():
            if name == "lambda_rank":
                relevance = np.minimum((yt * 5).astype(int), 4)
                model.fit(Xt, relevance, group=group, sample_weight=sample_weight)
            else:
                model.fit(Xt, yt, sample_weight=sample_weight)
            outputs[name].append(pd.Series(model.predict(test), index=test.index))
        mature = label_end.reindex(train.index.get_level_values("date")).max()
        audit.append({"test_start": str(boundary.date()), "test_end": str(index[end - 1].date()),
                      "train_start": str(train.index.get_level_values("date").min().date()),
                      "train_end": str(train.index.get_level_values("date").max().date()),
                      "latest_label_end": str(mature.date()), "rows": len(train)})
        if verbose:
            print(f"model fold {len(audit):02d}: {boundary.date()}..{index[end-1].date()}, "
                  f"{len(train)} rows, last label {mature.date()}", flush=True)
    result = {name: pd.concat(parts).unstack("symbol").reindex(index)
              for name, parts in outputs.items()}
    result["rank_ensemble"] = sum(result[n].rank(axis=1, pct=True) for n in names) / len(names)
    return result, audit


def portfolio_targets(scores, snapshot, settings, name, rebalance_offset=0):
    """Monthly holdings, cash for failed slots, optional rank buffer/vol budget."""
    if not 0 <= rebalance_offset < settings.rebalance_days:
        raise ValueError("rebalance offset must be inside the rebalance interval")
    close = snapshot["close"]
    scores = scores.reindex_like(close).replace([np.inf, -np.inf], np.nan)
    gate = close.pct_change(config.ROTATION_MOM_DAYS, fill_method=None)
    targets = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    schedule = pd.Series(False, index=close.index)
    current = pd.Series(0.0, index=close.columns)
    market_vol = close.pct_change(fill_method=None).mean(axis=1).rolling(60).std() * np.sqrt(252)
    for i, date in enumerate(close.index):
        if i < settings.train_min:
            continue
        if i % settings.rebalance_days == rebalance_offset:
            schedule.loc[date] = True
            ranked = scores.loc[date].dropna().sort_values(ascending=False, kind="stable")
            picks = list(ranked.head(settings.top_k).index)
            if name == "buffered_momentum":
                retain = [s for s in ranked.head(2 * settings.top_k).index if current[s] > 0]
                picks = (retain + [s for s in ranked.index if s not in retain])[:settings.top_k]
            current = pd.Series(0.0, index=close.columns)
            for symbol in picks:
                if config.ROTATION_ABS_MOM and not gate.loc[date, symbol] > config.ROTATION_ABS_THRESH:
                    continue
                z = snapshot["fastsell"].loc[date, symbol]
                if config.ROTATION_FASTSELL_GATE and pd.notna(z) and z < config.ROTATION_FASTSELL_Z:
                    continue
                current[symbol] = 1 / settings.top_k
            if name == "vol_control" and pd.notna(market_vol.loc[date]) and market_vol.loc[date] > 0:
                current *= min(1.0, .15 / market_vol.loc[date])
        targets.loc[date] = current
    targets = targets.mul(snapshot["sox_gate"], axis=0)
    return targets, schedule


def staggered_targets(scores, snapshot, settings, name):
    """Average four fixed phase targets in one executable aggregate portfolio.

    Each phase holds 20 bars with the default settings. The aggregate portfolio
    resets weights whenever any phase rebalances (normally every five bars).
    This is an exploratory remedy for the observed phase sensitivity, and must
    not enter the original development selection retrospectively.
    """
    phases = sorted(set((0, settings.rebalance_days // 4,
                         settings.rebalance_days // 2, 3 * settings.rebalance_days // 4)))
    targets, schedules = [], []
    for offset in phases:
        target, schedule = portfolio_targets(scores, snapshot, settings, name, offset)
        targets.append(target)
        schedules.append(schedule)
    return sum(targets) / len(targets), pd.concat(schedules, axis=1).any(axis=1)


def staggered_diagnostic(scores, snapshot, settings, selected, holdout_idx, oos_idx):
    """Compare a fixed phase average, without changing original selection."""
    benchmark = snapshot["benchmark"].pct_change(fill_method=None)
    results = {}
    for name in dict.fromkeys((selected, "production")):
        target, schedule = staggered_targets(scores[name], snapshot, settings, name)
        returns, turnover = backtest_open_execution(target, snapshot["open"], snapshot["close"],
                                                    lag=settings.lag, cost=settings.cost, rebalance=schedule)
        stress = []
        for cost, lag in ((.005, settings.lag), (.008, settings.lag), (settings.cost, settings.lag + 1)):
            r, t = backtest_open_execution(target, snapshot["open"], snapshot["close"],
                                           lag=lag, cost=cost, rebalance=schedule)
            stress.append({"cost": cost, "lag": lag, **metrics(r.reindex(holdout_idx), t, benchmark)})
        results[name] = {"final_split": metrics(returns.reindex(holdout_idx), turnover, benchmark),
                         "full_common": metrics(returns.reindex(oos_idx), turnover, benchmark),
                         "stress": stress}
    return {"status": "exploratory_after_final_split_was_seen",
            "included_in_original_selection": False, "strategies": results}


def metrics(returns, turnover, benchmark):
    r = returns.dropna()
    b = benchmark.reindex(r.index)
    if b.isna().any():
        raise RuntimeError("benchmark gaps in evaluation window")
    equity = ev.equity_curve(r)
    # Include starting capital, so an initial loss counts as a drawdown.
    peak = equity.cummax().clip(lower=1.0)
    return {"cagr": ev.annualized_return(r), "sharpe": ev.annualized_sharpe(r),
            "mdd": float((equity / peak - 1).min()),
            "benchmark_cagr": ev.annualized_return(b),
            "alpha": ev.active_return(r, b), "ir": ev.information_ratio(r, b),
            "annual_turnover": float(turnover.reindex(r.index).sum() * 252 / len(r))}


def choose_candidate(development):
    """Require development improvement over production before selection."""
    baseline = development["production"]
    eligible = [name for name, row in development.items()
                if row["cagr"] > baseline["cagr"] and row["sharpe"] > baseline["sharpe"]
                and row["mdd"] >= baseline["mdd"] - .05
                and row["annual_turnover"] <= max(18., baseline["annual_turnover"])]
    return max(eligible, key=lambda n: development[n]["sharpe"]) if eligible else "production"


def block_bootstrap(active, block=20, draws=2000, seed=42):
    """Exploratory CI for annualized mean active return, not CAGR or Sharpe."""
    values = active.dropna().to_numpy()
    if len(values) < 2 * block:
        return {"mean": None, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        sample = np.concatenate([values[(s + np.arange(block)) % len(values)] for s in starts])[:len(values)]
        means.append(sample.mean() * 252)
    lower, upper = np.quantile(means, [.025, .975])
    return {"mean": float(values.mean() * 252), "lower": float(lower), "upper": float(upper)}


def run_rule_candidate(name="trend_quality", settings=None, symbols=None):
    """Replay a rule candidate without training models or changing app defaults.

    Returns historical targets, execution returns and the snapshot date. The
    caller must check freshness before treating any target as actionable.
    """
    settings = settings or ResearchSettings()
    settings.validate()
    snapshot = load_snapshot(symbols)
    _, scores, _ = features_and_rules(snapshot["close"], snapshot["volume"])
    if name not in scores:
        raise ValueError(f"rule candidate must be one of {tuple(scores)}")
    if len(snapshot["close"]) <= settings.train_min:
        raise RuntimeError("insufficient candidate history")
    target, schedule = portfolio_targets(scores[name], snapshot, settings, name)
    returns, turnover = backtest_open_execution(target, snapshot["open"], snapshot["close"],
                                                lag=settings.lag, cost=settings.cost, rebalance=schedule)
    return {"candidate": name, "data": snapshot["metadata"], "targets": target,
            "returns": returns, "turnover": turnover, "rebalance": schedule}


def run_research(settings=None, symbols=None, output_dir=None, verbose=True):
    settings = settings or ResearchSettings()
    settings.validate()
    snapshot = load_snapshot(symbols)
    index = snapshot["close"].index
    if len(index) < settings.train_min + settings.holdout_days + 252:
        raise RuntimeError("need at least one development year plus final holdout")
    X, scores, _ = features_and_rules(snapshot["close"], snapshot["volume"])
    y, label_end = make_targets(snapshot["open"], settings)
    models, audit = walk_forward_scores(X, y, label_end, index, settings, verbose)
    scores.update(models)
    scores["ai_momentum_blend"] = .5 * scores["rank_ensemble"] + .5 * scores["multi_momentum"]
    development_idx = index[settings.train_min:-settings.holdout_days]
    holdout_idx = index[-settings.holdout_days:]
    oos_idx = index[settings.train_min:]
    benchmark = snapshot["benchmark"].pct_change(fill_method=None)
    paths, turns, weights, schedules = {}, {}, {}, {}
    development, holdout, all_oos = {}, {}, {}
    for name, signal in scores.items():
        target, schedule = portfolio_targets(signal, snapshot, settings, name)
        # All candidates start with cash on the same signal date. No hidden
        # inventory in rules while ML candidates are still warming up.
        target.loc[index < index[settings.train_min]] = 0.0
        schedule.loc[index < index[settings.train_min]] = False
        returns, turnover = backtest_open_execution(target, snapshot["open"], snapshot["close"],
                                                    lag=settings.lag, cost=settings.cost, rebalance=schedule)
        paths[name], turns[name], weights[name], schedules[name] = returns, turnover, target, schedule
        development[name] = metrics(returns.reindex(development_idx), turnover, benchmark)
        holdout[name] = metrics(returns.reindex(holdout_idx), turnover, benchmark)
        all_oos[name] = metrics(returns.reindex(oos_idx), turnover, benchmark)
        if verbose:
            print(f"{name:22s} dev Sharpe {development[name]['sharpe']:.2f} | "
                  f"holdout CAGR {holdout[name]['cagr']:.1%}, "
                  f"Sharpe {holdout[name]['sharpe']:.2f}", flush=True)
    selected = choose_candidate(development)
    selected_holdout, baseline_holdout = holdout[selected], holdout["production"]
    ci = block_bootstrap((paths[selected] - paths["production"]).reindex(holdout_idx), seed=settings.seed)
    stress = []
    for cost, lag in ((.005, settings.lag), (.008, settings.lag), (settings.cost, settings.lag + 1)):
        row = {"cost": cost, "lag": lag}
        for name in dict.fromkeys((selected, "production")):
            r, t = backtest_open_execution(weights[name], snapshot["open"], snapshot["close"],
                                           lag=lag, cost=cost, rebalance=schedules[name])
            row[name] = metrics(r.reindex(holdout_idx), t, benchmark)
        stress.append(row)
    # Fixed schedule perturbations are diagnostics only; never select a phase.
    phase_stress = []
    for offset in sorted(set((0, settings.rebalance_days // 4,
                              settings.rebalance_days // 2, 3 * settings.rebalance_days // 4))):
        row = {"offset": offset}
        for name in dict.fromkeys((selected, "production")):
            target, schedule = portfolio_targets(scores[name], snapshot, settings, name, offset)
            r, t = backtest_open_execution(target, snapshot["open"], snapshot["close"],
                                           lag=settings.lag, cost=settings.cost, rebalance=schedule)
            row[name] = metrics(r.reindex(holdout_idx), t, benchmark)
        phase_stress.append(row)
    years = []
    for year, dates in pd.Series(oos_idx, index=oos_idx).groupby(oos_idx.year):
        idx = pd.DatetimeIndex(dates.values)
        years.append({"year": int(year), **{
            name: metrics(paths[name].reindex(idx), turns[name], benchmark)
            for name in dict.fromkeys((selected, "production"))}})
    evidence = {"selected_differs": selected != "production",
                "higher_holdout_cagr": selected_holdout["cagr"] > baseline_holdout["cagr"],
                "higher_holdout_sharpe": selected_holdout["sharpe"] > baseline_holdout["sharpe"],
                "not_worse_holdout_drawdown": selected_holdout["mdd"] >= baseline_holdout["mdd"],
                "active_mean_ci_positive": ci["lower"] is not None and ci["lower"] > 0}
    report = {"settings": asdict(settings), "data": snapshot["metadata"],
              "production_gates": {k: getattr(config, k) for k in (
                  "ROTATION_MOM_DAYS", "ROTATION_SKIP_DAYS", "ROTATION_ABS_MOM",
                  "ROTATION_ABS_THRESH", "ROTATION_FASTSELL_GATE", "ROTATION_FASTSELL_WIN",
                  "ROTATION_FASTSELL_Z", "ROTATION_SOX_GATE", "ROTATION_SOX_MA")},
              "development_start": str(development_idx[0].date()),
              "development_end": str(development_idx[-1].date()),
              "holdout_start": str(holdout_idx[0].date()), "holdout_end": str(holdout_idx[-1].date()),
              "selected_on_development": selected, "development": development,
              "holdout": holdout, "all_prequential": all_oos, "model_folds": audit,
              "stress": stress, "rebalance_phase_stress": phase_stress,
              "exploratory_staggered": staggered_diagnostic(
                  scores, snapshot, settings, selected, holdout_idx, oos_idx),
              "yearly": years, "holdout_active_mean_ci": ci,
              "evidence": evidence, "production_changed": False,
              "deferred_days": {n: len(r.attrs.get("deferred_dates", [])) for n, r in paths.items()},
              "limitations": [
                  "Historical snapshot ends at price_asof; these are not current trading signals.",
                  "Final split is retrospective; earlier research has already seen this snapshot.",
                  "Current 50-stock universe retains selection/survivorship bias; no delisted execution model.",
                  "Cached adjusted prices and chip dates are used as provided; release times not verified.",
                  "No limit-lock, volume/capacity, slippage beyond flat costs, or cash interest modeling.",
                  "Bootstrap interval is exploratory and not corrected for multiple strategy comparisons."]}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        pd.DataFrame({**{n: r.reindex(oos_idx) for n, r in paths.items()},
                      "006208": benchmark.reindex(oos_idx)}).to_csv(output / "daily_returns.csv", index_label="date")
        pd.DataFrame(holdout).T.to_csv(output / "holdout_metrics.csv", index_label="candidate")
        write_summary(report, output / "summary.md")
        write_chart(paths, benchmark, oos_idx, holdout_idx, selected, output / "comparison.png")
    return report


def write_chart(paths, benchmark, oos_idx, holdout_idx, selected, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    for col, (idx, title) in enumerate(((oos_idx, "Full historical comparison"),
                                       (holdout_idx, "Final retrospective split"))):
        for name, values in (("production", paths["production"]),
                             (selected, paths[selected]), ("006208", benchmark)):
            equity = ev.equity_curve(values.reindex(idx))
            axes[0, col].plot(idx, equity, label=name, linewidth=1.5)
            axes[1, col].plot(idx, (equity / equity.cummax().clip(lower=1.0) - 1) * 100, linewidth=1.2)
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Wealth (starting capital = 1)")
        axes[1, col].set_ylabel("Drawdown (%)")
        axes[0, col].legend(loc="upper left", fontsize=9)
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    fig.suptitle("Fixed candidate research: costs included, t+2 open execution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(report, path):
    selected = report["selected_on_development"]
    lines = ["# 固定候選策略與模型研究", "",
             f"行情快照：{report['data']['start']}～{report['data']['price_asof']}，{len(report['data']['symbols'])} 檔。",
             f"開發區段：{report['development_start']}～{report['development_end']}。",
             f"本次選型排除區段：{report['holdout_start']}～{report['holdout_end']}。",
             f"只依開發區段選出的候選：`{selected}`；正式策略未變更。", "",
             "下表為扣成本、延遲開盤成交的末段結果；所有候選使用相同交易日與股票池。", "",
             "| 候選 | 開發 Sharpe | 末段 CAGR | 末段 Sharpe | 末段最大回撤 | 年換手 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in report["holdout"].items():
        lines.append(f"| {name} | {report['development'][name]['sharpe']:.2f} | {m['cagr']:.2%} | "
                     f"{m['sharpe']:.2f} | {m['mdd']:.2%} | {m['annual_turnover']:.1f} |")
    benchmark = report["holdout"]["production"]["benchmark_cagr"]
    lines += ["", f"同區段 006208 買進持有 CAGR：{benchmark:.2%}。",
              "", "選定候選相對正式策略的年化平均日超額報酬，20 日區塊 bootstrap 95% 區間："]
    ci = report["holdout_active_mean_ci"]
    lines.append(f"{ci['lower']:.2%}～{ci['upper']:.2%}。這不是 CAGR 差異的信賴區間，也未做多重比較校正。"
                 if ci["lower"] is not None else "區段不足，未估計區間。")
    lines += ["", "固定候選在完整共同區段的比較：", "",
              "| 策略 | CAGR | Sharpe | 最大回撤 |",
              "|---|---:|---:|---:|"]
    for name in dict.fromkeys(("production", selected)):
        m = report["all_prequential"][name]
        lines.append(f"| {name} | {m['cagr']:.2%} | {m['sharpe']:.2f} | {m['mdd']:.2%} |")
    lines += ["", "壓力測試（仍只比較原先選定候選，不依結果換策略）：", "",
              "| 成本／成交延遲 | 候選 CAGR | 正式策略 CAGR | 候選最大回撤 |",
              "|---|---:|---:|---:|"]
    for row in report["stress"]:
        m, b = row[selected], row["production"]
        lines.append(f"| {row['cost']:.1%} / t+{row['lag']} | {m['cagr']:.2%} | {b['cagr']:.2%} | {m['mdd']:.2%} |")
    lines += ["", "換股排程位移敏感度（0、5、10、15 日；不選最佳排程）：", "",
              "| 位移 | 候選 CAGR | 正式策略 CAGR | 候選 Sharpe |",
              "|---|---:|---:|---:|"]
    for row in report["rebalance_phase_stress"]:
        m, b = row[selected], row["production"]
        lines.append(f"| {row['offset']} | {m['cagr']:.2%} | {b['cagr']:.2%} | {m['sharpe']:.2f} |")
    phase_wins = sum(row[selected]["cagr"] > row["production"]["cagr"]
                     for row in report["rebalance_phase_stress"])
    phase_count = len(report["rebalance_phase_stress"])
    lines += ["", f"候選只在 {phase_wins}/{phase_count} 種排程的 CAGR 勝過正式策略；換股日期敏感，尚未具備穩健替換證據。"]
    if "exploratory_staggered" in report:
        lines += ["", "後續探索：四批分散換股，降低單一排程依賴。這是在看過末段後新增的研究，",
                  "未納入原先選型；下列數字屬探索結果，不是新的未見資料驗證。", "",
                  "| 分批策略 | 末段 CAGR | 末段 Sharpe | 末段最大回撤 | 年換手 |",
                  "|---|---:|---:|---:|---:|"]
        for name, row in report["exploratory_staggered"]["strategies"].items():
            m = row["final_split"]
            lines.append(f"| {name} | {m['cagr']:.2%} | {m['sharpe']:.2f} | {m['mdd']:.2%} | {m['annual_turnover']:.1f} |")
    lines += ["", "限制：", "", *[f"- {s}" for s in report["limitations"]], "",
              "固定模型：LightGBM 排名迴歸、LambdaRank、XGBoost、Ridge；集成採等權百分位排名。",
              "所有模型僅使用標籤結束日早於當折預測日的樣本，固定每 126 日重訓；不依末段選模型或調參。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/strategy_models_2026_10_03")
    args = parser.parse_args()
    result = run_research(output_dir=args.output)
    print("Development-selected candidate:", result["selected_on_development"])
