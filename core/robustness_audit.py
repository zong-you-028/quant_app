"""Retrospective causal and fixed-stress audit of the actual app rotation.

Uses only a TemporaryDirectory copy of the public market/SOX seed. Network and
SQLite access outside that directory are forbidden. No journal import, order,
production change, optimization or genuinely unseen historical sample is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, closing
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from urllib.parse import unquote, urlsplit
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NUMERIC_TOLERANCE = 1e-12


@dataclass
class MarketSnapshot:
    prices: dict[str, pd.DataFrame]
    chips: dict[str, pd.DataFrame]
    sox: pd.Series
    symbols: list[str]
    benchmark: str

    @property
    def calendar(self):
        return self.prices[self.benchmark].index

    def prefix(self, cutoff):
        cutoff = pd.Timestamp(cutoff)
        return MarketSnapshot(
            {s: frame.loc[:cutoff].copy() for s, frame in self.prices.items()},
            {s: frame.loc[:cutoff].copy() for s, frame in self.chips.items()},
            self.sox.loc[:cutoff].copy(), list(self.symbols), self.benchmark)

    def perturb_future(self, cutoff, source="all"):
        if source not in ("prices", "chips", "sox", "all"):
            raise ValueError("unknown perturbation source")
        changed = MarketSnapshot(
            {s: f.copy() for s, f in self.prices.items()},
            {s: f.copy() for s, f in self.chips.items()},
            self.sox.copy(), list(self.symbols), self.benchmark)
        cutoff = pd.Timestamp(cutoff)
        if source in ("prices", "all"):
            for i, (symbol, frame) in enumerate(changed.prices.items()):
                mask = frame.index > cutoff
                factor = .25 + (i % 17) * .23
                frame.loc[mask, ["open", "high", "low", "close"]] *= factor
                frame.loc[mask, "volume"] *= 7 + i % 5
        if source in ("chips", "all"):
            for frame in changed.chips.values():
                frame.loc[frame.index > cutoff, "big_shares"] *= .07
        if source in ("sox", "all"):
            mask = changed.sox.index > cutoff
            changed.sox.loc[mask] *= np.where(np.arange(mask.sum()) % 2, 4., .2)
        return changed


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def database_path(database):
    value = os.fspath(database)
    if value == ":memory:":
        return None
    if value.startswith("file:"):
        value = unquote(urlsplit(value).path)
        if os.name == "nt" and len(value) > 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    return Path(value).resolve()


@contextmanager
def isolated_public_seed(seed_dir=None):
    """Install strict runtime guards before any pipeline read is possible."""
    seed_dir = Path(seed_dir or ROOT / "data_seed").resolve()
    formal = (ROOT / "data" / "market.db").resolve()
    counters = {"formal_database_attempts": 0, "outside_database_attempts": 0,
                "external_download_attempts": 0, "isolated_connections": 0}
    with tempfile.TemporaryDirectory(prefix="quant_robustness_") as directory:
        directory = Path(directory).resolve()
        with gzip.open(seed_dir / "market.db.gz", "rb") as source, (directory / "market.db").open("wb") as target:
            shutil.copyfileobj(source, target)
        shutil.copyfile(seed_dir / "sox.csv", directory / "sox.csv")
        connect = sqlite3.connect

        def guarded(database, *args, **kwargs):
            path = database_path(database)
            if path == formal:
                counters["formal_database_attempts"] += 1
                raise AssertionError("formal database access forbidden")
            if path is not None and not path.is_relative_to(directory):
                counters["outside_database_attempts"] += 1
                raise AssertionError("database outside isolated seed forbidden")
            counters["isolated_connections"] += 1
            return connect(database, *args, **kwargs)

        def no_network(*args, **kwargs):
            counters["external_download_attempts"] += 1
            raise AssertionError("market download forbidden in offline audit")

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"APP_DATA_DIR": str(directory),
                                                        "JOURNAL_DATABASE_URL": ""}))
            stack.enter_context(patch.object(sqlite3, "connect", guarded))
            stack.enter_context(patch.object(sqlite3.dbapi2, "connect", guarded))
            import config
            import requests
            stack.enter_context(patch.object(config, "DATA_DIR", str(directory)))
            stack.enter_context(patch.object(config, "DB_PATH", str(directory / "market.db")))
            stack.enter_context(patch.object(requests.sessions.Session, "request", no_network))
            yield directory, counters


def load_public_snapshot(directory):
    import config
    path = Path(directory) / "market.db"
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables <= {"stock_info", "ohlcv", "chip_weekly", "sqlite_sequence"}:
            raise ValueError("seed includes tables outside public market data")
        bars = pd.read_sql_query("SELECT * FROM ohlcv ORDER BY date,symbol", connection)
        chips = pd.read_sql_query("SELECT * FROM chip_weekly ORDER BY date,symbol", connection)
    bars["date"] = pd.to_datetime(bars["date"])
    chips["date"] = pd.to_datetime(chips["date"])
    prices = {str(symbol): frame.drop(columns=["symbol"]).set_index("date").sort_index()
              for symbol, frame in bars.groupby("symbol")}
    holding = {str(symbol): frame.drop(columns=["symbol"]).set_index("date").sort_index()
               for symbol, frame in chips.groupby("symbol")}
    sox = pd.read_csv(Path(directory) / "sox.csv", index_col=0).iloc[:, 0]
    sox.index = pd.to_datetime(sox.index)
    if config.BENCHMARK_SYMBOL not in prices:
        raise ValueError("benchmark required for common trading calendar")
    return MarketSnapshot(prices, holding, sox.sort_index().astype(float),
                          list(config.UNIVERSE), config.BENCHMARK_SYMBOL)


def replay(snapshot, *, mode="buffered_momentum", cost=.003, lag=2,
           chip_delay=0, sox_delay=0):
    """Actual app ranking/selection/execution, with in-memory source readers.

    External delays mean additional Taiwan-calendar bars of source-gate
    availability, on top of the common execution lag. No new gate is fitted.
    """
    if lag < 1 or chip_delay < 0 or sox_delay < 0 or not 0 <= cost < 1:
        raise ValueError("invalid fixed stress scenario")
    import config
    from core import rotation, data_pipeline, chip_data, market_regime
    from core.execution import backtest_open_execution
    from core.strategy_models import data_quality

    capture = {}
    original_chip = chip_data.fastsell_z_panel
    original_sox = market_regime.sox_regime_series
    original_panel = rotation._load_panel

    def prices(symbol):
        return snapshot.prices.get(symbol, pd.DataFrame()).copy()

    def chips(symbol):
        return snapshot.chips.get(symbol, pd.DataFrame()).copy()

    def delayed_chip(symbols, index, *args, **kwargs):
        panel = original_chip(symbols, index, *args, **kwargs)
        return panel.shift(chip_delay) if panel is not None else None

    def delayed_sox(index, *args, **kwargs):
        return original_sox(index, *args, **kwargs).shift(sox_delay).fillna(1.)

    def execute(*args, **kwargs):
        returns, turns = backtest_open_execution(*args, **kwargs)
        capture["turnover"] = turns
        capture["rebalance"] = kwargs["rebalance"].copy()
        return returns, turns

    def quality(symbols=None, prices=None, **kwargs):
        dates = {s: f.index.max() for s, f in snapshot.chips.items() if not f.empty}
        return data_quality(symbols=symbols, prices=prices, chip_dates=dates, **kwargs)

    def common_baseline_panel(symbols, mom_days, min_obs=None, skip_days=None):
        legacy_returns, _, _, names = original_panel(symbols, mom_days, min_obs, skip_days)
        closes = pd.DataFrame({s: snapshot.prices[s]["close"] for s in legacy_returns.columns})
        closes = closes.reindex(snapshot.calendar)
        skip = config.ROTATION_SKIP_DAYS if skip_days is None else skip_days
        returns = closes.pct_change(fill_method=None)
        scores = rotation.build_rank_scores(closes, "production", mom_days=mom_days, skip_days=skip)
        gates = closes.pct_change(mom_days, fill_method=None)
        return returns, scores, gates, names

    with ExitStack() as stack:
        for obj, name, value in (
            (rotation, "ensure_data", lambda symbol: None),
            (rotation, "load_ohlcv", prices),
            (rotation, "get_stock_name", lambda symbol: symbol),
            (rotation, "backtest_open_execution", execute),
            (rotation, "data_quality", quality),
            (data_pipeline, "load_chip_weekly", chips),
            (market_regime, "load_sox", lambda: snapshot.sox.copy()),
            (market_regime, "sox_regime_series", delayed_sox),
            (chip_data, "fastsell_z_panel", delayed_chip),
            (config, "ROTATION_EXECUTION_LAG", lag),
        ):
            stack.enter_context(patch.object(obj, name, value))
        if mode == "production_common_calendar":
            stack.enter_context(patch.object(rotation, "_load_panel", common_baseline_panel))
        model = rotation.run_rotation(symbols=snapshot.symbols,
                                      score_mode="production" if mode == "production_common_calendar" else mode,
                                      cost_per_turnover=cost)
    return {"model": model, **capture}


def compare_past(expected, actual, cutoff):
    """Raises on any change in past mathematical outputs or schedules."""
    cutoff = pd.Timestamp(cutoff)
    numerical = {}
    for key in ("target_weights", "net_returns", "full_equity"):
        left, right = expected["model"][key].loc[:cutoff], actual["model"][key].loc[:cutoff]
        if isinstance(left, pd.DataFrame):
            # Prefix source eligibility can omit a not-yet-qualified stock.
            # A missing all-zero target column is not a portfolio difference.
            columns = left.columns.union(right.columns, sort=False)
            left, right = left.reindex(columns=columns, fill_value=0.), right.reindex(columns=columns, fill_value=0.)
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        else:
            numerical[key] = float((left - right).abs().max())
            pd.testing.assert_series_equal(left, right, check_exact=False,
                                           rtol=NUMERIC_TOLERANCE, atol=NUMERIC_TOLERANCE)
    for key in ("rebalance", "turnover"):
        left, right = expected[key].loc[:cutoff], actual[key].loc[:cutoff]
        if key == "rebalance":
            pd.testing.assert_series_equal(left, right, check_exact=True)
        else:
            numerical[key] = float((left - right).abs().max())
            pd.testing.assert_series_equal(left, right, check_exact=False,
                                           rtol=NUMERIC_TOLERANCE, atol=NUMERIC_TOLERANCE)
    return numerical


def execution_at(replayed, snapshot, cutoff):
    """Current-state metadata reconstructed strictly through cutoff only."""
    from core.execution import backtest_open_execution
    target = replayed["model"]["target_weights"].loc[:cutoff]
    opens = pd.DataFrame({s: snapshot.prices[s]["open"] for s in target.columns}).reindex(target.index)
    closes = pd.DataFrame({s: snapshot.prices[s]["close"] for s in target.columns}).reindex(target.index)
    returns, _ = backtest_open_execution(
        target, opens, closes, lag=replayed["model"]["execution_lag"],
        cost=replayed["model"]["cost_per_turnover"], rebalance=replayed["rebalance"].loc[:cutoff])
    return {key: returns.attrs[key] for key in ("effective_target_weights", "last_execution_date", "execution_deferred")}


def causal_audit(snapshot, cutoffs=None, verbose=False):
    """Prefix replay plus independent/combined future source contamination."""
    calendar = snapshot.calendar
    if cutoffs is None:
        positions = sorted({300, 755, 1000, 1500, 2000, len(calendar) - 3})
        cutoffs = [calendar[i] for i in positions if 252 <= i < len(calendar) - 1]
    full = replay(snapshot)
    rows = []
    for cutoff in cutoffs:
        prefix = replay(snapshot.prefix(cutoff))
        numerical = compare_past(full, prefix, cutoff)
        expected_state = execution_at(full, snapshot, cutoff)
        prefix_state = {"effective_target_weights": prefix["model"]["model_current_weights"],
                        "last_execution_date": prefix["model"]["model_current_execution_date"],
                        "execution_deferred": prefix["model"]["model_execution_deferred"]}
        if expected_state != prefix_state:
            raise AssertionError("prefix execution-state metadata differs")
        rows.append({"cutoff": str(pd.Timestamp(cutoff).date()), "check": "prefix_and_execution_state", "passed": True,
                     "prefix_omitted_zero_target_columns": sorted(set(full["model"]["target_weights"]) - set(prefix["model"]["target_weights"]))})
        rows[-1]["numeric_max_abs_differences"] = numerical
        for source in ("prices", "chips", "sox", "all"):
            changed = snapshot.perturb_future(cutoff, source)
            future = replay(changed)
            numerical = compare_past(full, future, cutoff)
            if execution_at(future, changed, cutoff) != expected_state:
                raise AssertionError("future source changed past execution metadata")
            rows.append({"cutoff": str(pd.Timestamp(cutoff).date()), "check": "future_" + source, "passed": True,
                         "numeric_max_abs_differences": numerical})
        if verbose:
            print(f"causal prefix {pd.Timestamp(cutoff).date()}: five checks passed", flush=True)
    return rows


def fixed_scenarios():
    scenarios = [{"name": f"cost_{cost:.3f}_lag_{lag}", "cost": cost, "lag": lag,
                  "chip_delay": 0, "sox_delay": 0}
                 for cost in (.003, .005, .008) for lag in (2, 3, 4)]
    for source in ("chip", "sox"):
        for delay in (1, 2):
            scenarios.append({"name": f"{source}_delay_{delay}", "cost": .003, "lag": 2,
                              "chip_delay": delay if source == "chip" else 0,
                              "sox_delay": delay if source == "sox" else 0})
    scenarios.append({"name": "both_sources_delay_2", "cost": .003, "lag": 2,
                      "chip_delay": 2, "sox_delay": 2})
    return scenarios


def performance(returns, turnover, benchmark):
    from core.strategy_research import metrics
    return metrics(returns, turnover, benchmark)


def stress_audit(snapshot, verbose=False):
    calendar = snapshot.calendar
    # Matches the original 756-bar common comparison boundary, but keeps the
    # actual app's historical portfolio initialization (no artificial reset).
    start = calendar[min(756, len(calendar) - 2)]
    benchmark = snapshot.prices[snapshot.benchmark]["close"].pct_change(fill_method=None)
    rows, paths = [], {}
    for scenario in fixed_scenarios():
        args = {k: v for k, v in scenario.items() if k != "name"}
        paired = {}
        for mode in ("buffered_momentum", "production_common_calendar"):
            result = replay(snapshot, mode=mode, **args)
            returns = result["model"]["net_returns"].loc[start:]
            if not returns.index.equals(benchmark.loc[start:].index):
                raise ValueError("unequal comparison calendar")
            metric = performance(returns, result["turnover"], benchmark)
            metric["deferred_trade_days_full_history"] = result["model"]["deferred_trade_days"]
            paired[mode] = metric
            paths[f"{scenario['name']}__{mode}"] = returns
        rows.append({**scenario, **paired,
                     "paired_cagr_difference": paired["buffered_momentum"]["cagr"] - paired["production_common_calendar"]["cagr"],
                     "paired_sharpe_difference": paired["buffered_momentum"]["sharpe"] - paired["production_common_calendar"]["sharpe"]})
        if verbose:
            print(f"stress {scenario['name']}: active CAGR {paired['buffered_momentum']['cagr']:.2%}; common-calendar baseline {paired['production_common_calendar']['cagr']:.2%}", flush=True)
    paths["006208"] = benchmark.loc[start:]
    return {"start": str(start.date()), "end": str(calendar[-1].date()), "rows": rows}, pd.DataFrame(paths)


def yearly_concentration(paths):
    """Disclose all years and all leave-one-year-out diagnostic differences.

    Exclusion is a concentration diagnostic, never a new held-out selection.
    The primary metric is 252 * mean paired daily return difference, not CAGR.
    """
    active = paths["cost_0.003_lag_2__buffered_momentum"]
    baseline = paths["cost_0.003_lag_2__production_common_calendar"]
    benchmark = paths["006208"]
    if active.isna().any() or baseline.isna().any() or benchmark.isna().any():
        raise ValueError("complete paired dates required for year concentration")
    yearly, exclusions = [], []
    differences = active - baseline
    for year in sorted(set(active.index.year)):
        mask = active.index.year == year
        row = {"year": int(year), "observations": int(mask.sum()),
               "start": str(active.index[mask][0].date()), "end": str(active.index[mask][-1].date()),
               "active_minus_baseline_annualized_daily_mean": float(differences.loc[mask].mean() * 252),
               "active_minus_etf_annualized_daily_mean": float((active - benchmark).loc[mask].mean() * 252)}
        for name, values in (("buffered_momentum", active), ("production_common_calendar", baseline), ("006208", benchmark)):
            row[name] = performance(values.loc[mask], pd.Series(0., index=values.index), benchmark)
            # This report consumes returns, not saved per-year trades. A zero
            # placeholder passed to metrics must not be reported as turnover.
            row[name].pop("annual_turnover")
        yearly.append(row)
        remaining = differences.loc[~mask]
        exclusions.append({"excluded_year": int(year), "remaining_observations": len(remaining),
                           "annualized_mean_active_minus_baseline": float(remaining.mean() * 252),
                           "annualized_mean_active_minus_etf": float((active - benchmark).loc[~mask].mean() * 252)})
    best = max(yearly, key=lambda r: r["active_minus_baseline_annualized_daily_mean"])
    return {"status": "retrospective_year_concentration_diagnostic", "yearly": yearly,
            "leave_one_calendar_year_out": exclusions,
            "all_years_annualized_mean_active_minus_baseline": float(differences.mean() * 252),
            "highest_annualized_difference_year": best["year"],
            "excluding_highest_difference_year": next(r for r in exclusions if r["excluded_year"] == best["year"]),
            "difference_units": "252 times arithmetic paired daily mean; not CAGR difference",
            "no_exclusion_used_for_strategy_selection": True}


def paired_confidence(paths):
    """Fixed pointwise retrospective CIs, not the original family correction."""
    from core.overfitting_audit import shared_block_bootstrap
    active = paths["cost_0.003_lag_2__buffered_momentum"]
    baselines = {"production_common_calendar": paths["cost_0.003_lag_2__production_common_calendar"],
                 "006208": paths["006208"]}
    comparisons = {}
    for name, baseline in baselines.items():
        frame = pd.DataFrame({name: baseline, "buffered_momentum": active})
        comparisons[name] = {}
        for block in (20, 40, 60):
            result = shared_block_bootstrap(frame, baseline=name, block=block, draws=3000, seed=314159)
            candidate = result["candidates"]["buffered_momentum"]
            comparisons[name][str(block)] = {
                "rows": result["rows"], "block_days": block, "draws": result["draws"], "seed": result["seed"],
                "annualized_mean_active": candidate["annualized_mean_active"],
                "pointwise_95_lower": candidate["pointwise_95_lower"],
                "pointwise_95_upper": candidate["pointwise_95_upper"],
                "single_candidate_one_sided_p": candidate["max_mean_adjusted_p"]}
    return {"status": "pointwise_retrospective_not_selection_adjusted",
            "primary_baseline": "production_common_calendar", "secondary_reference": "006208",
            "comparisons": comparisons, "original_11_candidate_correction_recomputed": False,
            "difference_units": "252 times arithmetic paired daily mean; not CAGR difference",
            "limitation": "These pointwise intervals do not correct earlier selection, phase/universe choices or multiple diagnostics; they do not replace the saved 11-candidate audit."}


def write_summary(report, output):
    lines = ["# 固定模型：歷史因果性與壓力驗證", "",
             "**全部結果是已看過歷史的回溯診斷，不能排除過擬合，沒有依結果調參。**", "",
             f"公開行情快照截至 {report['price_asof']}；實際使用 `core.rotation.run_rotation` 的排名、保留名單、風險閘門與開盤成交引擎。",
             "資料來自 public seed 的暫存副本；不下載新資料、不存取正式 DB、不寫帳本。", "",
             "## 因果性", "",
             f"{len(report['causal_checks'])} 項前綴／未來污染檢查通過：過去目標、換股排程和成交狀態一致；成本後報酬、權益與換手採1e-12浮點容差，逐項差值保存。",
             "早期前綴未滿80筆的3711被app略過，全資料保留其當時全零目標欄；補零後目標逐筆一致，但加總維度造成約機器精度的數值差。未因此修改production。",
             "這證明所檢查輸入下的程式因果性，不代表消除資料發布時點、選型或存活者偏誤。", "",
             "## 固定壓力情境", "",
             f"比較區段 {report['stress']['start']}～{report['stress']['end']}；沿用 app 全歷史暖機及起始持倉，截取同期报酬，不重設歷史持倉。",
             "原策略基準保留60日／跳過10日公式，重新在006208共同日曆計算；稱 `production_common_calendar`，不宣稱與舊 app 的股票日期聯集流程逐筆一致。",
             "成本按每單位買賣總換手計；外部來源延遲以額外台股交易日的已計算閘門可得性模擬，仍另加成交 lag。", "",
             "| 情境 | 模型 CAGR | 原策略 CAGR | 模型 Sharpe | 模型回撤 | 模型年換手 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in report["stress"]["rows"]:
        active, baseline = row["buffered_momentum"], row["production_common_calendar"]
        lines.append(f"| {row['name']} | {active['cagr']:.2%} | {baseline['cagr']:.2%} | {active['sharpe']:.2f} | {active['mdd']:.2%} | {active['annual_turnover']:.2f} |")
    lines += ["", "所有情境固定列出；不將最佳成本、延遲或報酬當作策略新設定。", "",
              "## 單一年度集中度", "",
              "固定採成本0.3%、lag2。相對均值是日報酬差算術平均乘252，並非CAGR差；首末年可能不足完整一年。", "",
              "| 年 | 日數 | 模型 CAGR／Sharpe | 共日日曆原策略 CAGR／Sharpe | 006208 CAGR／Sharpe | 相對原策略年化平均日差 |",
              "|---|---:|---|---|---|---:|"]
    concentration = report["year_concentration"]
    for row in concentration["yearly"]:
        values = [row[n] for n in ("buffered_momentum", "production_common_calendar", "006208")]
        formatted = [f"{m['cagr']:.2%}／{m['sharpe']:.2f}" for m in values]
        lines.append(f"| {row['year']} | {row['observations']} | " + " | ".join(formatted) +
                     f" | {row['active_minus_baseline_annualized_daily_mean']:.2%} |")
    excluded = concentration["excluding_highest_difference_year"]
    lines += ["", f"全期相對原策略年化平均日差 {concentration['all_years_annualized_mean_active_minus_baseline']:.2%}；",
              f"排除年度差最高的 {excluded['excluded_year']} 年後為 {excluded['annualized_mean_active_minus_baseline']:.2%}。",
              "這是事後集中度診斷，不能稱為該年度未見樣本外測試；全部逐年排除結果都保存，不以排除結果挑策略。", "",
              "| 排除年 | 剩餘日數 | 相對原策略年化平均日差 | 相對006208年化平均日差 |",
              "|---|---:|---:|---:|"]
    for row in concentration["leave_one_calendar_year_out"]:
        lines.append(f"| {row['excluded_year']} | {row['remaining_observations']} | {row['annualized_mean_active_minus_baseline']:.2%} | {row['annualized_mean_active_minus_etf']:.2%} |")
    lines += ["", "## 同日期配對不確定性", "",
              "固定模型成本0.3%、lag2的配對循環區塊bootstrap：20／40／60日，各3000次，seed314159；主比較為共日日曆原策略，006208為次要參考。",
              "以下是已看過歷史的單項95%區間，未校正先前選型；不能與舊報告+5.16%的點估計及11候選校正p=0.389混用，也不取代該診斷。", "",
              "| 比較基準 | 區塊日數 | 年化平均日差 | 單項95%區間 | 單候選單尾p |",
              "|---|---:|---:|---|---:|"]
    for name, rows in report["paired_bootstrap"]["comparisons"].items():
        for block, row in rows.items():
            lines.append(f"| {name} | {block} | {row['annualized_mean_active']:.2%} | {row['pointwise_95_lower']:.2%}～{row['pointwise_95_upper']:.2%} | {row['single_candidate_one_sided_p']:.3f} |")
    lines += ["", "## 仍未解決", ""]
    lines += [f"- {note}" for note in report["limitations"]]
    lines += ["", "重播：`python -m core.robustness_audit`；測試：`python -m pytest -q test_robustness_audit.py`。", "",
              "來源檔案 SHA-256、逐項檢查與所有情境逐日報酬見同目錄 `results.json`、`causal_checks.csv`、`stress_daily_returns.csv`。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(output_dir=None, verbose=True):
    output = Path(output_dir or ROOT / "research" / "robustness_2026_10_03")
    sources = {str(path.relative_to(ROOT)): fingerprint(path) for path in (
        ROOT / "data_seed/market.db.gz", ROOT / "data_seed/sox.csv", ROOT / "config.py",
        ROOT / "core/rotation.py", ROOT / "core/strategy_models.py", ROOT / "core/execution.py",
        ROOT / "core/chip_data.py", ROOT / "core/market_regime.py", ROOT / "core/overfitting_audit.py", Path(__file__))}
    with isolated_public_seed() as (directory, counters):
        snapshot = load_public_snapshot(directory)
        checks = causal_audit(snapshot, verbose=verbose)
        stress, paths = stress_audit(snapshot, verbose=verbose)
        if any(counters[k] for k in ("formal_database_attempts", "outside_database_attempts", "external_download_attempts")):
            raise AssertionError("offline isolation guard was invoked")
        stock_calendar = pd.DatetimeIndex([])
        for symbol in snapshot.symbols:
            stock_calendar = stock_calendar.union(snapshot.prices[symbol].index)
        absent_from_benchmark = stock_calendar.difference(snapshot.calendar)
        report = {"status": "retrospective_already_seen_history", "strategy_parameters_changed": False,
                  "active_model": "buffered_momentum", "sources_sha256": sources,
                  "price_asof": str(snapshot.calendar[-1].date()), "sox_asof": str(snapshot.sox.index[-1].date()),
                  "universe": snapshot.symbols, "causal_checks": checks, "stress": stress,
                  "baseline": "production_common_calendar",
                  "numeric_comparison": {"absolute_tolerance": NUMERIC_TOLERANCE,
                                         "relative_tolerance": NUMERIC_TOLERANCE,
                                         "targets_and_rebalance_exact": True},
                  "calendar_coverage": {"benchmark_bars": len(snapshot.calendar),
                                        "stock_union_bars": len(stock_calendar),
                                        "stock_source_dates_absent_from_benchmark": len(absent_from_benchmark),
                                        "absent_date_sample": [str(d.date()) for d in absent_from_benchmark[:20]],
                                        "official_exchange_calendar_verified": False},
                  "year_concentration": yearly_concentration(paths),
                  "paired_bootstrap": paired_confidence(paths),
                  "isolation": dict(counters),
                  "limitations": [
                      "All dates were already available before this audit; no new chronological out-of-sample evidence.",
                      "Price perturbations start after pipeline adjustment; historical raw-price revisions/corporate-action reconstruction are not validated.",
                      "The existing +/-11% heuristic is not official total-return or corporate-action data; dividends, splits and missing-bar jumps remain unresolved.",
                      "The current 50-stock universe is not historical point-in-time membership and omits delisted holdings.",
                      "Date-only SOX/chip data do not certify publication timestamps; gate delays are stress assumptions, not verified release times.",
                      "The app chip reader reindexes before forward-filling, whereas older research unions source dates first; this audit tests actual app behavior and does not reconcile non-calendar releases.",
                      "The benchmark calendar omits dates present in the stock-source union; it is not independently certified as the exchange trading calendar.",
                      "Flat costs do not model asymmetric taxes, spread, liquidity capacity, locked limits, cash interest or brokerage fills.",
                      "The 006208 reference uses cached close-to-close returns without brokerage entry/exit costs, not a fill-verified ETF paper account.",
                      "Shorter/deferred missing-open cases are covered by synthetic execution tests, not guaranteed by complete historical bars.",
                      "Actual app initialization/calendar differs from the original research cash start; these metrics do not reproduce prior report values.",
                      "Passing causal checks cannot remove selection bias, earlier undisclosed experiments or statistical overfitting."]}
    for relative, digest in sources.items():
        if fingerprint(ROOT / relative) != digest:
            raise RuntimeError("audit source changed during run: " + relative)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(checks).to_csv(output / "causal_checks.csv", index=False)
    paths.to_csv(output / "stress_daily_returns.csv", index_label="date")
    write_summary(report, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "research/robustness_2026_10_03"))
    args = parser.parse_args()
    report = run_audit(args.output)
    print(f"Retrospective checks: {len(report['causal_checks'])}; fixed stress scenarios: {len(report['stress']['rows'])}; no rule changes.")
