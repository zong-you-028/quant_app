"""Fixed-50 versus current-market-cap-150 retrospective research only.

No app config, strategy, frozen paper manifest, download or formal DB changes.
Required inputs: independent public market DB, SOX CSV and ranking JSON.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager, ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.robustness_audit import (MarketSnapshot, database_path, isolated_public_seed,
                                   load_public_snapshot, replay)
from core.overfitting_audit import shared_block_bootstrap

# Freeze the original ordered universe; never read a subsequently changed
# config.UNIVERSE and accidentally call it the original 50-stock baseline.
FIXED_50 = (
    "2330", "2317", "2454", "2308", "2303", "2412", "2882", "2881", "1301", "2603",
    "2891", "3711", "2002", "2886", "2884", "1303", "2327", "2357", "3008", "2382",
    "2395", "5871", "2880", "2892", "2885", "1216", "2207", "2379", "3045", "2912",
    "1101", "2887", "4938", "5880", "2883", "2890", "2345", "3037", "2301", "3034",
    "2105", "9910", "2474", "1402", "2409", "2354", "2360", "6505", "3443", "3017",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ranking(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    symbols = spec.get("symbols")
    if symbols is None:
        symbols = [row.get("symbol") for row in spec.get("top150", [])]
    if not isinstance(symbols, list) or any(not isinstance(s, str) or not s for s in symbols):
        raise ValueError("ranking needs explicit string symbols")
    if len(symbols) != 150 or len(set(symbols)) != 150:
        raise ValueError("exactly 150 unique ranked symbols required; do not silently shrink")
    if not spec.get("asof"):
        raise ValueError("ranking asof date required")
    asof = pd.Timestamp(spec["asof"])
    if pd.isna(asof) or asof.tzinfo is not None or asof != asof.normalize():
        raise ValueError("ranking asof must be a timezone-naive daily date")
    return spec, list(symbols), asof


@contextmanager
def provided_public_snapshot(database, sox, symbols):
    """Read-only source backup, then existing strict public-seed isolation.

    SQLite backup captures a committed WAL correctly. Only named source mode=ro
    and this temporary destination are allowed during preparation. The replay
    then runs under the existing public-seed DB/network guards.
    """
    database, sox = Path(database).resolve(), Path(sox).resolve()
    formal = (ROOT / "data" / "market.db").resolve()
    if database == formal:
        raise ValueError("formal app database is forbidden; use independent research cache")
    if not database.is_file() or not sox.is_file():
        raise FileNotFoundError("independent research DB and SOX files required")
    connect = sqlite3.connect
    counters = {"formal_database_attempts": 0, "outside_database_attempts": 0,
                "source_readonly_connections": 0, "external_download_attempts": 0}
    with tempfile.TemporaryDirectory(prefix="quant_universe_seed_") as temporary:
        temporary = Path(temporary).resolve()

        def guarded(argument, *args, **kwargs):
            path = database_path(argument)
            if path == formal:
                counters["formal_database_attempts"] += 1
                raise AssertionError("formal app database forbidden")
            if path == database:
                if not kwargs.get("uri") or "mode=ro" not in os.fspath(argument):
                    raise AssertionError("research source is read-only")
                counters["source_readonly_connections"] += 1
            elif path is not None and not path.is_relative_to(temporary):
                counters["outside_database_attempts"] += 1
                raise AssertionError("unexpected database outside temporary seed")
            return connect(argument, *args, **kwargs)

        def no_network(*args, **kwargs):
            counters["external_download_attempts"] += 1
            raise AssertionError("network forbidden during universe comparison")

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"APP_DATA_DIR": str(temporary),
                                                        "JOURNAL_DATABASE_URL": ""}))
            stack.enter_context(patch.object(sqlite3, "connect", guarded))
            stack.enter_context(patch.object(sqlite3.dbapi2, "connect", guarded))
            import requests
            stack.enter_context(patch.object(requests.sessions.Session, "request", no_network))
            public_db = temporary / "market.db"
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
                tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables <= {"ohlcv", "chip_weekly", "stock_info", "meta", "sqlite_sequence"}:
                    raise ValueError("research database includes private/non-market tables")
                if not {"ohlcv", "chip_weekly"} <= tables:
                    raise ValueError("research database missing required market tables")
                with closing(sqlite3.connect(public_db)) as target:
                    source.backup(target)
                    target.execute("DROP TABLE IF EXISTS meta")
                    target.commit()
            copy_sha256 = digest(public_db)
            with public_db.open("rb") as source, gzip.open(temporary / "market.db.gz", "wb") as target:
                shutil.copyfileobj(source, target)
            shutil.copyfile(sox, temporary / "sox.csv")
        # Preparation's guard is restored before the nested isolation installs
        # its own guard; its original connector must allow its own temp root.
        with isolated_public_seed(temporary) as (directory, inner_counters):
            import config
            with patch.object(config, "UNIVERSE", list(symbols)):
                snapshot = load_public_snapshot(directory)
            yield snapshot, {"source": counters, "model": inner_counters,
                             "public_backup_sha256": copy_sha256}
        for group in (counters, inner_counters):
            if any(group.get(k, 0) for k in ("formal_database_attempts", "outside_database_attempts", "external_download_attempts")):
                raise AssertionError("isolation detected attempted forbidden access")


def for_symbols(snapshot, symbols, asof):
    clipped = snapshot.prefix(asof)
    return MarketSnapshot(clipped.prices, clipped.chips, clipped.sox,
                          list(symbols), clipped.benchmark)


def coverage(snapshot, asof):
    """New listings stay in the declared pool; do not invent their history."""
    from core.strategy_models import build_rank_scores
    import config
    entries, missing_prices, short_prices, missing_chips, short_chips = {}, [], [], [], []
    stale_prices, stale_chips, closes = [], [], {}
    for symbol in snapshot.symbols:
        prices = snapshot.prices.get(symbol, pd.DataFrame())
        chips = snapshot.chips.get(symbol, pd.DataFrame())
        good = prices.loc[prices["close"].gt(0) & prices["close"].notna()] if not prices.empty else prices
        if good.empty:
            missing_prices.append(symbol)
        elif len(good) < 252:
            short_prices.append(symbol)
        if not good.empty and good.index[-1].normalize() < asof:
            stale_prices.append(symbol)
        if chips.empty:
            missing_chips.append(symbol)
        elif len(chips) <= 60:
            short_chips.append(symbol)
        if not chips.empty and chips.index[-1].normalize() < asof - pd.Timedelta(days=7):
            stale_chips.append(symbol)
        closes[symbol] = good["close"] if not good.empty else pd.Series(dtype=float)
        entries[symbol] = {
            "price_observations": len(good),
            "price_start": str(good.index[0].date()) if not good.empty else None,
            "price_asof": str(good.index[-1].date()) if not good.empty else None,
            "chip_observations": len(chips),
            "chip_start": str(chips.index[0].date()) if not chips.empty else None,
            "chip_asof": str(chips.index[-1].date()) if not chips.empty else None,
            "short_price_history": len(good) < 252,
            "below_app_80_price_reader_threshold": len(good) < config.ROTATION_MIN_OBS,
        }
    close_panel = pd.DataFrame(closes).reindex(snapshot.calendar)
    scores = build_rank_scores(close_panel, "buffered_momentum")
    eligible = scores.notna().sum(axis=1)
    latest = list(scores.iloc[-1].dropna().index) if len(scores) else []
    # Short true histories are legitimate warmup exclusions. Missing/stale
    # sources are not: they must be resolved instead of silently reducing 150.
    blocked = bool(missing_prices or stale_prices or missing_chips or stale_chips)
    return {"declared_symbols": len(snapshot.symbols), "prices_present": len(snapshot.symbols) - len(missing_prices),
            "chips_present": len(snapshot.symbols) - len(missing_chips),
            "latest_rank_qualified_count": len(latest), "latest_rank_qualified_symbols": latest,
            "missing_prices": missing_prices, "short_price_history_below_252": short_prices,
            "stale_prices": stale_prices, "missing_chips": missing_chips,
            "chip_reader_warmup_below_61": short_chips, "stale_chips": stale_chips,
            "complete_source_coverage": not blocked, "symbols": entries,
            "eligibility": eligible}


def scenarios():
    return [{"name": f"cost_{cost:.3f}_lag_{lag}", "cost": cost, "lag": lag}
            for cost in (.003, .005, .008) for lag in (2, 3, 4)]


def compare_models(snapshot50, snapshot150, start, verbose=True):
    from core.strategy_research import metrics
    benchmark = snapshot50.prices[snapshot50.benchmark]["close"].reindex(snapshot50.calendar).pct_change(fill_method=None)
    rows, returns, turns, targets = [], {}, {}, {}
    for scenario in scenarios():
        row = dict(scenario)
        for name, snapshot in (("fixed_50", snapshot50), ("market_cap_150", snapshot150)):
            result = replay(snapshot, cost=scenario["cost"], lag=scenario["lag"])
            model = result["model"]
            net = model["net_returns"].loc[start:]
            if not net.index.equals(benchmark.loc[start:].index) or net.isna().any():
                raise AssertionError("both universes must use complete identical comparison dates")
            if (model["target_weights"].gt(0).sum(axis=1) > 8).any():
                raise AssertionError("universe expansion changed eight-stock limit")
            row[name] = metrics(net, result["turnover"], benchmark)
            row[name]["deferred_trade_days_full_history"] = model["deferred_trade_days"]
            returns[f"{scenario['name']}__{name}"] = net
            turns[f"{scenario['name']}__{name}"] = result["turnover"].loc[start:]
            if scenario["name"] == "cost_0.003_lag_2":
                targets[name] = model["target_weights"]
        row["paired_cagr_delta_150_minus_50"] = row["market_cap_150"]["cagr"] - row["fixed_50"]["cagr"]
        row["paired_sharpe_delta_150_minus_50"] = row["market_cap_150"]["sharpe"] - row["fixed_50"]["sharpe"]
        rows.append(row)
        if verbose:
            print(f"{scenario['name']}: fixed50 {row['fixed_50']['cagr']:.2%}, declared150 {row['market_cap_150']['cagr']:.2%}", flush=True)
    returns["006208"] = benchmark.loc[start:]
    return rows, pd.DataFrame(returns), pd.DataFrame(turns), targets


def year_and_confidence(paths, turns):
    from core.strategy_research import metrics
    short = {name: paths[f"cost_0.003_lag_2__{name}"] for name in ("fixed_50", "market_cap_150")}
    benchmark = paths["006208"]
    yearly = []
    active = short["market_cap_150"] - short["fixed_50"]
    for year in sorted(set(paths.index.year)):
        dates = paths.index[paths.index.year == year]
        row = {"year": int(year), "observations": len(dates), "start": str(dates[0].date()),
               "end": str(dates[-1].date()), "annualized_mean_daily_difference_150_minus_50": float(active.loc[dates].mean() * 252)}
        for name, values in short.items():
            row[name] = metrics(values.loc[dates], turns[f"cost_0.003_lag_2__{name}"], benchmark)
        row["006208"] = metrics(benchmark.loc[dates], pd.Series(0., index=benchmark.index), benchmark)
        row["006208"].pop("annual_turnover")
        yearly.append(row)
    paired, confidence = pd.DataFrame(short), {}
    for block in (20, 40, 60):
        result = shared_block_bootstrap(paired, baseline="fixed_50", block=block, draws=3000, seed=314159)
        candidate = result["candidates"]["market_cap_150"]
        confidence[str(block)] = {"rows": result["rows"], "draws": result["draws"], "seed": result["seed"],
                                   "annualized_mean_daily_difference": candidate["annualized_mean_active"],
                                   "pointwise_95_lower": candidate["pointwise_95_lower"],
                                   "pointwise_95_upper": candidate["pointwise_95_upper"],
                                   "single_candidate_one_sided_p": candidate["max_mean_adjusted_p"]}
    return {"yearly": yearly, "paired_pointwise_bootstrap": confidence,
            "bootstrap_status": "retrospective_not_adjusted_for_universe_selection",
            "bootstrap_difference_units": "252 * arithmetic paired daily return mean; not CAGR difference"}


def make_summary(report):
    lines = ["# 固定50檔與可取得收盤價普通股的估算市值150檔：歷史比較", "",
             "**目前選池之後的歷史診斷含選股及存活者偏誤，不是新樣本外，也不證明最佳化成功。**", "",
             "模型保持低換手多視窗、8個等權名額、前16名續留、20交易日換股；未變更app設定或既有紙上驗證manifest。", "",
             f"名單市值日期：{report['ranking_asof']}；資料上限：{report['price_asof']}。",
             "兩池使用同一006208日曆、費半来源、外資急賣閘門、成本與成交延遲，全部情境原樣保留。", "",
             "## 資料覆蓋", "", "| 股票池 | 宣告檔數 | 有行情 | 有籌碼 | 最新排名資格 | 完整來源 |",
             "|---|---:|---:|---:|---:|---|"]
    metadata = report.get("ranking_source_metadata", {})
    scope = metadata.get("ranking_scope", "supplied candidate pool; not certified full-market ranking")
    note = f"排名範圍：{scope}。這是依普通股收盤與股數來源估算的候選池，不稱官方完整全市場市值榜。"
    if metadata.get("provisional") or metadata.get("source_incomplete"):
        excluded = metadata.get("excluded_unpriced", [])
        note += f" 名單仍暫定；{len(excluded)}檔普通股缺同日有效價格被明列排除，交易狀態／完整涵蓋尚未確認。"
    lines.insert(9, note)
    for name, c in report["coverage"].items():
        lines.append(f"| {name} | {c['declared_symbols']} | {c['prices_present']} | {c['chips_present']} | {c['latest_rank_qualified_count']} | {c['complete_source_coverage']} |")
    lines += ["", "短上市歷史保留於宣告池，依252筆連續行情與動能錨點暖機，不補假價；coverage明示短歷史與實際可排名檔數。完全缺行情／外資或来源過期時拒絕比較，不縮池冒稱完整150檔。"]
    floor = report.get("acquisition_provenance", {}).get("listing_floor_filter")
    if floor:
        trimmed = floor.get("trimmed_this_run", {})
        lines += ["", "新增行情採『日期不早於官方current-market ISIN上市／櫃日期』的保守下限，排除興櫃及其他更早資料。",
                  f"本次修剪{len(trimmed)}檔的早期價格；逐檔日期及移除筆數保存於results.json的acquisition_provenance。",
                  "此規則也可能移除轉板前合法上市櫃行情，不能稱為完整歷史PIT資格；原50檔公開seed保持原價格。"]
    if report["status"] != "retrospective_completed":
        lines += ["", "**來源覆蓋未完成，未輸出50對150績效勝負。**", "",
                  "补齊results.json所列來源後，以相同指令重播。"]
        return "\n".join(lines) + "\n"
    lines += ["", f"共同區段：{report['evaluation']['start']}～{report['evaluation']['end']}，{report['evaluation']['observations']}筆。",
              "沿用app歷史暖機與持倉，截取同期成本後報酬，不在評估起點重設持倉。", "",
              "## 固定成本與成交壓力", "",
              "| 成本／lag | 50 CAGR | 150 CAGR | 50 Sharpe | 150 Sharpe | 50回撤 | 150回撤 | 50年換手 | 150年換手 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["scenarios"]:
        a, b = row["fixed_50"], row["market_cap_150"]
        lines.append(f"| {row['cost']:.1%}／t+{row['lag']} | {a['cagr']:.2%} | {b['cagr']:.2%} | {a['sharpe']:.2f} | {b['sharpe']:.2f} | {a['mdd']:.2%} | {b['mdd']:.2%} | {a['annual_turnover']:.2f} | {b['annual_turnover']:.2f} |")
    lines += ["", "## 逐年（成本0.3%、t+2）", "", "首末年可能不足一年；年化平均日差不是CAGR差。", "",
              "| 年／日數 | 50 CAGR／Sharpe | 150 CAGR／Sharpe | 006208 CAGR／Sharpe | 年化平均日差150−50 |",
              "|---|---|---|---|---:|"]
    for row in report["statistics"]["yearly"]:
        values = [row[n] for n in ("fixed_50", "market_cap_150", "006208")]
        lines.append(f"| {row['year']}／{row['observations']} | " + " | ".join(f"{v['cagr']:.2%}／{v['sharpe']:.2f}" for v in values) +
                     f" | {row['annualized_mean_daily_difference_150_minus_50']:.2%} |")
    lines += ["", "## 配對不確定性", "",
              "20／40／60日循環區塊，3000次、seed314159；單項回溯區間未校正選池／選型，不取代先前11候選PBO。", "",
              "| 區塊日數 | 年化平均日差150−50 | 單項95%區間 | 單候選單尾p |", "|---|---:|---|---:|"]
    for block, row in report["statistics"]["paired_pointwise_bootstrap"].items():
        lines.append(f"| {block} | {row['annualized_mean_daily_difference']:.2%} | {row['pointwise_95_lower']:.2%}～{row['pointwise_95_upper']:.2%} | {row['single_candidate_one_sided_p']:.3f} |")
    lines += ["", "## 限制", "", *[f"- {note}" for note in report["limitations"]], "",
              "重播：`python research/universe_150_2026_10_03/compare_universes.py --db <獨立DB> --sox <SOX.csv> --universe <排名JSON>`。"]
    return "\n".join(lines) + "\n"


def run_comparison(database, sox, ranking, output=None, verbose=True):
    if Path(database).resolve() == (ROOT / "data" / "market.db").resolve():
        raise ValueError("formal app database forbidden before hashing or opening")
    output = Path(output or Path(__file__).with_name("comparison"))
    spec, symbols150, asof = read_ranking(ranking)
    sources = {"research_database": {"path": str(Path(database).resolve()), "sha256": digest(database)},
               "sox": {"path": str(Path(sox).resolve()), "sha256": digest(sox)},
               "ranking": {"path": str(Path(ranking).resolve()), "sha256": digest(ranking)}}
    acquisition_path = Path(database).resolve().with_name("acquisition.json")
    acquisition = {}
    if acquisition_path.is_file():
        sources["acquisition_manifest"] = {"path": str(acquisition_path), "sha256": digest(acquisition_path)}
        manifest = json.loads(acquisition_path.read_text(encoding="utf-8"))
        acquisition = {k: manifest[k] for k in ("asof", "ranking_sha256", "base_seed_sha256",
                       "price_convention", "observed_at_utc", "listing_floor_filter") if k in manifest}
    code = {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), ROOT / "config.py", ROOT / "core/rotation.py", ROOT / "core/strategy_models.py",
        ROOT / "core/robustness_audit.py", ROOT / "core/execution.py", ROOT / "core/chip_data.py",
        ROOT / "core/market_regime.py", ROOT / "core/overfitting_audit.py")}
    with provided_public_snapshot(database, sox, symbols150) as (source, isolation):
        import config
        if config.BENCHMARK_SYMBOL in symbols150:
            raise ValueError("market-cap stock pool must exclude benchmark ETF")
        if config.ROTATION_TOP_K != 8 or config.ROTATION_REBAL_DAYS != 20:
            raise ValueError("requires unchanged eight-slot, 20-day model")
        if source.calendar[-1].normalize() < asof:
            raise ValueError("benchmark cache older than ranking asof")
        snapshots = {"fixed_50": for_symbols(source, FIXED_50, asof),
                     "market_cap_150": for_symbols(source, symbols150, asof)}
        coverages = {name: coverage(s, asof) for name, s in snapshots.items()}
        eligibility = pd.DataFrame({name: c.pop("eligibility") for name, c in coverages.items()})
        report = {"status": "incomplete_sources_no_performance_claim", "research_only": True,
                  "strategy_parameters_changed": False, "app_config_changed": False,
                  "ranking_asof": str(asof.date()), "price_asof": str(snapshots["fixed_50"].calendar[-1].date()),
                  "ranking_source_metadata": {k: v for k, v in spec.items() if k not in ("symbols", "top150", "all_ranked")},
                  "universes": {"fixed_50": list(FIXED_50), "market_cap_150": symbols150},
                  "coverage": coverages, "sources": sources, "code_sha256": code,
                  "acquisition_provenance": acquisition,
                  "model_parameters": {"model": "buffered_momentum", "top_k": 8, "retention_rank_limit": 16,
                                       "rebalance_days": 20, "momentum_windows": [60, 120, 252], "skip_days": 10},
                  "isolation": isolation, "limitations": [
                      "Current market-cap membership was selected after observing historical markets; it retains selection/survivorship bias and is not historical PIT membership.",
                      "All comparison dates are retrospective; neither universe nor favorable stress setting is automatically deployed or selected.",
                      "Ranking/market-cap methodology is supplied in source JSON, not independently certified historical membership.",
                      "The supplied candidate ranking estimates priced TWSE/TPEx ordinary-share capitalization; issued-common and listed-share capitalizations can differ, and unpriced official-classified stocks are explicitly excluded.",
                      "Prices retain the same unverified +/-11% corporate-action heuristic; neither dataset is certified total-return data.",
                      "Added-stock prices use a conservative current-market ISIN listing-date floor that can exclude legitimate older OTC transfers as well as emerging history; this is not complete PIT eligibility.",
                      "Historical chip dates are not publication timestamps; absent rolling Z observations retain existing gate warmup behavior.",
                      "The benchmark calendar is the existing app calendar, not an independently verified exchange-session calendar.",
                      "Flat costs omit spreads, capacity, locked limits, asymmetric taxes and corporate-action cash/quantity accounting.",
                      "006208 is a close-to-close reference without brokerage entry/exit costs, not a fill-verified paper account.",
                      "The frozen forward paper model and original 50-stock manifest are unchanged by this research."]}
        if all(c["complete_source_coverage"] for c in coverages.values()):
            calendar = snapshots["fixed_50"].calendar
            if len(calendar) < 878:
                raise ValueError("need fixed 756-bar boundary plus at least 120 common evaluation bars")
            start = calendar[756]
            rows, paths, turns, targets = compare_models(snapshots["fixed_50"], snapshots["market_cap_150"], start, verbose)
            report.update({"status": "retrospective_completed", "scenarios": rows,
                           "evaluation": {"start": str(start.date()), "end": str(calendar[-1].date()),
                                          "observations": len(paths), "warmup_common_boundary": 756,
                                          "same_006208_calendar": True, "historical_app_positions_carried": True},
                           "statistics": year_and_confidence(paths, turns)})
            output.mkdir(parents=True, exist_ok=True)
            paths.to_csv(output / "daily_returns.csv", index_label="date")
            turns.to_csv(output / "daily_turnover.csv", index_label="date")
            for name, target in targets.items():
                target.to_csv(output / f"targets_{name}.csv", index_label="date")
    for name, item in sources.items():
        if digest(item["path"]) != item["sha256"]:
            raise RuntimeError("source changed during comparison: " + name)
    for relative, fingerprint in code.items():
        if digest(ROOT / relative) != fingerprint:
            raise RuntimeError("code changed during comparison: " + relative)
    output.mkdir(parents=True, exist_ok=True)
    eligibility.to_csv(output / "daily_rank_eligibility.csv", index_label="date")
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "summary.md").write_text(make_summary(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--sox", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--output", default=str(Path(__file__).with_name("comparison")))
    args = parser.parse_args()
    result = run_comparison(args.db, args.sox, args.universe, args.output)
    print(json.dumps({"status": result["status"], "coverage": {name: {k: v for k, v in c.items() if k in
           ("declared_symbols", "prices_present", "chips_present", "latest_rank_qualified_count", "complete_source_coverage")}
           for name, c in result["coverage"].items()}}, ensure_ascii=False))
    if result["status"] != "retrospective_completed":
        sys.exit(2)
