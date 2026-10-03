"""Offline phase robustness diagnostics; never change production defaults.

All dates in this snapshot have already been researched.  Development-only
criteria are a reproducible screening rule, not an untouched validation test.
The final period is diagnostic only.  No phase is selected or optimized.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from core.strategy_research import (
    ResearchSettings, backtest_open_execution, features_and_rules,
    load_snapshot, metrics, portfolio_targets,
)

ORIGINAL_NAMES = (
    "production", "multi_momentum", "trend_quality", "buffered_momentum", "vol_control",
)
SCREEN_CRITERIA = {
    "development_sharpe_win_share_at_least": .75,
    "development_median_paired_cagr_delta_greater_than": 0.0,
    "development_median_paired_mdd_delta_at_least": 0.0,
    "phase_selection_allowed": False,
    "holdout_used_to_select_strategy": False,
}


def summarize_phases(rows, baseline_name="production"):
    """Compare each candidate against production at the same phase."""
    frame = pd.DataFrame(rows)
    summaries = {}
    for segment in ("development", "final_diagnostic", "full_common"):
        subset = frame[frame.segment == segment]
        baseline = subset[subset.candidate == baseline_name].set_index("phase")
        summaries[segment] = {}
        for name, part in subset.groupby("candidate", sort=False):
            part = part.set_index("phase").sort_index()
            paired = part[["cagr", "sharpe", "mdd"]] - baseline[["cagr", "sharpe", "mdd"]]
            summary = {
                "phase_count": len(part),
                "cagr_median": float(part.cagr.median()),
                "cagr_min": float(part.cagr.min()),
                "cagr_max": float(part.cagr.max()),
                "sharpe_median": float(part.sharpe.median()),
                "mdd_median": float(part.mdd.median()),
                "mdd_worst": float(part.mdd.min()),
                "annual_turnover_median": float(part.annual_turnover.median()),
                "cagr_win_share": float((paired.cagr > 0).mean()),
                "sharpe_win_share": float((paired.sharpe > 0).mean()),
                "mdd_not_worse_share": float((paired.mdd >= 0).mean()),
                "median_paired_cagr_delta": float(paired.cagr.median()),
                "median_paired_sharpe_delta": float(paired.sharpe.median()),
                "median_paired_mdd_delta": float(paired.mdd.median()),
                "worst_paired_cagr_delta": float(paired.cagr.min()),
            }
            summary["development_screen_pass"] = bool(
                segment == "development" and name != baseline_name
                and summary["sharpe_win_share"] >= .75
                and summary["median_paired_cagr_delta"] > 0
                and summary["median_paired_mdd_delta"] >= 0
            )
            summaries[segment][name] = summary
    return summaries


def exploratory_scores(close, original_scores, valid):
    """Three fixed economically motivated formulas, declared before their run.

    Risk adjustment compares persistent momentum per unit of past volatility.
    The 2K rank buffer reduces costly switches between nearly tied candidates.
    Quality plus buffering preserves smooth trends while reducing rank churn.
    All rank calculations and volatility estimates use historical data only.
    """
    daily = close.pct_change(fill_method=None)
    adjusted = []
    volatility = daily.rolling(60).std().replace(0, np.nan) * np.sqrt(252)
    for horizon in (60, 120, 252):
        momentum = close.shift(10) / close.shift(horizon) - 1
        adjusted.append((momentum / volatility).where(valid).replace(
            [np.inf, -np.inf], np.nan).rank(axis=1, pct=True))
    score = sum(adjusted) / 3
    return {
        "risk_adjusted_momentum": (score, "multi_momentum"),
        "risk_adjusted_buffered": (score, "buffered_momentum"),
        "quality_buffered": (original_scores["trend_quality"], "buffered_momentum"),
    }


def evaluate_candidates(candidates, snapshot, settings, rows, paths, verbose=True):
    index = snapshot["close"].index
    benchmark = snapshot["benchmark"].pct_change(fill_method=None)
    segments = {
        "development": index[settings.train_min:-settings.holdout_days],
        "final_diagnostic": index[-settings.holdout_days:],
        "full_common": index[settings.train_min:],
    }
    for name, (score, implementation_name) in candidates.items():
        for phase in range(settings.rebalance_days):
            target, schedule = portfolio_targets(
                score, snapshot, settings, implementation_name, phase)
            returns, turnover = backtest_open_execution(
                target, snapshot["open"], snapshot["close"],
                lag=settings.lag, cost=settings.cost, rebalance=schedule)
            paths[f"{name}__phase_{phase:02d}"] = returns.reindex(segments["full_common"])
            for segment, period in segments.items():
                rows.append({"candidate": name, "phase": phase, "segment": segment,
                             "deferred_days": len(returns.attrs.get("deferred_dates", [])),
                             **metrics(returns.reindex(period), turnover, benchmark)})
        if verbose:
            summary = summarize_phases(rows)
            dev = summary["development"][name]
            final = summary["final_diagnostic"][name]
            print(f"{name}: dev median CAGR {dev['cagr_median']:.2%}, "
                  f"Sharpe wins {dev['sharpe_win_share']:.0%}, "
                  f"pass={dev['development_screen_pass']}; "
                  f"final median CAGR {final['cagr_median']:.2%}, "
                  f"CAGR wins {final['cagr_win_share']:.0%}, "
                  f"Sharpe wins {final['sharpe_win_share']:.0%}", flush=True)


def write_report(report, output):
    descriptions = {
        "production": "原有 60 日動能、跳過 10 日、8 檔等權。",
        "multi_momentum": "60／120／252 日動能（跳過 10 日）的每日橫截面排名平均。",
        "trend_quality": "50% 多視窗動能＋25% 距年高排名＋25% 近 60 日上漲比例排名。",
        "buffered_momentum": "多視窗動能；原持股仍在前 2K 名時優先保留，以降低換手。",
        "vol_control": "多視窗動能；股票池近 60 日波動預算 15%，不加槓桿。",
        "risk_adjusted_momentum": "三視窗動能各除近 60 日年化波動，再取每日排名平均。",
        "risk_adjusted_buffered": "風險調整動能加前 2K 名原持股保留。",
        "quality_buffered": "trend_quality 分數加前 2K 名原持股保留。",
    }
    report["by_model"] = {}
    for name, dev in report["summary"]["development"].items():
        final = report["summary"]["final_diagnostic"][name]
        diagnostic_pass = bool(name != "production"
                               and dev["development_screen_pass"]
                               and final["sharpe_win_share"] >= .75
                               and final["median_paired_cagr_delta"] > 0
                               and final["median_paired_mdd_delta"] >= 0)
        report["by_model"][name] = {
            "formula": descriptions[name],
            "data_asof": report["data"]["price_asof"],
            "development_screen_pass": dev["development_screen_pass"],
            "historical_final_diagnostic_pass": diagnostic_pass,
            "status": "baseline" if name == "production" else ("historical_phase_robust" if diagnostic_pass else "research_only"),
            "exploratory_after_previous_results": name in report["exploratory_candidates"],
            "development": dev, "final_diagnostic": final,
            "replace_default": False,
            "ui_note": "歷史研究選項；末段已看過，尚無前瞻驗證，不代表實盤收益。",
        }
    # Recommendations use historical diagnostics and must be labelled as a
    # new exploratory judgement, rather than the original development winner.
    report["suggested_app_research_options"] = [
        name for name in ("buffered_momentum", "multi_momentum")
        if report["by_model"].get(name, {}).get("historical_final_diagnostic_pass")]
    report["suggestion_status"] = "post_hoc_historical_diagnostic_judgement_not_original_selection"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(report["phase_rows"]).to_csv(output / "phase_metrics.csv", index=False, encoding="utf-8-sig")
    flattened = [{"segment": segment, "candidate": name, **item}
                 for segment, items in report["summary"].items()
                 for name, item in items.items()]
    pd.DataFrame(flattened).to_csv(output / "phase_summary.csv", index=False, encoding="utf-8-sig")
    lines = ["# 20 種換股排程可行性研究", "",
             f"行情截至 {report['data']['price_asof']}；快照 SHA256：`{report['data']['snapshot_sha256']}`。",
             f"開發區段 {report['development_start']}～{report['development_end']}；"
             f"末段診斷 {report['final_start']}～{report['final_end']}。", "",
             "所有候選用相同 50 檔快取、交易日曆、SOX／絕對動能／外資急賣閘門、8 檔等權、"
             "20 日換股、訊號後第 2 個交易日開盤成交；每單位換手成本 0.3%。",
             "股票池波動控制候選保留其原有 15% 波動預算；buffer 候選保留前 2K 名內原持股。",
             "起始現金與暖機沿用既有研究；各候選原本有效歷史要求也沿用。", "",
             "篩選條件先固定：至少 15／20 個排程的開發 Sharpe 勝同排程原策略、"
             "配對 CAGR 差的中央値為正、配對最大回撤差的中央値不為負。"
             "不挑最佳排程；末段不參與策略排序。", "",
             "所有歷史日期均已被研究過；本次通過僅代表歷史稳健性篩選，不能稱為新的樣本外證據。", ""]
    for segment, label in (("development", "開發區段"), ("final_diagnostic", "已看過末段：僅診斷")):
        lines += [f"## {label}", "",
                  "| 策略 | CAGR 中位 | Sharpe 中位 | 最差回撤 | CAGR 勝出排程 | Sharpe 勝出排程 | 配對 CAGR 差中位 | 開發篩選通過 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for name, item in report["summary"][segment].items():
            pass_label = ("是" if item["development_screen_pass"] else "否") if segment == "development" else "—"
            lines.append(f"| {name} | {item['cagr_median']:.2%} | {item['sharpe_median']:.2f} | "
                         f"{item['mdd_worst']:.2%} | {item['cagr_win_share']:.0%} | "
                         f"{item['sharpe_win_share']:.0%} | {item['median_paired_cagr_delta']:.2%} | {pass_label} |")
        lines += [""]
    lines += ["## 結論", "", f"開發篩選選出的候選：`{report['selected_on_development']}`。",
              f"最值得先提供的研究比較選項：{', '.join(report['suggested_app_research_options']) or '未發現'}。"
              "這項建議參考已看過末段的診斷，屬新的探索判斷；不同於事前開發選型。",
              "可提供為有日期標示的研究比較選項；本次證據不足以替換 default。",
              "正式替換需先改善資料真實性，固定規則並累積前瞻紙上交易及可信成交成本證據。", "",
              "## 事後探索", "", report["exploratory_status"], "",
              "若有新增：risk_adjusted_momentum 為 60／120／252 日（跳過近 10 日）動能除以同長度"
              "近 60 日日報酬年化波動後的橫截面排名平均；risk_adjusted_buffered 使用同一分數並採前 2K 保留；"
              "quality_buffered 使用既有 trend_quality 分數並採前 2K 保留。公式與參數在新增回測前固定。", "",
              "## 限制", "",
              "目前 50 檔股票池含選擇／存活者偏差；快取除權息與籌碼發布時間未獨立核實；"
              "未模擬漲跌停無法成交、成交量容量、精細滑價及現金利息。排程彼此高度相關，"
              "20 種排程不是 20 次獨立樣本。新增候選及本次篩選門檻都是看過原研究後的探索。", "",
              "重播：`python -m core.strategy_feasibility --output research/feasibility_2026_10_03`。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feasibility(output_dir, allow_exploratory=True, verbose=True):
    settings = ResearchSettings(cost=.003, lag=2, top_k=8, rebalance_days=20)
    settings.validate()
    snapshot = load_snapshot()
    index = snapshot["close"].index
    if len(index) <= settings.train_min + settings.holdout_days:
        raise RuntimeError("insufficient common history")
    _, scores, valid = features_and_rules(snapshot["close"], snapshot["volume"])
    rows, paths = [], {}
    originals = {name: (scores[name], name) for name in ORIGINAL_NAMES}
    evaluate_candidates(originals, snapshot, settings, rows, paths, verbose)
    summary = summarize_phases(rows)
    screen_pass = [name for name, item in summary["development"].items() if item["development_screen_pass"]]
    # This decides whether to expand exploratory diagnostics, never selects a
    # strategy using the final-period result or picks a favorable phase.
    stable_diagnostic = [name for name in screen_pass
                         if summary["final_diagnostic"][name]["sharpe_win_share"] >= .75
                         and summary["final_diagnostic"][name]["median_paired_cagr_delta"] > 0
                         and summary["final_diagnostic"][name]["median_paired_mdd_delta"] >= 0]
    report = {
        "settings": asdict(settings), "data": snapshot["metadata"],
        "development_start": str(index[settings.train_min].date()),
        "development_end": str(index[-settings.holdout_days - 1].date()),
        "final_start": str(index[-settings.holdout_days].date()),
        "final_end": str(index[-1].date()), "screen_criteria": SCREEN_CRITERIA,
        "original_candidates": list(ORIGINAL_NAMES), "original_development_pass": screen_pass,
        "original_stable_final_diagnostic": stable_diagnostic,
        "exploratory_candidates": [], "phase_rows": rows, "summary": summary,
        "selected_on_development": max(screen_pass, key=lambda n: summary["development"][n]["sharpe_median"]) if screen_pass else "production",
        "replace_default": False, "production_changed": False,
        "exploratory_status": "既有候選已有通過開發篩選與跨排程末段診斷者，未追加新策略。" if stable_diagnostic else "尚未新增事後探索候選。",
    }
    output = Path(output_dir)
    write_report(report, output)
    if allow_exploratory and not stable_diagnostic:
        report["exploratory_status"] = "原候選未同时達到固定開發條件及跨排程末段診斷，追加三個固定公式；屬事後探索，不參與原研究選型。"
        extras = exploratory_scores(snapshot["close"], scores, valid)
        report["exploratory_candidates"] = list(extras)
        if verbose:
            print(report["exploratory_status"], flush=True)
        evaluate_candidates(extras, snapshot, settings, rows, paths, verbose)
        report["summary"] = summarize_phases(rows)
        eligible = [name for name, item in report["summary"]["development"].items() if item["development_screen_pass"]]
        report["selected_on_development"] = max(eligible, key=lambda n: report["summary"]["development"][n]["sharpe_median"]) if eligible else "production"
        write_report(report, output)
    pd.DataFrame(paths).to_csv(output / "daily_returns.csv", index_label="date", encoding="utf-8-sig")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research/feasibility_2026_10_03")
    parser.add_argument("--original-only", action="store_true")
    args = parser.parse_args()
    result = run_feasibility(args.output, allow_exploratory=not args.original_only)
    print("Development-screen choice:", result["selected_on_development"], flush=True)
