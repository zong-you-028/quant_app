"""Retrospective selection-risk diagnostics from saved net-return paths only.

No price download, database access, model refit, rule tuning or app change.
CSCV diagnoses Sharpe-based selection among the disclosed candidate family;
it is not chronological forecasting validation or proof of future performance.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


APP_CANDIDATE = "buffered_momentum"
BASELINE = "production"
PBO_SOURCE = "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"


def validated_returns(frame):
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] < 2 or len(frame) < 8:
        raise ValueError("need at least two candidates and eight observations")
    if frame.columns.has_duplicates or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("candidate names and ordered observation dates must be unique")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("returns must be complete, finite and greater than -100%")
    return frame.astype(float)


def sharpes(values):
    mean = values.mean(axis=0)
    mean[np.abs(mean) < 1e-14] = 0.0
    std = values.std(axis=0, ddof=1)
    # A cash path has an explicit zero-score convention. A nonzero constant
    # return has undefined Sharpe and must not silently become a weak model.
    if ((std <= 1e-14) & (mean != 0)).any():
        raise ValueError("nonzero constant returns have undefined Sharpe")
    return np.divide(mean * np.sqrt(252), std, out=np.zeros_like(mean), where=std > 1e-14)


def cscv(frame, blocks=10, active=APP_CANDIDATE):
    """Canonical equal-block CSCV; tied IS winners share each split's weight.

    Each chronological block is indivisible. Recombined returns are used for
    statistics, never interpreted as newly executed trades. Tail rows are
    excluded only to make blocks equal. No retraining occurs across these sets.
    """
    frame = validated_returns(frame)
    if blocks < 4 or blocks % 2 or len(frame) // blocks < 2:
        raise ValueError("need an even block count and at least two rows per block")
    used = (len(frame) // blocks) * blocks
    values = frame.iloc[:used].to_numpy()
    names = list(frame.columns)
    chunks = np.arange(used).reshape(blocks, -1)
    results = []
    fixed_ranks = []
    active_position = names.index(active) if active in names else None
    for selected in itertools.combinations(range(blocks), blocks // 2):
        held_out = [b for b in range(blocks) if b not in selected]
        is_scores = sharpes(values[chunks[list(selected)].ravel()])
        oos_scores = sharpes(values[chunks[held_out].ravel()])
        ranks = pd.Series(oos_scores).rank(method="average").to_numpy()
        winners = np.flatnonzero(np.isclose(is_scores, is_scores.max(), rtol=1e-12, atol=1e-12))
        if active_position is not None:
            fixed_ranks.append(float(ranks[active_position] / (len(names) + 1)))
        for winner in winners:
            relative = float(ranks[winner] / (len(names) + 1))
            results.append({"is_blocks": ",".join(map(str, selected)),
                            "winner": names[winner], "weight": 1. / len(winners),
                            "is_sharpe": float(is_scores[winner]),
                            "oos_sharpe": float(oos_scores[winner]),
                            "relative_oos_rank": relative,
                            "rank_logit": float(np.log(relative / (1 - relative)))})
    details = pd.DataFrame(results)
    mass = details["weight"].sum()
    by_winner = {}
    for name, group in details.groupby("winner"):
        weight = group["weight"].sum()
        by_winner[name] = {"selection_share": float(weight / mass),
                           "conditional_below_median_rate": float(
                               group.loc[group["rank_logit"] <= 0, "weight"].sum() / weight)}
    report = {"blocks": blocks, "split_count": len(list(itertools.combinations(range(blocks), blocks // 2))),
              "candidate_count": len(names), "rows_used": used, "tail_rows_excluded": len(frame) - used,
              "pbo": float(details.loc[details["rank_logit"] <= 0, "weight"].sum() / mass),
              "selected_oos_negative_sharpe_rate": float(
                  details.loc[details["oos_sharpe"] < 0, "weight"].sum() / mass),
              "by_is_winner": by_winner,
              "active_fixed_below_median_rate": float(np.mean(np.asarray(fixed_ranks) <= .5)) if fixed_ranks else None,
              "selection_metric": "annualized net-return Sharpe; risk-free rate zero"}
    return report, details


def shared_block_bootstrap(frame, baseline=BASELINE, block=20, draws=3000, seed=314159):
    """Paired circular blocks, plus a centered max-mean family diagnostic.

    All candidate columns share each resampled date index. The family null is
    that no candidate's expected daily return exceeds baseline. Centering each
    difference at zero implements its least-favorable boundary. P-values and
    intervals are exploratory under block stationarity, not guarantees.
    """
    frame = validated_returns(frame)
    if baseline not in frame or block < 1 or len(frame) < 2 * block or draws < 100:
        raise ValueError("need baseline, two complete blocks and at least 100 draws")
    active = frame.drop(columns=baseline).subtract(frame[baseline], axis=0)
    values = active.to_numpy()
    mean = values.mean(axis=0) * 252
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(values)
    for start in range(0, draws, 128):
        size = min(128, draws - start)
        origins = rng.integers(0, n, size=(size, int(np.ceil(n / block))))
        indices = ((origins[..., None] + np.arange(block)) % n).reshape(size, -1)[:, :n]
        estimates.append(values[indices].mean(axis=1) * 252)
    estimates = np.concatenate(estimates)
    centered = estimates - mean
    null_max = centered.max(axis=1)
    absolute_max = np.abs(centered).max(axis=1)
    family_radius = float(np.quantile(absolute_max, .95))
    candidates = {}
    for i, name in enumerate(active):
        lower, upper = np.quantile(estimates[:, i], [.025, .975])
        candidates[name] = {"annualized_mean_active": float(mean[i]),
                            "pointwise_95_lower": float(lower), "pointwise_95_upper": float(upper),
                            "simultaneous_95_lower": float(mean[i] - family_radius),
                            "simultaneous_95_upper": float(mean[i] + family_radius),
                            "max_mean_adjusted_p": float((1 + (null_max >= mean[i]).sum()) / (draws + 1))}
    return {"block_days": block, "draws": draws, "seed": seed, "rows": n,
            "candidate_count_excluding_baseline": len(active.columns),
            "family_max_mean_p": float((1 + (null_max >= mean.max()).sum()) / (draws + 1)),
            "statistic": "252 * arithmetic mean daily return difference; not CAGR difference",
            "candidates": candidates}


def source_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_audit(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    path = source_dir / "daily_returns.csv"
    metadata = json.loads((source_dir / "results.json").read_text(encoding="utf-8"))
    all_paths = pd.read_csv(path, index_col="date", parse_dates=True)
    # The ETF was not one of the eleven strategy candidates. It is a market
    # reference and is excluded from this candidate-selection probability.
    family = validated_returns(all_paths.drop(columns="006208"))
    if set(family.columns) != set(metadata["development"]):
        raise ValueError("saved returns do not match the disclosed candidate family")
    windows = {"development": family.loc[metadata["development_start"]:metadata["development_end"]],
               "final_diagnostic": family.loc[metadata["holdout_start"]:metadata["holdout_end"]],
               "all_historical": family}
    report = {"active_candidate": APP_CANDIDATE, "baseline": BASELINE,
              "price_asof": metadata["data"]["price_asof"],
              "candidate_family": list(family.columns),
              "sources": {str(path): source_fingerprint(path),
                          str(source_dir / "results.json"): source_fingerprint(source_dir / "results.json")},
              "cscv": {}, "bootstrap": {},
              "limitations": [
                  "所有日期均已在先前研究中看過；本次是回溯診斷，不是新的樣本外證據。",
                  "PBO 描述這 11 個已揭露候選的 Sharpe 選型程序，並非 app 單一模型過擬合的機率。",
                  "原開發篩選也考量 CAGR、回撤與換手；CSCV 並未完全重現原篩選程序。",
                  "更早的專案實驗、股票池選擇與後續排程選擇，未完整納入這組候選。",
                  "CSCV 使用先前循序預測產生的報酬路徑，不為每個組合重新訓練 ML；對稱分割不能視為因果前瞻驗證。",
                  "Bootstrap 假設區塊能近似序列依賴與平穩性；p 值屬探索性診斷，會受市場階段影響。",
                  "20 種排程位移高度相關，不是 20 次獨立實驗。",
                  "除權息、籌碼發布時點、存活者偏差、成交容量、漲跌停與細部成本仍有原研究的限制。"],
              "method_source": PBO_SOURCE,
              "strategy_parameters_changed": False,
              "conclusion": "尚未排除過度擬合；固定單模式供研究與紙上驗證，不宣稱未來優勢。"}
    output_dir.mkdir(parents=True, exist_ok=True)
    for window in ("development", "all_historical"):
        report["cscv"][window] = {}
        for blocks in (8, 10, 12):
            result, details = cscv(windows[window], blocks=blocks)
            report["cscv"][window][str(blocks)] = result
            details.to_csv(output_dir / f"cscv_{window}_{blocks}.csv", index=False)
    for window, returns in windows.items():
        report["bootstrap"][window] = {}
        for block in (20, 40, 60):
            report["bootstrap"][window][str(block)] = shared_block_bootstrap(returns, block=block)
    if report["sources"][str(path)] != source_fingerprint(path):
        raise RuntimeError("source returns changed during the audit")
    (output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(report, output_dir / "summary.md")
    return report


def write_report(report, destination):
    lines = ["# 低換手多視窗模型：過度擬合診斷", "",
             f"行情截至 {report['price_asof']}；固定使用原先 11 個候選的含成本、t+2 開盤成交逐日報酬。",
             "未重新訓練、下載行情、搜尋參數或變更策略規則。", "",
             "**結論：尚未排除過度擬合，維持研究／紙上驗證用途。**", "",
             "## 1. 選型不穩定診斷（CSCV／PBO）", "",
             "將時間依序分成等長區塊，窮舉半數選型、其餘檢查的組合。PBO 是選型區段 Sharpe 贏家，",
             "換到互補區段後排名落入候選中位以下（含中位）的比例；不是此單一模型『過擬合的機率』。",
             "這是已計算報酬的對稱歷史診斷，不是新的時間順序前瞻測試。", "",
             "| 區段 | 區塊數 | 組合數 | PBO | 固定低換手模型落入下半排名比例 |",
             "|---|---:|---:|---:|---:|"]
    for window, rows in report["cscv"].items():
        for blocks, m in rows.items():
            label = "開發區段" if window == "development" else "全部已看過歷史"
            lines.append(f"| {label} | {blocks} | {m['split_count']} | {m['pbo']:.1%} | {m['active_fixed_below_median_rate']:.1%} |")
    lines += ["", "## 2. 相對原策略的優勢是否明確", "",
              "共享日期的循環區塊 bootstrap 同時重抽全部候選，保留候選間相關性。",
              "中心化後的最大平均超額報酬用於探索性多重比較校正；檢查 20／40／60 日區塊，",
              "不挑選最有利的區塊。下列數字是年化『日報酬差算術平均』，不是 CAGR 差，也不是預測報酬。", "",
              "| 區段 | 區塊天數 | 低換手年化平均超額 | 單項 95% 區間 | 全候選同時 95% 區間 | 校正 p 值 |",
              "|---|---:|---:|---|---|---:|"]
    labels = {"development": "開發區段", "final_diagnostic": "已看過末段", "all_historical": "全部已看過歷史"}
    for window, rows in report["bootstrap"].items():
        for block, row in rows.items():
            m = row["candidates"][APP_CANDIDATE]
            lines.append(f"| {labels[window]} | {block} | {m['annualized_mean_active']:.2%} | "
                         f"{m['pointwise_95_lower']:.2%}～{m['pointwise_95_upper']:.2%} | "
                         f"{m['simultaneous_95_lower']:.2%}～{m['simultaneous_95_upper']:.2%} | {m['max_mean_adjusted_p']:.3f} |")
    lines += ["", "p 值低於 0.05 才屬本診斷中較明確的正向證據；區間包含零時不能認定穩定超額。",
              "即使某一項通過，仍不足以排除研究自由度與資料偏差。", "",
              "## 3. 解讀與下一步", "",
              "低換手模型跨換股日期的歷史結果較一致，但它是在已看過末段後被採用。",
              "既有無前視／標籤到期淨化測試能檢查程式因果性，不能消除模型選擇偏差。",
              "固定目前公式與參數，更新真實行情後開始保存前瞻紙上記錄，與原策略及 006208 同期比較。",
              "不要利用本次 PBO／p 值反覆改參數，再把同一歷史當成新測試。", "",
              "## 限制", ""]
    lines += [f"- {note}" for note in report["limitations"]]
    lines += ["", f"方法一手來源：[The Probability of Backtest Overfitting]({PBO_SOURCE})。",
              "", "重播：`python -m core.overfitting_audit`。原始結果與逐組合紀錄在同目錄。"]
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="research/strategy_models_2026_10_03")
    parser.add_argument("--output", default="research/overfitting_2026_10_03")
    arguments = parser.parse_args()
    result = run_audit(arguments.source, arguments.output)
    print(result["conclusion"])
