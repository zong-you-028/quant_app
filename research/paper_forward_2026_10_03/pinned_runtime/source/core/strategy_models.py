"""One causal app ranking model, with explicit variants for research replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


# The app has one strategy. Other specifications are retained only so historical
# research and explicit baseline replays remain reproducible.
ACTIVE_MODEL_ID = "buffered_momentum"


MODEL_SPECS = {
    "production": {"label": "基準｜60 日動能", "description": "跳過近期 10 日，依相對動能選股。",
                   "status": "既有基準", "score_kind": "momentum_return"},
    "multi_momentum": {"label": "多視窗動能", "description": "60／120／252 日動能排名平均，降低單一視窗依賴。",
                       "status": "研究候選", "score_kind": "percentile"},
    "trend_quality": {"label": "趨勢穩定模型", "description": "多視窗動能、接近年高與上漲天數比例的固定組合。",
                      "status": "研究候選・排程敏感", "score_kind": "percentile"},
    "buffered_momentum": {"label": "低換手多視窗模型", "description": "上期模型名單仍在前 16 名優先保留，再以中長期動能補足 8 檔。",
                          "status": "歷史跨排程檢驗通過・過擬合未排除・待前瞻驗證", "score_kind": "percentile"},
}


def model_spec(mode=None):
    mode = mode or ACTIVE_MODEL_ID
    if mode not in MODEL_SPECS:
        raise ValueError(f"unknown ranking model: {mode}")
    return mode, dict(MODEL_SPECS[mode])


def active_model_spec():
    """Metadata for the single strategy used by every app workflow."""
    return model_spec(ACTIVE_MODEL_ID)[1]


def rank_components(closes):
    """Same-date ranks; missing listing history is never forward-filled."""
    valid = closes.gt(0) & closes.notna().rolling(252).sum().ge(252)
    momentum = {h: closes.shift(10) / closes.shift(h) - 1 for h in (60, 120, 252)}
    ranks = {h: v.where(valid).replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)
             for h, v in momentum.items()}
    returns = closes.pct_change(fill_method=None)
    vol = returns.rolling(60).std().replace(0, np.nan)
    high = (closes / closes.rolling(252).max() - 1).where(valid).rank(axis=1, pct=True)
    stability = (returns > 0).astype(float).where(returns.notna()).rolling(60).mean()
    stability = stability.where(valid).rank(axis=1, pct=True)
    risk_ranks = {h: (v / vol).where(valid).replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)
                  for h, v in momentum.items()}
    return ranks, high, stability, risk_ranks


def build_rank_scores(closes, mode=None, mom_days=None, skip_days=None):
    """Price model scores; percentile values are not forecast returns."""
    mode, _ = model_spec(mode)
    if mode == "production":
        mom = config.ROTATION_MOM_DAYS if mom_days is None else mom_days
        skip = config.ROTATION_SKIP_DAYS if skip_days is None else skip_days
        return closes.shift(skip) / closes.shift(mom) - 1
    ranks, high, stability, risk_ranks = rank_components(closes)
    multi = sum(ranks.values()) / 3
    if mode in ("multi_momentum", "buffered_momentum"):
        return multi
    if mode == "trend_quality":
        return .5 * multi + .25 * high + .25 * stability
    return sum(risk_ranks.values()) / 3


def data_quality(symbols=None, prices=None, asof=None, chip_dates=None):
    """Read-only freshness summary using Taipei's existing update target rule.

    target_date is an expected publication date, not a verified exchange holiday
    calendar. Each symbol's date is inspected, rather than the union's max date.
    """
    from core.data_pipeline import _last_trading_day, get_conn, ensure_db
    symbols = list(dict.fromkeys(symbols or config.UNIVERSE))
    target = _last_trading_day(asof)
    if prices is None:
        ensure_db()
        conn = get_conn()
        try:
            query_symbols = list(dict.fromkeys([*symbols, config.BENCHMARK_SYMBOL]))
            values = conn.execute(
                "SELECT symbol,MAX(date) FROM ohlcv WHERE symbol IN (" +
                ",".join("?" for _ in query_symbols) + ") GROUP BY symbol", query_symbols).fetchall()
        finally:
            conn.close()
        last_dates = {s: pd.Timestamp(d) for s, d in values if d is not None}
    else:
        last_dates = {}
        for symbol in dict.fromkeys([*symbols, config.BENCHMARK_SYMBOL]):
            frame = prices.get(symbol)
            if frame is not None and not frame.empty and "close" in frame:
                valid = frame.loc[frame["close"].notna() & frame["close"].gt(0)]
                if not valid.empty:
                    last_dates[symbol] = pd.Timestamp(valid.index.max())
    stale_symbols = [s for s in symbols if s not in last_dates or last_dates[s].normalize() < target]
    latest = max(last_dates.values()) if last_dates else None
    oldest = min(last_dates.values()) if last_dates else None
    benchmark_date = last_dates.get(config.BENCHMARK_SYMBOL)
    benchmark_stale = benchmark_date is None or benchmark_date.normalize() < target
    reasons = []
    if stale_symbols:
        reasons.append(f"{len(stale_symbols)} 檔股票未達資料目標")
    if benchmark_stale:
        reasons.append("006208 基準資料未達目標")
    stale_chip_symbols = []
    chip_asof = None
    if getattr(config, "ROTATION_FASTSELL_GATE", False):
        chip_symbols = [s for s in symbols if s != config.BENCHMARK_SYMBOL]
        if chip_dates is None:
            ensure_db()
            conn = get_conn()
            try:
                values = conn.execute(
                    "SELECT symbol,MAX(date) FROM chip_weekly WHERE symbol IN (" +
                    ",".join("?" for _ in chip_symbols) + ") GROUP BY symbol", chip_symbols).fetchall()
                chip_dates = {s: pd.Timestamp(d) for s, d in values if d is not None}
            finally:
                conn.close()
        chip_dates = {s: pd.Timestamp(d) for s, d in chip_dates.items() if d is not None}
        # The cache is weekly-compatible. This catches gross staleness; it does
        # not attest to the source's release time or exchange holiday calendar.
        stale_chip_symbols = [s for s in chip_symbols if s not in chip_dates
                              or chip_dates[s].normalize() < target - pd.Timedelta(days=7)]
        available = [chip_dates[s] for s in chip_symbols if s in chip_dates]
        chip_asof = min(available) if available else None
        if stale_chip_symbols:
            reasons.append(f"{len(stale_chip_symbols)} 檔外資持股資料過期或缺漏（容許 7 日）")
    sox_date = None
    if config.ROTATION_SOX_GATE:
        try:
            sox = pd.read_csv(Path(config.DATA_DIR) / "sox.csv", index_col=0)
            sox_date = pd.Timestamp(pd.to_datetime(sox.index).max())
        except (OSError, ValueError, IndexError):
            pass
        # Conservative gross-staleness check allows US/Taiwan holiday mismatch.
        if sox_date is None or sox_date + pd.Timedelta(days=3) < target:
            reasons.append("SOX 市場資料過期或缺漏")
    return {"asof": str(latest.date()) if latest is not None else None,
            "oldest_asof": str(oldest.date()) if oldest is not None else None,
            "target_date": str(target.date()), "stale": bool(reasons),
            "benchmark_asof": str(benchmark_date.date()) if benchmark_date is not None else None,
            "sox_asof": str(sox_date.date()) if sox_date is not None else None,
            "chip_asof": str(chip_asof.date()) if chip_asof is not None else None,
            "stale_chip_symbols": stale_chip_symbols,
            "reasons": reasons,
            "stale_symbols": stale_symbols, "coverage": (len(symbols)-len(stale_symbols))/len(symbols) if symbols else 0.,
            "total_symbols": len(symbols), "symbol_dates": {s: str(d.date()) for s, d in last_dates.items()},
            "target_basis": "台北時間18:00與工作日推估；未接入正式休市日曆"}


def validation_status(mode):
    """Expose research status, never treat retrospective research as live proof."""
    mode, spec = model_spec(mode)
    path = Path(config.BASE_DIR) / "research" / "feasibility_2026_10_03" / "results.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        candidates = report.get("by_model", report.get("candidates", {}))
        candidate = candidates.get(mode, {})
        passed = candidate.get("development_screen_pass", False)
        final_diagnostic = candidate.get("historical_final_diagnostic_pass", False)
        if passed and final_diagnostic:
            return spec["status"]
    except (OSError, ValueError, TypeError):
        pass
    return spec["status"]
