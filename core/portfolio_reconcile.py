"""Read-only stock-set comparison: executed model, upcoming target and journal."""
from __future__ import annotations

import math
import pandas as pd

from core.strategy_models import ACTIVE_MODEL_ID


def reconcile_portfolio(result: dict, positions: list) -> dict:
    """Never writes records, places orders or treats pending targets as executed.

    Eight model slots are fixed. We compare stock kinds; position sizes and
    current market-value weights need a separately chosen investment budget.
    """
    issues = []
    weights = result.get("model_current_weights")
    available = isinstance(weights, dict)
    if not available:
        weights = {}
        issues.append("尚未取得已執行的模型名單，請重新計算")
    current_weights = {}
    for symbol, value in weights.items():
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("模型權重無效")
        if weight > 0:
            current_weights[str(symbol).strip().upper()] = weight
    current = list(current_weights)
    next_target = list(dict.fromkeys(str(s).strip().upper() for s in result.get("holdings", []) if s))
    if len(current) > 8 or len(next_target) > 8 or sum(current_weights.values()) > 1.000001:
        raise ValueError("模型目標超過 8 檔或配置超過 100%，請重新計算")
    if result.get("score_mode") != ACTIVE_MODEL_ID:
        issues.append("模型與目前模式不符，請重新計算")
    quality = result.get("data_quality") or {}
    asof = result.get("model_current_asof")
    try:
        dated = (asof and quality.get("target_date")
                 and pd.Timestamp(asof) >= pd.Timestamp(quality["target_date"]))
    except (ValueError, TypeError):
        dated = False
    if quality.get("stale") is not False or not dated:
        issues.append("資料過期或日期尚未確認，只能做歷史對照")
    # Positions include manual, rotation and DCA; multiple lots count once.
    actual = {}
    for position in positions:
        symbol = str(position.get("symbol") or "").strip().upper()
        shares = float(position.get("shares", 0))
        if not symbol or not math.isfinite(shares) or shares < 0:
            raise ValueError("帳本持倉資料無效，請先核對投資紀錄")
        if shares == 0:
            continue
        row = actual.setdefault(symbol, {"symbol": symbol, "name": position.get("name") or "", "shares": 0.})
        row["shares"] += shares
    names = result.get("names") or {}

    def row(symbol):
        value = dict(actual.get(symbol, {"symbol": symbol, "name": "", "shares": 0.}))
        value["name"] = value["name"] or names.get(symbol, "")
        value["target_weight"] = current_weights.get(symbol, 0.)
        return value

    current_set, next_set = set(current), set(next_target)
    pending = current_set != next_set
    # Do not encourage selling an early purchase just to buy it back at t+2.
    early = [s for s in actual if pending and s in next_set and s not in current_set]
    exits = [s for s in actual if available and s not in current_set and s not in early]
    kept = [s for s in current if s in actual]
    unheld_outgoing = [s for s in current if s not in actual and pending and s not in next_set]
    missing = [s for s in current if s not in actual and s not in unheld_outgoing]
    verified = not issues
    return {
        "verified": verified, "state_available": available, "issues": issues, "asof": asof,
        "model_execution_date": result.get("model_current_execution_date"),
        "execution_deferred": bool(result.get("model_execution_deferred")),
        "current": [row(s) for s in current], "max_stocks": 8,
        "model_cash_weight": max(0., 1. - sum(current_weights.values())),
        "actual_count": len(actual), "keep": [row(s) for s in kept],
        "missing": [row(s) for s in missing], "exit": [row(s) for s in exits],
        "early": [row(s) for s in early],
        "unheld_outgoing": [row(s) for s in unheld_outgoing],
        "actionable_buys": missing if verified else [],
        "actionable_sells": exits if verified else [],
        "pending_change": pending, "next_target": [row(s) for s in next_target],
        "planned_execution_date": result.get("target_execution_date"),
        "planned_buys": [row(s) for s in next_target if s not in actual],
        "planned_exits": [row(s) for s in actual if s not in next_set],
    }
