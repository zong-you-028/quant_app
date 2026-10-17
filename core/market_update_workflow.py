"""One user action restores shared data and drains bounded update batches.

Importing this module performs no storage or network operations. Each batch
uses the daily updater's own lease and commit policy; this layer only schedules
unfinished symbols and combines their latest confirmed outcomes.
"""
from __future__ import annotations

import math
import time

from core import daily_market_update as dmu


STATUSES = ("updated", "current", "stale", "failed", "pending")


def _date(value):
    return str(value)[:10] if value is not None else None


def _advanced(before, after):
    # A first observation establishes the baseline; it does not prove that a
    # source call advanced storage rather than returning existing seed data.
    return before is not None and after is not None and _date(after) > _date(before)


def _safe_storage():
    try:
        value = dmu.storage_status()
        return value if isinstance(value, dict) else {"status": value}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def _sox_date(value):
    if isinstance(value, dict):
        return _date(value.get("asof"))
    if value is None or getattr(value, "empty", False):
        return None
    index = getattr(value, "index", None)
    return _date(index[-1]) if index is not None and len(index) else None


def _statuses(result, queue):
    supplied = result.get("symbol_statuses") or {}
    answer = {}
    for symbol in queue:
        status = supplied.get(symbol)
        if isinstance(status, dict):
            status = status.get("status")
        if status not in STATUSES:
            status = next((name for name in ("pending", "failed", "stale", "updated", "current")
                           if symbol in result.get(name + "_symbols", [])), "pending")
        answer[symbol] = status
    return answer


def run_update(symbols, *, startup=False, on_start=None, progress=None,
               on_saved=None, time_budget_seconds=90, max_batches=64):
    """Run bounded chunks automatically, with fair and finite source retries.

    Unattempted pending symbols go ahead of an interrupted symbol. A failed or
    stale source gets at most one further try after pending work has drained;
    declared nonretryable errors never repeat. An interrupted monthly backfill
    can continue beyond two attempts while its stored price/chip date advances.
    Two source attempts without such progress, two identical idle batches, or
    ``max_batches`` stop continuation with honest pending outcomes and a reason.
    """
    if not isinstance(max_batches, int) or isinstance(max_batches, bool) or max_batches < 1:
        raise ValueError("max_batches must be a positive integer")
    if (not isinstance(time_budget_seconds, (int, float))
            or isinstance(time_budget_seconds, bool)
            or not math.isfinite(time_budget_seconds) or not 0 < time_budget_seconds <= 300):
        raise ValueError("time_budget_seconds must be inside (0, 300]")
    syms = list(dict.fromkeys(str(s).strip().upper() for s in symbols if s and str(s).strip()))
    if "006208" in syms:
        syms.remove("006208")
        syms.insert(0, "006208")
    started = time.monotonic()
    restoration = None
    if on_start:
        on_start(0, len(syms), "", "還原已保存的雲端行情")
    try:
        restoration = dmu.restore_cache()
    except Exception as exc:
        restoration = {"status": "failed", "error": str(exc)}
    if (isinstance(restoration, dict) and restoration.get("configured")
            and not restoration.get("durable")):
        message = "共用行情資料庫目前無法讀取，暫停下載；既有快取仍可檢視，請稍後再更新。"
        detail = restoration.get("error") or restoration.get("message") or message
        return {
            "busy": False, "updated": 0, "current": 0, "stale": 0, "failed": 0,
            "pending": len(syms), "pending_symbols": list(syms),
            "failed_symbols": [], "stale_symbols": [],
            "symbol_statuses": dict.fromkeys(syms, "pending"),
            "data_dates": {}, "chip_dates": {}, "errors": {},
            "asof": None, "latest_asof": None, "expected_asof": None,
            "attempted_symbols": [], "retried_symbols": [], "recovered": 0,
            "nonretryable_symbols": [], "stale_chip_symbols": [],
            "daily_skipped_symbols": [], "on_saved_errors": {},
            "batches": 0, "automatic_continuation": False, "time_budget_reached": False,
            "continuation_stop_reason": message, "public_message": message,
            "no_source_progress_symbols": [], "sox_error": "", "sox_asof": None,
            "restoration": restoration, "storage": _safe_storage(),
            "cloud_errors": {"restore": str(detail)}, "cloud_error": str(detail),
            "total_symbols": len(syms), "startup": bool(startup),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
    sox_error, sox_asof = "", None
    try:
        if on_start:
            on_start(0, len(syms), "SOX", "先檢查費半市場資料")
        sox_asof = _sox_date(dmu.refresh_sox(raise_errors=True))
        if sox_asof is None:
            raise RuntimeError("沒有可用費半資料日期")
    except Exception as exc:
        sox_error = str(exc)

    statuses = {s: "pending" for s in syms}
    data_dates, chip_dates, errors, saved_errors = {}, {}, {}, {}
    attempts = {s: 0 for s in syms}
    stagnant_attempts = {s: 0 for s in syms}
    ever_updated, ever_attempted, seen = set(), set(), set()
    nonretryable, stale_chips, daily_skipped, retried, blocked = set(), set(), set(), set(), set()
    queue = list(syms)
    phase = "initial"
    batches, idle_repeats = 0, 0
    last_fingerprint = None
    stop_reason, busy = "", False
    latest = {}
    cloud_errors = {}

    def starting(done, total, symbol, status):
        if on_start:
            on_start(len(seen), len(syms), symbol, status)

    def completed(done, total, symbol, status):
        if symbol in statuses:
            seen.add(symbol)
        if progress:
            progress(len(seen), len(syms), symbol, status)

    while queue and batches < max_batches:
        batches += 1
        if batches > 1 and on_start:
            on_start(len(seen), len(syms), "", f"自動接續更新第 {batches} 批")
        prior_dates, prior_chips = dict(data_dates), dict(chip_dates)
        prior_statuses = dict(statuses)
        try:
            latest = dmu.update_symbols(
                queue, ignore_throttle=not startup, max_attempts=1,
                time_budget_seconds=time_budget_seconds,
                on_start=starting, progress=completed, on_saved=on_saved)
        except Exception as exc:
            stop_reason = "資料更新中止：" + str(exc)
            errors.update({s: str(exc) for s in queue})
            break
        if latest.get("busy"):
            busy = True
            stop_reason = "另一個畫面正在更新，未啟動重複下載。"
            break
        batch_statuses = _statuses(latest, queue)
        attempted = set(latest.get("attempted_symbols") or []) & set(queue)
        ever_attempted.update(attempted)
        data_dates.update({s: _date(value) for s, value in latest.get("data_dates", {}).items() if s in statuses})
        chip_dates.update({s: _date(value) for s, value in latest.get("chip_dates", {}).items() if s in statuses})
        saved_errors.update(latest.get("on_saved_errors") or {})
        stale_chips.difference_update(queue)
        stale_chips.update(latest.get("stale_chip_symbols") or [])
        nonretryable.update(latest.get("nonretryable_symbols") or [])
        daily_skipped.update(latest.get("daily_skipped_symbols") or [])
        for key in ("cloud_save_error", "cloud_save_errors", "cloud_errors"):
            value = latest.get(key)
            if isinstance(value, dict):
                cloud_errors.update(value)
            elif value:
                cloud_errors[f"batch_{batches}"] = value
        for symbol in latest.get("cloud_saved_symbols", []):
            cloud_errors.pop(symbol, None)
        source_progress = False
        for symbol, status in batch_statuses.items():
            statuses[symbol] = status
            if status == "updated":
                ever_updated.add(symbol)
            if symbol in (latest.get("errors") or {}):
                errors[symbol] = latest["errors"][symbol]
            elif status not in ("failed", "pending"):
                errors.pop(symbol, None)
            advanced = (_advanced(prior_dates.get(symbol), data_dates.get(symbol))
                        or _advanced(prior_chips.get(symbol), chip_dates.get(symbol)))
            source_progress |= advanced or (status != "pending" and prior_statuses[symbol] == "pending")
            if symbol in attempted:
                attempts[symbol] += 1
                stagnant_attempts[symbol] = 0 if advanced else stagnant_attempts[symbol] + 1
            if status == "pending" and stagnant_attempts[symbol] >= 2:
                blocked.add(symbol)
        remaining = [s for s in queue if statuses[s] == "pending" and s not in blocked]
        # Stable order within each group makes source behavior reproducible.
        remaining.sort(key=lambda s: attempts[s] > 0)
        fingerprint = (tuple(remaining), tuple(sorted(attempted)),
                       tuple((s, data_dates.get(s), chip_dates.get(s)) for s in remaining))
        if remaining and not source_progress:
            idle_repeats = idle_repeats + 1 if fingerprint == last_fingerprint else 1
            if idle_repeats >= 2:
                blocked.update(remaining)
                remaining = []
        else:
            idle_repeats = 0
        last_fingerprint = fingerprint
        if cloud_errors:
            stop_reason = "雲端行情保存未完成，已暫停後續下載；本機進度仍保留，請稍後重試保存。"
            break
        if remaining:
            queue = remaining
            continue
        if phase == "initial":
            phase = "retry"
            retry = [s for s in syms if statuses[s] in ("failed", "stale")
                     and s not in nonretryable and attempts[s] < 2]
            if retry:
                retried.update(retry)
                queue = retry
                idle_repeats, last_fingerprint = 0, None
                continue
        queue = []

    pending = [s for s in syms if statuses[s] == "pending"]
    if not stop_reason and pending:
        stop_reason = ("自動接續已達批次上限，已完成資料已保存。"
                       if queue and batches >= max_batches
                       else "來源連續未推進資料日期，已停止重複請求；已完成資料已保存。")
    # Retaining an earlier successful update avoids reclassifying it as a cache
    # hit merely because another chunk reported it as current later.
    for symbol in ever_updated:
        if statuses[symbol] == "current":
            statuses[symbol] = "updated"
    result = {key: value for key, value in latest.items() if key not in (
        *STATUSES, "symbol_statuses", "data_dates", "chip_dates", "errors",
        "failed_symbols", "stale_symbols", "pending_symbols", "retried_symbols",
        "nonretryable_symbols", "stale_chip_symbols", "daily_skipped_symbols")}
    result.update({name: sum(value == name for value in statuses.values()) for name in STATUSES})
    for name in ("failed", "stale", "pending"):
        result[name + "_symbols"] = [s for s in syms if statuses[s] == name]
    dates = [data_dates.get(s) for s in syms if data_dates.get(s)]
    result.update({
        "busy": busy, "symbol_statuses": statuses, "data_dates": data_dates,
        "chip_dates": chip_dates, "errors": errors, "on_saved_errors": saved_errors,
        "asof": min(dates) if dates and len(dates) == len(syms) else None,
        "latest_asof": max(dates) if dates else None,
        "attempted_symbols": [s for s in syms if s in ever_attempted],
        "retried_symbols": [s for s in syms if s in retried],
        "recovered": sum(statuses[s] in ("updated", "current") for s in retried),
        "nonretryable_symbols": [s for s in syms if s in nonretryable],
        "stale_chip_symbols": [s for s in syms if s in stale_chips],
        "daily_skipped_symbols": [s for s in syms if s in daily_skipped],
        "batches": batches, "automatic_continuation": batches > 1,
        "time_budget_reached": bool(pending and latest.get("time_budget_reached")),
        "continuation_stop_reason": stop_reason,
        "no_source_progress_symbols": [s for s in syms if s in blocked],
        "sox_error": sox_error, "sox_asof": sox_asof,
        "restoration": restoration, "storage": _safe_storage(),
        "cloud_errors": cloud_errors, "total_symbols": len(syms),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "startup": bool(startup),
    })
    return result
