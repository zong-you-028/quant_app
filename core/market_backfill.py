"""App-only, resumable official-exchange monthly price backfills.

The frozen research data pipeline is unchanged. Each validated chronological
chunk is merged and checkpointed before another request starts. Calendar-month
completion is separate from MAX(quote date), because a month may end on a
weekend or a holiday. The optional callback runs after both local commits; its
errors propagate, leaving the local rows available for persistence retry.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core import data_pipeline as dp


ORIGINAL_REFRESH = dp._refresh_market_data
PREFIX = "app_backfill:"
FIELDS = ("open", "high", "low", "close", "volume")
CHECKPOINT_KEYS = {"version", "symbol", "completed_through", "stored_last"}


def _cursor(symbol, before, expected):
    """Ignore unanchored/stale metadata; it can never skip unknown price rows."""
    initial = before + pd.Timedelta(days=1)
    raw = dp._meta_get(PREFIX + symbol)
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(value, dict) or set(value) != CHECKPOINT_KEYS:
            return initial
        if value["version"] != 1 or value["symbol"] != symbol:
            return initial
        dates = {}
        for key in ("completed_through", "stored_last"):
            text = value[key]
            date = pd.Timestamp(text)
            if not isinstance(text, str) or date.strftime("%Y-%m-%d") != text:
                return initial
            dates[key] = date
        if (dates["stored_last"] != before
                or not before <= dates["completed_through"] <= expected):
            return initial
        through = dates["completed_through"]
        if through.to_period("M") != before.to_period("M"):
            return initial
        if (through > before
                and (through != through.to_period("M").end_time.normalize()
                     or through.to_period("M") >= expected.to_period("M"))):
            return initial
        return max(initial, dates["completed_through"] + pd.Timedelta(days=1))
    except (ValueError, TypeError, KeyError, OverflowError):
        return initial


def _normalise(frame, symbol, start, end):
    """Reject malformed/mixed/out-of-range chunks before changing any rows."""
    if frame is None or frame.empty:
        raise dp.SourceUnavailableError(f"{symbol} {start:%Y-%m} 補段行情為空，未推進補抓日期")
    frame = frame.copy()
    if "date" not in frame and isinstance(frame.index, pd.DatetimeIndex):
        frame.index.name = "date"
        frame = frame.reset_index()
    if not {"date", *FIELDS} <= set(frame):
        raise dp.SourceUnavailableError(f"{symbol} 補段行情缺少必要欄位")
    if "symbol" in frame and not frame["symbol"].astype(str).eq(symbol).all():
        raise dp.SourceUnavailableError(f"{symbol} 補段行情混入其他代號")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if (dates.isna().any() or dates.duplicated().any()
            or getattr(dates.dt, "tz", None) is not None
            or not dates.eq(dates.dt.normalize()).all()
            or not dates.between(start, end).all()):
        raise dp.SourceUnavailableError(f"{symbol} 補段行情日期不符，保留已保存資料")
    values = frame.loc[:, list(FIELDS)].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(values.to_numpy()).all()
             and values.loc[:, ["open", "high", "low", "close"]].gt(0).all().all()
             and values["volume"].ge(0).all()
             and values["high"].ge(values[["open", "low", "close"]].max(axis=1)).all()
             and values["low"].le(values[["open", "high", "close"]].min(axis=1)).all())
    if not valid:
        raise dp.SourceUnavailableError(f"{symbol} 補段行情含無效價量，保留已保存資料")
    output = values.copy()
    output.insert(0, "date", dates.dt.strftime("%Y-%m-%d"))
    output.insert(0, "symbol", symbol)
    return output.sort_values("date").reset_index(drop=True)


def _checkpoint(symbol, through, checkpoint):
    stored = dp.last_ohlcv_date(symbol)
    if stored is None or stored.normalize() > through:
        raise dp.SourceUnavailableError(f"{symbol} 補段保存日期不一致")
    payload = {"version": 1, "symbol": symbol,
               "completed_through": through.strftime("%Y-%m-%d"),
               "stored_last": stored.normalize().strftime("%Y-%m-%d")}
    dp._meta_set(PREFIX + symbol, json.dumps(payload, separators=(",", ":")))
    if checkpoint is not None:
        checkpoint(symbol)


def refresh_symbol(symbol, checkpoint=None):
    """Refresh one symbol, checkpointing only multi-month cached-price gaps.

    Metadata is ``app_backfill:<symbol>`` with exactly ``version=1``, ``symbol``,
    ISO ``completed_through`` and ISO ``stored_last``. Reuse requires the stored
    last-quote anchor to match the current database and completion <= target.
    Both dates must be in the same month; completion after the quote anchor is
    allowed only at the calendar end of a fully historical month.
    Completed historical months may end after their last traded bar; a partial
    current month is checkpointed only through its actual latest committed bar.
    Uncached, single-month, fresh-price and chip-only work uses ORIGINAL_REFRESH.
    """
    symbol = str(symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol must be nonempty")
    if not dp.has_symbol(symbol):
        return ORIGINAL_REFRESH(symbol)
    before = dp.last_ohlcv_date(symbol)
    expected = dp._last_trading_day().normalize()
    if before is None:
        return ORIGINAL_REFRESH(symbol)
    before = before.normalize()
    if (len(pd.bdate_range(before, expected)) <= 2
            or (before + pd.Timedelta(days=1)).to_period("M") == expected.to_period("M")):
        return ORIGINAL_REFRESH(symbol)
    cursor = _cursor(symbol, before, expected)
    # A weekend-only tail needs no request and cannot contain a trading quote.
    while cursor <= expected and cursor.weekday() >= 5:
        cursor += pd.Timedelta(days=1)
    if cursor > expected:
        return ORIGINAL_REFRESH(symbol)
    from core import tpex_prices

    for month in pd.period_range(cursor, expected, freq="M"):
        start = max(cursor, month.start_time.normalize())
        end = min(month.end_time.normalize(), expected)
        historical = month < expected.to_period("M")
        # For an initial partial historical month, a complete response also
        # certifies a holiday-only tail after the already cached last quote.
        request_start = month.start_time.normalize() if historical else start
        dp._request_timeout()
        dp._update_phase(symbol, f"逐月保存 {month} 行情")
        if dp._is_tpex(symbol):
            raw = tpex_prices.recent_prices(symbol, request_start, end, dp._request_timeout)
        else:
            raw = dp._twse_recent_prices(symbol, asof=end, start=request_start)
        rows = _normalise(raw, symbol, request_start, end)
        dates = pd.to_datetime(rows["date"])
        fresh = rows.loc[dates >= start]
        previous = dp.last_ohlcv_date(symbol).normalize()
        if fresh.empty:
            # Only a full, successful past-month response ending exactly at our
            # existing anchor can establish a nontrading calendar tail.
            if not historical or dates.max() != previous or previous.to_period("M") != month:
                raise dp.SourceUnavailableError(f"{symbol} {month} 行情未前進，停止本輪補抓")
            _checkpoint(symbol, end, checkpoint)
        else:
            dp._merge_twse_prices(symbol, fresh)
            after = dp.last_ohlcv_date(symbol)
            if after is None or after.normalize() <= previous:
                raise dp.SourceUnavailableError(f"{symbol} {month} 行情未保存前進，停止本輪補抓")
            after = after.normalize()
            if after != pd.to_datetime(fresh["date"]).max():
                raise dp.SourceUnavailableError(f"{symbol} {month} 保存結果與補段資料不一致")
            _checkpoint(symbol, end if historical else after, checkpoint)
        if not historical:
            break  # An incompletely published current month is never busy-looped.
        cursor = end + pd.Timedelta(days=1)

    latest = dp.last_ohlcv_date(symbol)
    if (latest is not None and latest.normalize() >= expected
            and dp._chip_required(symbol) and dp.chip_needs_update(symbol)):
        dp._update_phase(symbol, "更新外資持股")
        dp.fetch_chip_data(symbol)
