"""Official OTC quote readers. No database writes and no TWSE fallbacks."""
import json
import math
import time

import pandas as pd
import requests

DAILY_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
MONTH_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
_CACHE = None


def _number(value):
    try:
        number = float(str(value).replace(",", "").strip())
        return number if math.isfinite(number) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _roc_date(value):
    text = str(value).replace("/", "").strip()
    if len(text) != 7 or not text.isdigit():
        raise ValueError("Unexpected TPEx date")
    return pd.Timestamp(int(text[:3]) + 1911, int(text[3:5]), int(text[5:]))


def parse_daily(rows):
    result = []
    for row in rows:
        result.append({"symbol": str(row["SecuritiesCompanyCode"]), "date": _roc_date(row["Date"]),
                       "open": _number(row["Open"]), "high": _number(row["High"]),
                       "low": _number(row["Low"]), "close": _number(row["Close"]),
                       "volume": _number(row["TradingShares"])})
    frame = pd.DataFrame(result)
    if frame.empty:
        raise RuntimeError("TPEx latest quotes empty")
    if frame.date.nunique() != 1 or frame.duplicated(["symbol", "date"]).any():
        raise RuntimeError("TPEx latest dates or symbols inconsistent")
    return frame.dropna().query("open > 0 and high > 0 and low > 0 and close > 0 and volume >= 0")


def parse_month(payload, symbol, month):
    if payload.get("stat") != "ok" or str(payload.get("code")) != symbol:
        raise RuntimeError("TPEx monthly response unavailable or mismatched")
    tables = payload.get("tables", [])
    if not tables or not tables[0].get("data"):
        raise RuntimeError("TPEx monthly prices empty")
    fields = tables[0].get("fields", [])
    # TPEx monthly volume is in thousand shares, unlike daily OpenAPI shares.
    if (len(fields) != 9 or fields[1] not in ("成交張數", "成交仟股")
            or "開盤" != fields[3] or "收盤" != fields[6]):
        raise RuntimeError("Unexpected TPEx monthly quote schema/volume unit")
    result = []
    for row in tables[0]["data"]:
        if len(row) != 9:
            raise RuntimeError("Unexpected TPEx quote row length")
        date = _roc_date(row[0])
        if date.to_period("M") != pd.Timestamp(month).to_period("M"):
            raise RuntimeError("TPEx monthly date mismatch")
        result.append({"date": date, "open": _number(row[3]), "high": _number(row[4]),
                       "low": _number(row[5]), "close": _number(row[6]),
                       "volume": _number(row[1]) * 1000})
    frame = pd.DataFrame(result).dropna().query("open > 0 and high > 0 and low > 0 and close > 0 and volume >= 0")
    return frame.set_index("date").sort_index()


def _get(url, timeout, params=None):
    response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout())
    response.raise_for_status()
    return json.loads(response.content.decode("utf-8"))


def latest_prices(symbol, timeout=lambda: 20):
    global _CACHE
    if _CACHE is None or time.monotonic() - _CACHE[0] >= 300:
        _CACHE = (time.monotonic(), parse_daily(_get(DAILY_URL, timeout)))
    frame = _CACHE[1]
    return frame[frame.symbol == symbol].drop(columns=["symbol"]).set_index("date").copy()


def recent_prices(symbol, start, asof, timeout=lambda: 20):
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(asof).normalize()
    frames = []
    for month in pd.period_range(start, end, freq="M"):
        date = month.start_time
        payload = _get(MONTH_URL, timeout, {"code": symbol, "date": date.strftime("%Y/%m/%d"), "response": "json"})
        frames.append(parse_month(payload, symbol, date))
    frame = pd.concat(frames).sort_index()
    if frame.index.duplicated().any():
        raise RuntimeError("Duplicate TPEx quote dates")
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def clear_cache():
    global _CACHE
    _CACHE = None
