# -*- coding: utf-8 -*-
"""可解釋的技術面出場雷達（純計算，無 UI / DB 相依）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
MIN_BARS = 250


@dataclass(frozen=True)
class RadarSettings:
    horizon: str = "swing"          # short / swing / long
    atr_mode: str = "balanced"      # aggressive / balanced / loose
    buy_price: Optional[float] = None
    buy_date: Optional[str] = None
    max_loss_pct: Optional[float] = None


def _finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def clean_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """清理、排序、去重並拒絕無效 OHLC；不補造缺漏交易日。"""
    if raw is None or not isinstance(raw, pd.DataFrame):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("行情缺少欄位：" + "、".join(missing))
    df = raw.loc[:, REQUIRED_COLUMNS].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in raw.columns:
            df.index = pd.to_datetime(raw["date"], errors="coerce")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    valid = ((df[["open", "high", "low", "close"]] > 0).all(axis=1)
             & (df["high"] >= df[["open", "close", "low"]].max(axis=1))
             & (df["low"] <= df[["open", "close", "high"]].min(axis=1)))
    df = df.loc[valid]
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    return df


def add_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_ohlcv(raw)
    close = df["close"]
    for n in (5, 10, 20, 60, 120, 240):
        df[f"sma{n}"] = close.rolling(n, min_periods=n).mean()
    df["ema12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["ema26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).where(loss.ne(0), 100.0)

    prev_close = close.shift(1)
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["atr22"] = tr.ewm(alpha=1 / 22, adjust=False, min_periods=22).mean()
    df["volume_ma20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["relative_volume"] = df["volume"] / df["volume_ma20"].replace(0, np.nan)
    df["chandelier"] = df["high"].rolling(22, min_periods=22).max() - 3 * df["atr22"]
    return df


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    """回傳已被右側 K 棒確認的 pivot；最後 right 根不可能成為 pivot。"""
    highs, lows = [], []
    if len(df) < left + right + 1:
        return highs, lows
    for i in range(left, len(df) - right):
        window = df.iloc[i - left:i + right + 1]
        h, low = df["high"].iloc[i], df["low"].iloc[i]
        if h == window["high"].max() and (window["high"] == h).sum() == 1:
            highs.append((df.index[i], float(h), i + right))
        if low == window["low"].min() and (window["low"] == low).sum() == 1:
            lows.append((df.index[i], float(low), i + right))
    return highs, lows


def _signal(kind, code, title, detail, price=None):
    return {"kind": kind, "code": code, "title": title, "detail": detail,
            "price": _finite(price)}


def _fmt(value):
    return "--" if value is None else f"{value:,.2f}"


def analyze_exit_radar(raw: pd.DataFrame, settings: RadarSettings | None = None) -> dict:
    settings = settings or RadarSettings()
    df = add_indicators(raw)
    if len(df) < MIN_BARS:
        return {"available": False, "bars": len(df), "required_bars": MIN_BARS,
                "message": f"資料不足：目前 {len(df)} 個交易日，至少需要 {MIN_BARS} 個交易日。",
                "frame": df}

    latest, prev = df.iloc[-1], df.iloc[-2]
    close = float(latest["close"])
    leading, confirm, reinforce = [], [], []
    highs, lows = find_pivots(df)
    recent_highs, recent_lows = highs[-2:], lows[-2:]
    pivot_low = recent_lows[-1][1] if recent_lows else None

    hist = df["macd_hist"].dropna()
    if len(hist) >= 4 and all(hist.iloc[-i] < hist.iloc[-i-1] for i in (1, 2, 3)):
        leading.append(_signal("leading", "macd_hist_fading", "MACD 動能連續 3 日下降",
                               "動能先轉弱，但仍需價格跌破確認。"))
    if prev["macd"] >= prev["macd_signal"] and latest["macd"] < latest["macd_signal"]:
        leading.append(_signal("leading", "macd_cross_down", "MACD 死亡交叉",
                               "MACD 線今日跌破訊號線。"))
    if prev["rsi14"] >= 50 and latest["rsi14"] < 50:
        leading.append(_signal("leading", "rsi_below_50", "RSI 跌破 50",
                               f"RSI(14) 為 {latest['rsi14']:.1f}，多方動能降溫。"))

    # 價格創更高 pivot、RSI/MACD 卻形成較低 pivot，僅使用已確認 pivot。
    if len(recent_highs) == 2:
        (d1, p1, _), (d2, p2, _) = recent_highs
        if p2 > p1:
            r1, r2 = _finite(df.loc[d1, "rsi14"]), _finite(df.loc[d2, "rsi14"])
            m1, m2 = _finite(df.loc[d1, "macd"]), _finite(df.loc[d2, "macd"])
            if (r1 is not None and r2 is not None and r2 < r1) or \
               (m1 is not None and m2 is not None and m2 < m1):
                leading.append(_signal("leading", "bearish_divergence", "價格與動能頂背離",
                                       "價格波段高點墊高，但 RSI 或 MACD 高點降低。"))

    body = abs(float(latest["close"] - latest["open"]))
    upper_shadow = float(latest["high"] - max(latest["open"], latest["close"]))
    if latest["relative_volume"] >= 1.5 and upper_shadow > max(body, float(latest["atr14"]) * .35):
        leading.append(_signal("leading", "high_volume_upper_shadow", "爆量長上影",
                               f"量比 {latest['relative_volume']:.1f} 倍，盤中賣壓明顯。"))

    if close < latest["sma20"]:
        confirm.append(_signal("confirm", "below_sma20", "收盤跌破 20 日線",
                               f"收盤 {_fmt(close)} 低於 20 日線 {_fmt(latest['sma20'])}。", latest["sma20"]))
    if df["close"].iloc[-2:].lt(df["sma20"].iloc[-2:]).all():
        confirm.append(_signal("confirm", "below_sma20_two_days", "連續兩日位於 20 日線下",
                               "跌破已延續兩個收盤，假跌破機率降低。", latest["sma20"]))
    if latest["sma20"] < df["sma20"].iloc[-4]:
        confirm.append(_signal("confirm", "sma20_turning_down", "20 日線開始下彎",
                               "20 日均價低於三個交易日前。", latest["sma20"]))
    if close < latest["sma60"]:
        confirm.append(_signal("confirm", "below_sma60", "收盤跌破 60 日線",
                               f"收盤低於中期防守價 {_fmt(latest['sma60'])}。", latest["sma60"]))
    if latest["sma20"] < latest["sma60"]:
        confirm.append(_signal("confirm", "ma_structure_broken", "均線多頭排列遭破壞",
                               "20 日線已低於 60 日線。"))
    if pivot_low is not None and close < pivot_low:
        confirm.append(_signal("confirm", "below_pivot_low", "跌破最近有效波段低點",
                               f"主要價格結構防守價為 {_fmt(pivot_low)}。", pivot_low))

    breaking_support = (close < latest["sma60"] or
                        (pivot_low is not None and close < pivot_low))
    if breaking_support and latest["relative_volume"] >= 1.5:
        reinforce.append(_signal("reinforce", "break_with_volume", "放量跌破支撐",
                                 f"成交量為 20 日均量的 {latest['relative_volume']:.1f} 倍。"))
    down = (close / float(prev["close"]) - 1) < 0
    if down and body >= float(latest["atr14"]) and latest["relative_volume"] >= 1.5:
        reinforce.append(_signal("reinforce", "wide_down_bar", "放量長黑",
                                 "實體跌幅超過一個 ATR，且成交量顯著放大。"))
    if (latest["open"] < prev["low"] and close < latest["sma60"]):
        reinforce.append(_signal("reinforce", "gap_below_support", "跳空跌破中期支撐",
                                 "今日開盤低於前日低點，收盤仍未站回 60 日線。"))

    important_confirm = any(s["code"] in {"below_sma60", "below_pivot_low"} for s in confirm)
    if important_confirm and reinforce:
        level = "red"
    elif confirm:
        level = "orange"
    elif leading:
        level = "yellow"
    else:
        level = "green"

    labels = {"green": ("趨勢健康", "#2E7D32"), "yellow": ("初步轉弱", "#F9A825"),
              "orange": ("確認轉弱", "#EF6C00"), "red": ("趨勢反轉風險", "#B71C1C")}
    advice = {
        "green": "趨勢結構目前保持完整，可續抱並依波動調整移動停利。",
        "yellow": "動能開始降溫，暫停追價或加碼，並考慮收緊移動停利。",
        "orange": "趨勢已有轉弱跡象，可依原定策略分批減碼，並觀察關鍵支撐是否站回。",
        "red": "主要趨勢結構已受到破壞，請依個人風險承受能力執行出場或降低部位。",
    }
    mult = {"aggressive": 2.0, "balanced": 3.0, "loose": 4.0}.get(settings.atr_mode, 3.0)
    buy_date = pd.to_datetime(settings.buy_date, errors="coerce") if settings.buy_date else None
    held = df[df.index >= buy_date] if buy_date is not None and not pd.isna(buy_date) else df.tail(60)
    fallback = buy_date is None or pd.isna(buy_date) or held.empty
    if held.empty:
        held = df.tail(60)
    highest = float(held["high"].max())
    atr_stop = highest - mult * float(latest["atr14"])
    drawdown = close / highest - 1
    supports = [x for x in (_finite(latest["sma20"]), _finite(latest["sma60"]), pivot_low) if x]
    below = sorted([x for x in supports if x < close], reverse=True)
    first_defense = below[0] if below else min(supports) if supports else None
    key_defense = pivot_low if pivot_low is not None else _finite(latest["sma60"])
    pending = []
    if close >= latest["sma20"]:
        pending.append(f"尚未跌破 20 日線 {_fmt(latest['sma20'])}")
    if pivot_low is not None and close >= pivot_low:
        pending.append(f"尚未跌破波段低點 {_fmt(pivot_low)}")
    if latest["relative_volume"] < 1.5:
        pending.append("目前沒有 1.5 倍以上放量確認")

    return {
        "available": True, "bars": len(df), "level": level,
        "status": labels[level][0], "color": labels[level][1], "advice": advice[level],
        "leading": leading, "confirm": confirm, "reinforce": reinforce,
        "signals": leading + confirm + reinforce, "pending": pending,
        "asof": df.index[-1].strftime("%Y-%m-%d"), "horizon": settings.horizon,
        "price_basis": "回溯還原權息日線", "last_close": close,
        "sma20": _finite(latest["sma20"]), "sma60": _finite(latest["sma60"]),
        "rsi14": _finite(latest["rsi14"]), "macd_hist": _finite(latest["macd_hist"]),
        "relative_volume": _finite(latest["relative_volume"]), "atr14": _finite(latest["atr14"]),
        "atr_stop": atr_stop, "atr_multiplier": mult,
        "chandelier": _finite(latest["chandelier"]), "pivot_low": pivot_low,
        "first_defense": first_defense, "key_defense": key_defense,
        "highest_since_entry": highest, "drawdown": drawdown,
        "holding_fallback": fallback, "frame": df,
    }

