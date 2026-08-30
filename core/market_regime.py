# -*- coding: utf-8 -*-
"""
market_regime.py - 費半(SOX)市場燈:策略切換門檻
台股動能盤(半導體為主)隔夜跟著費城半導體指數走,而 SOX 是「領先外部訊號」
(美股先收盤,台股開盤前就知道 -> 用它不算偷看未來)。
規則:SOX 跌破 N 日均線 = RISK OFF -> 整批轉現金;站回 = RISK ON -> 正常持有。
資料用 yfinance 抓 ^SOX,快取 data/sox.csv,每日更新時刷新。
"""
import os

import pandas as pd

import config

SOX_CSV = os.path.join(config.DATA_DIR, "sox.csv")


def refresh_sox(start="2014-06-01"):
    """用 yfinance 抓 ^SOX 全歷史,覆寫快取 data/sox.csv;回傳收盤 Series。"""
    try:
        import yfinance as yf
        sym = getattr(config, "ROTATION_SOX_SYMBOL", "^SOX")
        df = yf.download(sym, start=start, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return load_sox()
        s = df["Close"].squeeze()
        s.index = pd.to_datetime(s.index)
        s = s.rename("close").dropna()
        s.to_csv(SOX_CSV)
        return s
    except Exception:
        return load_sox()


def load_sox():
    """讀快取的 SOX 收盤;沒有則抓一次。失敗回 None。"""
    if os.path.exists(SOX_CSV):
        try:
            s = pd.read_csv(SOX_CSV, index_col=0)
            s.index = pd.to_datetime(s.index)
            return s.iloc[:, 0].rename("close")
        except Exception:
            pass
    return refresh_sox()


def sox_regime_series(index, ma=None, lag=None):
    """
    對齊到台股日索引的「RISK ON(1)/OFF(0)」序列。
    ★ 無前視 + 誠實時序:reindex 後 shift(lag)。lag=2 反映「台股盤後落後一天 +
      RISK OFF 隔天開盤跳空跟跌、最快也要再一天才出得掉」的真實可執行時序。
    無資料時全回 1.0(不擋,安全退回)。
    """
    ma = ma or getattr(config, "ROTATION_SOX_MA", 200)
    lag = getattr(config, "ROTATION_SOX_LAG", 2) if lag is None else lag
    s = load_sox()
    if s is None or len(s) < ma:
        return pd.Series(1.0, index=pd.DatetimeIndex(index))
    up = (s > s.rolling(ma).mean()).astype(float)
    return up.reindex(pd.DatetimeIndex(index), method="ffill").shift(lag).fillna(1.0)


def sox_status(ma=None) -> dict:
    """最新市場燈狀態:{ok, risk_on, close, ma, ma_len, asof, pct(高出均線%)}。"""
    ma = ma or getattr(config, "ROTATION_SOX_MA", 100)
    s = load_sox()
    if s is None or len(s) < ma:
        return {"ok": False, "risk_on": True}
    ma_val = float(s.rolling(ma).mean().iloc[-1])
    close = float(s.iloc[-1])
    return {"ok": True, "risk_on": bool(close > ma_val), "close": close,
            "ma": ma_val, "ma_len": ma, "asof": s.index[-1].strftime("%Y-%m-%d"),
            "pct": (close / ma_val - 1.0) * 100 if ma_val else 0.0}


if __name__ == "__main__":
    s = refresh_sox()
    print("SOX:", len(s), "筆", s.index.min().date(), "~", s.index.max().date())
    st = sox_status()
    print(f"市場燈:{'🟢 RISK ON' if st['risk_on'] else '🔴 RISK OFF'}  "
          f"SOX {st['close']:.0f} vs {st['ma_len']}MA {st['ma']:.0f} "
          f"({st['pct']:+.1f}%)  資料到 {st['asof']}")
