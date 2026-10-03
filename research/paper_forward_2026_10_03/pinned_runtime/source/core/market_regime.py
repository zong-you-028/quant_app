# -*- coding: utf-8 -*-
"""
market_regime.py - 費半(SOX)市場燈:策略切換門檻
台股動能盤(半導體為主)隔夜跟著費城半導體指數走,而 SOX 是「領先外部訊號」
(美股先收盤,台股開盤前就知道 -> 用它不算偷看未來)。
規則:SOX 跌破 N 日均線 = RISK OFF -> 整批轉現金;站回 = RISK ON -> 正常持有。
資料用 yfinance 抓 ^SOX,快取 data/sox.csv,每日更新時刷新。
"""
import os
import tempfile

import pandas as pd

import config

SOX_CSV = os.path.join(config.DATA_DIR, "sox.csv")


def _cached_sox(path=None):
    """Read-only fallback; never recursively starts another download."""
    try:
        frame = pd.read_csv(path or SOX_CSV, index_col=0)
        frame.index = pd.to_datetime(frame.index)
        series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").rename("close").dropna().sort_index()
        return series if not series.empty and series.gt(0).all() else None
    except (OSError, ValueError, IndexError):
        return None


def _write_sox(series):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", dir=os.path.dirname(SOX_CSV),
                                         delete=False, encoding="utf-8") as file:
            temporary = file.name
            series.to_csv(file)
        os.replace(temporary, SOX_CSV)
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


def refresh_sox(start="2014-06-01", raise_errors=False):
    """Incremental SOX refresh with a bounded request and atomic cache write."""
    cached = _cached_sox()
    try:
        if cached is None:
            # Custom APP_DATA_DIR/persistent volumes also get the public seed.
            seed = _cached_sox(os.path.join(config.BASE_DIR, "data_seed", "sox.csv"))
            if seed is not None:
                _write_sox(seed)
                cached = seed
        from core.data_pipeline import _last_trading_day
        if cached is not None and not cached.empty:
            if cached.index.max() + pd.Timedelta(days=3) >= _last_trading_day():
                return cached
            start = (cached.index.max() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        import yfinance as yf
        sym = getattr(config, "ROTATION_SOX_SYMBOL", "^SOX")
        df = yf.download(sym, start=start, progress=False, auto_adjust=True,
                         threads=False, timeout=8)
        if df is None or df.empty:
            raise RuntimeError("費半來源未回傳新資料")
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s = s.rename("close").dropna()
        if s.empty or not s.gt(0).all():
            raise RuntimeError("費半來源價格無效")
        if cached is not None:
            s = pd.concat([cached, s])
            s = s[~s.index.duplicated(keep="last")].sort_index()
        _write_sox(s)
        return s
    except Exception:
        if raise_errors:
            raise
        return cached


def load_sox():
    """讀快取的 SOX 收盤;沒有則抓一次。失敗回 None。"""
    cached = _cached_sox()
    return cached if cached is not None else refresh_sox()


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


def sox_status(ma=None, asof=None) -> dict:
    """最新市場燈狀態:{ok, risk_on, close, ma, ma_len, asof, pct(高出均線%)}。"""
    ma = ma or getattr(config, "ROTATION_SOX_MA", 100)
    s = load_sox()
    if s is not None and asof is not None:
        s = s.loc[s.index <= pd.Timestamp(asof)]
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
