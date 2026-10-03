# -*- coding: utf-8 -*-
"""
chip_data.py - 籌碼/資金流資料(外資買賣超、融資、外資期貨未平倉)
這些是「資金流」訊號,與價格動能正交(orthogonal),是值得試的不同資訊。
抓 FinMind 免費資料 -> 快取成 pickle(data/chip_cache.pkl),供 chip_lab 測試。
  外資買賣超 : TaiwanStockInstitutionalInvestorsBuySell(篩外資,net = buy - sell,股)
  融資       : TaiwanStockMarginPurchaseShortSale(融資餘額 / 融資限額 -> 使用率)
  外資期貨   : TaiwanFuturesInstitutionalInvestors(TX 台指期,外資多空未平倉淨口數)
"""
import os
import pickle

import pandas as pd

import config
from core.data_pipeline import _finmind_get

CACHE = os.path.join(config.DATA_DIR, "chip_cache.pkl")
_FOREIGN = {"Foreign_Investor", "Foreign_Dealer_Self", "外資", "外資及陸資(不含外資自營商)",
            "外資自營商", "外資及陸資"}


def _foreign_net(symbol, start):
    """外資每日淨買超(股):buy - sell,加總外資相關法人。"""
    df = _finmind_get("TaiwanStockInstitutionalInvestorsBuySell", symbol, start)
    if df is None or df.empty or "name" not in df.columns:
        return pd.Series(dtype=float)
    f = df[df["name"].isin(_FOREIGN)].copy()
    if f.empty:
        return pd.Series(dtype=float)
    f["net"] = (pd.to_numeric(f["buy"], errors="coerce")
                - pd.to_numeric(f["sell"], errors="coerce"))
    s = f.groupby("date")["net"].sum()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _margin(symbol, start):
    """融資餘額 bal 與融資限額 lim(張或股,單位一致即可算使用率)。"""
    df = _finmind_get("TaiwanStockMarginPurchaseShortSale", symbol, start)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    bal = pd.to_numeric(df.get("MarginPurchaseTodayBalance"), errors="coerce")
    lim = pd.to_numeric(df.get("MarginPurchaseLimit"), errors="coerce")
    return pd.DataFrame({"bal": bal, "lim": lim})


def foreign_futures_net(start="2015-01-01"):
    """外資台指期(TX)淨未平倉口數(多 - 空);正=偏多、負=淨空。市場層級訊號。"""
    df = _finmind_get("TaiwanFuturesInstitutionalInvestors", "TX", start)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    f = df[df["institutional_investors"].isin(_FOREIGN)].copy()
    if f.empty:                                  # 名稱可能不同,印出供除錯
        print("期貨法人名稱:", df["institutional_investors"].unique()[:6])
        return pd.Series(dtype=float)
    f["date"] = pd.to_datetime(f["date"])
    net = (pd.to_numeric(f["long_open_interest_balance_volume"], errors="coerce")
           - pd.to_numeric(f["short_open_interest_balance_volume"], errors="coerce"))
    out = pd.Series(net.values, index=f["date"]).groupby(level=0).sum().sort_index()
    return out


def build_chip_cache(symbols, start="2015-01-01"):
    """抓全部籌碼資料並快取;回傳 (flow, margin, fut)。"""
    flow, margin = {}, {}
    for i, s in enumerate(symbols):
        try:
            fn = _foreign_net(s, start)
            if not fn.empty:
                flow[s] = fn
        except Exception:
            pass
        try:
            mg = _margin(s, start)
            if not mg.empty:
                margin[s] = mg
        except Exception:
            pass
        print(f"  [{i+1}/{len(symbols)}] {s} flow={len(flow.get(s,[]))} margin={len(margin.get(s,[]))}")
    fut = foreign_futures_net(start)
    pickle.dump({"flow": flow, "margin": margin, "fut": fut}, open(CACHE, "wb"))
    print(f"已快取 -> {CACHE}  (flow {len(flow)} 檔, margin {len(margin)} 檔, fut {len(fut)} 日)")
    return flow, margin, fut


def load_chip_cache():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    return None


def holding_ratio_panel(symbols, index):
    """
    外資持股比例(big_shares/total_shares)面板,對齊交易日、ffill。
    來源:DB chip_weekly(fetch_real_data 每日更新時一併刷新)。無資料回 None。
    """
    from core.data_pipeline import load_chip_weekly
    out = {}
    for s in symbols:
        try:
            c = load_chip_weekly(s)
            if c is None or c.empty:
                continue
            r = (c["big_shares"] / c["total_shares"].replace(0, pd.NA)).dropna()
            if len(r) > 60:
                out[s] = r.astype(float)
        except Exception:
            continue
    if not out:
        return None
    return pd.DataFrame(out).sort_index().reindex(index).ffill()


def fastsell_z_panel(symbols, index, win=None):
    """
    外資「急賣」z 分數面板:z = 持股比 win 日變化 ÷ 自身近一年(250日)win日變化波動。
    z 很負 = 外資正在急速出貨。無資料回 None(閘門自動失效,安全)。
    """
    import config as _cfg
    win = win or getattr(_cfg, "ROTATION_FASTSELL_WIN", 5)
    hold = holding_ratio_panel(symbols, index)
    if hold is None:
        return None
    d = hold.diff(win)
    return d / d.rolling(250).std().replace(0, pd.NA)


def margin_usage_panel(symbols, index):
    """
    回傳「融資使用率(餘額/限額)」面板 DataFrame(date×symbol),對齊 index 並 ffill。
    給防禦模式用;無快取或無資料回 None(呼叫端退回標準模式)。
    """
    cache = load_chip_cache()
    if not cache or not cache.get("margin"):
        return None
    margin = cache["margin"]
    out = {}
    for s in symbols:
        m = margin.get(s)
        if m is None or m.empty:
            continue
        u = (m["bal"] / m["lim"].replace(0, float("nan")))
        out[s] = u
    if not out:
        return None
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().reindex(index).ffill()


if __name__ == "__main__":
    from core.signal_lab import DELISTED
    syms = list(config.UNIVERSE) + DELISTED
    build_chip_cache(syms)
