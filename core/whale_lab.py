# -*- coding: utf-8 -*-
"""
whale_lab.py - 「大戶買慢賣快」假說測試
假說:大戶(外資)怕推價 -> 緩慢分批買(吸籌);出事才賣 -> 動作快。
故「持有量曲線的斜率/平穩度」與「急賣偵測」可能是有效特徵。
與 chip_lab 測過的「流量水位濾網」不同 —— 這裡測的是「形狀」:
  slope   : 外資持股比 20 日變化(吸籌斜率)-> 與動能綜合排名
  steady  : 60 日內外資買超天數占比(吸籌平穩度;籌碼版 FIP)-> 前2K強挑最穩
  fastsell: 5 日持股比驟降 z 分數(急賣)-> 出場閘門(該 slot 轉現金)
評估同一套:去偏池(50+6 下市)、OOS(後40%)、2022 空頭、含 0.003 成本。
"""
import numpy as np
import pandas as pd

import config
from core.signal_lab import load_closes, backtest_signal, evaluate, DELISTED
from core.data_pipeline import load_chip_weekly
from core.chip_data import load_chip_cache


def _holding_ratio_panel(symbols, index):
    """外資持股比例(big_shares/total_shares)面板,對齊交易日、ffill。"""
    out = {}
    for s in symbols:
        try:
            c = load_chip_weekly(s)
            if c is None or c.empty:
                continue
            r = (c["big_shares"] / c["total_shares"].replace(0, np.nan)).dropna()
            if len(r) > 60:
                out[s] = r
        except Exception:
            continue
    if not out:
        return None
    return pd.DataFrame(out).sort_index().reindex(index).ffill()


def run():
    syms = list(config.UNIVERSE) + DELISTED
    close_df, ret_df = load_closes(syms)
    idx, cols = ret_df.index, ret_df.columns
    gate = close_df.pct_change(60, fill_method=None)
    mom = close_df.shift(10).pct_change(50, fill_method=None)      # 現行 skip10

    # --- 大戶行為特徵 ---
    hold = _holding_ratio_panel(syms, idx)                          # 外資持股比
    if hold is None:
        raise RuntimeError("無外資持股資料")
    slope20 = hold.diff(20)                                         # 吸籌斜率
    # 急賣:5 日持股比變化的 z 分數(相對自己近一年的 5 日變化波動)
    d5 = hold.diff(5)
    fastsell_z = d5 / d5.rolling(250).std().replace(0, np.nan)
    # 吸籌平穩度:60 日內「外資淨買超為正」天數占比(用每日買賣超流量)
    cache = load_chip_cache()
    steady = None
    if cache and cache.get("flow"):
        fnet = pd.DataFrame({s: cache["flow"][s] for s in cache["flow"]}).reindex(idx)
        steady = (fnet > 0).rolling(60).mean()

    k = config.ROTATION_TOP_K
    cost = config.COST_PER_TURNOVER
    rebal = config.ROTATION_REBAL_DAYS

    def bt(mode):
        w = pd.DataFrame(np.nan, index=idx, columns=cols)
        for d in idx[::rebal]:
            row = mom.loc[d].dropna().sort_values(ascending=False)
            g = gate.loc[d]
            cand = [s for s in row.index if pd.notna(g.get(s)) and g.get(s) > 0]
            if mode == "slope_combo":       # 動能排名 + 吸籌斜率排名
                rk_m = row.rank(pct=True)
                srow = slope20.loc[d] if d in slope20.index else pd.Series(dtype=float)
                rk_s = srow.reindex(row.index).rank(pct=True)
                comb = rk_m.add(rk_s, fill_value=0.5).reindex(cand)
                picks = list(comb.sort_values(ascending=False).head(k).index)
            elif mode == "steady_filter":   # 前2K強挑吸籌最平穩
                pool = cand[:2 * k]
                st = steady.loc[d] if (steady is not None and d in steady.index) else pd.Series(dtype=float)
                ranked = sorted(pool, key=lambda s: -(st.get(s) if pd.notna(st.get(s)) else 0.5))
                picks = ranked[:k]
            elif mode == "fastsell_gate":   # 基準選股 + 急賣出場(z<-1.5 該slot轉現金)
                z = fastsell_z.loc[d] if d in fastsell_z.index else pd.Series(dtype=float)
                picks = [s for s in cand[:k]
                         if not (pd.notna(z.get(s)) and z.get(s) < -1.5)]
            elif mode == "combo_all":       # 綜合:平穩度挑 + 急賣踢
                pool = cand[:2 * k]
                st = steady.loc[d] if (steady is not None and d in steady.index) else pd.Series(dtype=float)
                ranked = sorted(pool, key=lambda s: -(st.get(s) if pd.notna(st.get(s)) else 0.5))
                z = fastsell_z.loc[d] if d in fastsell_z.index else pd.Series(dtype=float)
                picks = [s for s in ranked
                         if not (pd.notna(z.get(s)) and z.get(s) < -1.5)][:k]
            else:                            # baseline
                picks = cand[:k]
            ww = pd.Series(0.0, index=cols)
            for s in picks:
                ww[s] = 1.0 / k
            w.loc[d] = ww.values
        w = w.ffill().fillna(0.0)
        port = (w.shift(1).fillna(0.0) * ret_df.fillna(0.0)).sum(axis=1)
        turn = w.diff().abs().sum(axis=1)
        turn.iloc[0] = w.iloc[0].abs().sum()
        return port - turn * cost

    names = {"baseline": "基準 skip10", "slope_combo": "+吸籌斜率綜合",
             "steady_filter": "+吸籌平穩度(前16挑8)", "fastsell_gate": "+急賣出場閘門",
             "combo_all": "平穩挑+急賣踢"}
    print(f"外資持股面板 {hold.shape[1]} 檔 · 流量 {0 if steady is None else steady.shape[1]} 檔")
    print(f"{'設定':<22}{'CAGR':>8}{'Sharpe':>8}{'OOS':>7}{'MDD':>8}{'2022空頭':>9}")
    for m in ["baseline", "slope_combo", "steady_filter", "fastsell_gate", "combo_all"]:
        r = evaluate(bt(m))
        print(f"{names[m]:<22}{r['cagr']*100:>7.1f}%{r['sharpe']:>8.2f}"
              f"{r['oos_sharpe']:>7.2f}{r['mdd']*100:>7.1f}%{r['bear']*100:>+8.1f}%")


if __name__ == "__main__":
    run()
