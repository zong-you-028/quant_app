# -*- coding: utf-8 -*-
"""
features_cs.py - 模組二之一:橫截面特徵 (Cross-Sectional Features)
在既有「時序特徵」之外,計算每個個股「當日在觀察清單(全市場代理)中的相對排名」,
例如 mom_20_cs_rank、rsi_14_cs_rank。相對強弱是「個股能否打贏大盤」的核心訊號。

★ 防 Data Leakage:
  橫截面排名只用「同一天、跨不同股票」的值(決策當下就能取得),不看任何未來;
  以 DataFrame.rank(axis=1) 一次對「每個日期列」跨股票欄做百分位,完全向量化。

資料流:
  build_feature_panel(symbols)
    → 每檔 ensure_data + build_features(時序特徵)
    → add_cs_ranks:對 CS_RANK_COLS 逐欄做每日跨股百分位,寫回各檔 {col}_cs_rank
    → 回傳 {代號: 加上橫截面欄位的特徵 DataFrame}
"""
import pandas as pd

import config
from core.data_pipeline import build_features, ensure_data


def build_feature_panel(symbols=None, with_cs: bool = True) -> dict:
    """
    對觀察清單每檔組裝特徵,回傳 {代號: DataFrame}。
    with_cs=True 時附加橫截面排名欄位(CS_FEATURE_COLS)。
    單檔載入失敗(網路/代號)直接略過,不中斷整體。
    """
    symbols = symbols or config.WATCHLIST
    panel = {}
    for s in symbols:
        try:
            ensure_data(s)
            panel[s] = build_features(s)
        except Exception:
            continue
    if not panel:
        raise RuntimeError("觀察清單全部載入失敗(無資料)")
    if with_cs:
        panel = add_cs_ranks(panel)
    return panel


def add_cs_ranks(panel: dict, rank_cols=None) -> dict:
    """
    對 panel 中每個指定欄位,計算「每日跨股票的百分位排名」(0~1),
    寫回每檔的 {col}_cs_rank 欄。某股當日缺值 -> 該欄排名以 0.5(中位)填補。
    """
    rank_cols = rank_cols or config.CS_RANK_COLS
    symbols = list(panel.keys())
    # 先複製,避免就地改到呼叫端的原始 frame
    panel = {s: panel[s].copy() for s in symbols}

    for col in rank_cols:
        # 寬表:index=日期,columns=代號,值=該特徵(只取有此欄的股票)
        wide = pd.DataFrame(
            {s: panel[s][col] for s in symbols if col in panel[s].columns}
        ).sort_index()
        if wide.empty:
            continue
        # 每一列(每個交易日)跨欄(跨股票)做百分位排名 -> 橫截面相對位置
        ranks = wide.rank(axis=1, pct=True)
        rank_name = col + "_cs_rank"
        for s in symbols:
            if s in ranks.columns:
                panel[s][rank_name] = ranks[s].reindex(panel[s].index).fillna(0.5)
            else:
                panel[s][rank_name] = 0.5     # 無此欄的股票給中位排名
    return panel


if __name__ == "__main__":
    from core.data_pipeline import seed_sample_data
    # 用多檔合成資料離線驗證橫截面邏輯(不依賴網路)
    syms = ["AAA", "BBB", "CCC", "DDD"]
    for i, s in enumerate(syms):
        seed_sample_data(s, seed=10 + i)
    panel = build_feature_panel(syms)
    a = panel["AAA"]
    print("AAA 特徵欄數:", a.shape[1], "| 樣本數:", len(a))
    print("新增橫截面欄:", [c for c in config.CS_FEATURE_COLS if c in a.columns])
    print("AAA 最近 3 日 mom_20 排名:\n", a["mom_20_cs_rank"].tail(3).round(3))
    # 同一天四檔的橫截面排名應落在 {0.25,0.5,0.75,1.0}(4 檔的百分位)
    last_day = a.index[-1]
    snap = {s: round(float(panel[s]["mom_20_cs_rank"].get(last_day, float("nan"))), 3)
            for s in syms}
    print("最後一日 mom_20_cs_rank(四檔橫截面):", snap)
