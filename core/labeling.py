# -*- coding: utf-8 -*-
"""
labeling.py - 模組一:三欄式標籤 (Triple-Barrier Method)
取代舊的「固定 20 日預期報酬」貼標,改用動態波動度決定停利/停損欄位:
  上欄(停利) = entry × (1 + pt_mult × σ)        先觸 → label 4(做多成功)
  下欄(停損) = entry × (1 − sl_mult × σ)        先觸 → label 0(下跌/停損)
  右欄(時間) = max_holding 個交易日內皆未觸碰  → label 2(中性盤整)
σ = 過去 vol_window 日報酬標準差(或 ATR/price);全部來自「當下或過去」資訊。

★ 防 Data Leakage:
  - σ 用 rolling(window) 結尾於 t,只看 t-window+1..t(決策時已知),不偷看未來。
  - 標籤本身依「未來路徑」判定 → 僅作為 y,本檔不產生任何特徵 X。
  - 視窗不足(資料尾端 max_holding 天)無法完整觀察者 → label=NaN 後丟棄。

★ 向量化:不對「每個進場日」做 Python 迴圈,而是對「持有位移 k=1..max_holding」
  做 max_holding 次全向量運算(semi-vectorized first-touch),記憶體與速度皆佳。

回傳 DataFrame(index 同輸入,已丟棄無法判定者):
  label(0/2/4)、t_event(幾根 K 觸界)、ret_event(進場→觸界的實現報酬)、
  barrier('up'/'dn'/'vert'/'tie')、vol(σ)、upper、lower
其中 ret_event 供模組三 Meta-Labeling 與模組五評估使用。
"""
import numpy as np
import pandas as pd

import config


def compute_volatility(df: pd.DataFrame, window: int = 20,
                       method: str = "std") -> pd.Series:
    """
    動態波動度 σ(以「占價格的比例」表示,直接乘在價格上當作欄位寬度)。
      method='std':過去 window 日「報酬」標準差(預設)。
      method='atr':ATR(window) / close,對跳空/振幅更敏感。
    皆為結尾於 t 的 rolling,不含未來。
    """
    close = df["close"]
    if method == "atr":
        high, low = df["high"], df["low"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low),
                        (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
        return (tr.rolling(window).mean() / close)
    return close.pct_change().rolling(window).std()


def triple_barrier_labels(df: pd.DataFrame, vol_window: int = 20,
                          pt_mult: float = 2.0, sl_mult: float = 1.5,
                          max_holding: int = 20, use_intraday: bool = True,
                          vol_method: str = "std") -> pd.DataFrame:
    """
    產生三欄式標籤。df 需含 'close'(use_intraday=True 時另需 'high'/'low')。
    use_intraday=True:用當日 high≥上欄 / low≤下欄 判定觸界(路徑式,較貼近真實)。
    use_intraday=False:僅用 close 判定(較保守、訊號較少)。
    """
    df = df.sort_index()
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float) if use_intraday else close
    low = df["low"].to_numpy(dtype=float) if use_intraday else close
    sigma = compute_volatility(df, vol_window, vol_method).to_numpy(dtype=float)

    upper = close * (1.0 + pt_mult * sigma)        # 停利欄(價格水準)
    lower = close * (1.0 - sl_mult * sigma)        # 停損欄(價格水準)

    idx = np.arange(n)
    label = np.full(n, 2.0)                          # 預設中性(後續可能改 NaN)
    t_event = np.full(n, float(max_holding))
    ret_event = np.full(n, np.nan)
    barrier = np.full(n, "vert", dtype=object)
    resolved = np.zeros(n, dtype=bool)

    # σ 暖機期(NaN)無法定義欄位 → 標記不可用,最終丟棄
    invalid = ~np.isfinite(sigma)
    resolved[invalid] = True
    label[invalid] = np.nan

    # 對「持有位移 k」逐步推進,每步全向量比對是否觸界(先觸先定 → 由小 k 先處理)
    for k in range(1, max_holding + 1):
        j = idx + k
        in_range = j < n
        active = in_range & (~resolved)
        if not active.any():
            break                                    # 之後 k 只會更少,可安全結束
        jj = np.where(in_range, j, 0)                # 越界者塞 0(該位置 active=False 不會用到)
        up_hit = active & (high[jj] >= upper)        # 觸上欄
        dn_hit = active & (low[jj] <= lower)         # 觸下欄
        cl_j = close[jj]

        def _assign(mask, lab, name):
            label[mask] = lab
            barrier[mask] = name
            t_event[mask] = k
            ret_event[mask] = cl_j[mask] / close[mask] - 1.0
            resolved[mask] = True

        _assign(up_hit & ~dn_hit, 4, "up")           # 只觸上 → 做多成功
        _assign(dn_hit & ~up_hit, 0, "dn")           # 只觸下 → 下跌/停損
        # 同根 K 同時觸上下欄:盤中先後不可知 → 保守歸中性 2(並標記 'tie' 供稽核)
        _assign(up_hit & dn_hit, 2, "tie")

    # 收尾:未觸界者區分「滿足整段視窗(垂直欄)」vs「資料尾端視窗不足(丟棄)」
    unresolved = (~resolved) & (~invalid)
    full_window = (idx + max_holding) < n
    vsel = unresolved & full_window                  # 撐到右欄 → 中性盤整
    jv = idx + max_holding
    label[vsel] = 2
    barrier[vsel] = "vert"
    t_event[vsel] = max_holding
    ret_event[vsel] = close[jv[vsel]] / close[vsel] - 1.0
    label[unresolved & ~full_window] = np.nan        # 視窗被截斷 → 無法判定

    out = pd.DataFrame({
        "label": label, "t_event": t_event, "ret_event": ret_event,
        "barrier": barrier, "vol": sigma, "upper": upper, "lower": lower,
    }, index=df.index)
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    out["t_event"] = out["t_event"].astype(int)
    return out


if __name__ == "__main__":
    from core.data_pipeline import seed_sample_data, build_features
    seed_sample_data("TEST")
    feat = build_features("TEST")
    lab = triple_barrier_labels(feat)
    print("樣本數(可用):", len(lab), "/", len(feat))
    print("標籤分布:\n", lab["label"].value_counts().sort_index())
    print("觸界分布:\n", lab["barrier"].value_counts())
    print("平均持有(觸界根數):", round(lab["t_event"].mean(), 1))
    print("各類平均實現報酬:\n",
          lab.groupby("label")["ret_event"].mean().round(4))
