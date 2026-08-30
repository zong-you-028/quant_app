# -*- coding: utf-8 -*-
"""
chip_lab.py - 籌碼/資金流因子與動能「結合」的嚴謹測試
把外資買賣超、融資使用率、外資期貨淨空單,與現行動能(skip10)在去偏池上對比:
  基準     : 純動能 skip10(現行)
  +外資濾網: 動能前 2K 強中,只取「外資近 20 日淨買超為正」的 K 檔
  外資綜合 : 動能排名 + 外資流量排名 各半
  +融資濾網: 動能前 2K 強中,剔除「融資使用率」最高(散戶過熱)者
  +期貨閘門: 外資台指期淨空(負)時,整體曝險打折
評估同實驗室:全期 / OOS(後40%)/ 2022 空頭,含 0.003 成本。
外資流量正規化:近 20 日外資淨買超 ÷ 近 20 日成交量(=外資佔成交的淨買比)。
"""
import numpy as np
import pandas as pd

import config
from core.signal_lab import load_closes, backtest_signal, evaluate, DELISTED
from core.chip_data import load_chip_cache
from core.data_pipeline import load_ohlcv
from core import evaluation as ev


def _volume_panel(symbols, index):
    vols = {}
    for s in symbols:
        try:
            v = load_ohlcv(s)["volume"].astype(float)
            vols[s] = v
        except Exception:
            pass
    return pd.DataFrame(vols).reindex(index)


def run():
    cache = load_chip_cache()
    if cache is None:
        raise RuntimeError("尚無 chip_cache,請先跑 python -m core.chip_data")
    flow, margin, fut = cache["flow"], cache["margin"], cache.get("fut")

    syms = list(config.UNIVERSE) + DELISTED
    close_df, ret_df = load_closes(syms)
    idx, cols = ret_df.index, ret_df.columns
    gate = close_df.pct_change(60, fill_method=None)
    mom = close_df.shift(10).pct_change(50, fill_method=None)     # 現行 skip10

    # 外資流量訊號:近20日外資淨買超 / 近20日成交量(對齊日索引)
    vol = _volume_panel(syms, idx)
    fnet = pd.DataFrame({s: flow[s] for s in flow}).reindex(idx)
    fflow = (fnet.rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan))

    # 融資使用率:餘額 / 限額(越高=散戶越擠)
    musage = {}
    for s in margin:
        m = margin[s]
        u = (m["bal"] / m["lim"].replace(0, np.nan))
        u.index = pd.to_datetime(u.index)
        musage[s] = u.reindex(idx).ffill()
    musage = pd.DataFrame(musage).reindex(columns=cols)

    # 外資期貨淨未平倉(市場層級)-> 20日均;負=偏空
    fut20 = None
    if fut is not None and len(fut):
        fut20 = pd.Series(fut).reindex(idx).ffill().rolling(20).mean()

    k = config.ROTATION_TOP_K
    cost = config.COST_PER_TURNOVER
    rebal = config.ROTATION_REBAL_DAYS

    def bt(mode):
        w = pd.DataFrame(np.nan, index=idx, columns=cols)
        for d in idx[::rebal]:
            row = mom.loc[d].dropna().sort_values(ascending=False)
            g = gate.loc[d]
            cand = [s for s in row.index if pd.notna(g.get(s)) and g.get(s) > 0]
            if mode == "foreign_filter":
                pool = cand[:2 * k]
                frow = fflow.loc[d] if d in fflow.index else pd.Series(dtype=float)
                picks = [s for s in pool if pd.notna(frow.get(s)) and frow.get(s) > 0][:k]
                if len(picks) < k:                       # 不足補原動能序
                    picks += [s for s in pool if s not in picks][:k - len(picks)]
            elif mode == "foreign_combo":
                rk_m = row.rank(pct=True)
                frow = fflow.loc[d] if d in fflow.index else pd.Series(dtype=float)
                rk_f = frow.reindex(row.index).rank(pct=True)
                comb = (rk_m.add(rk_f, fill_value=rk_m.min())).reindex(cand)
                picks = list(comb.sort_values(ascending=False).head(k).index)
            elif mode == "margin_filter":
                pool = cand[:2 * k]
                mrow = musage.loc[d] if d in musage.index else pd.Series(dtype=float)
                ranked = sorted(pool, key=lambda s: (mrow.get(s) if pd.notna(mrow.get(s)) else 0))
                picks = ranked[:k]                       # 取融資使用率最低的 k
            else:                                        # baseline
                picks = cand[:k]
            ww = pd.Series(0.0, index=cols)
            for s in picks:
                ww[s] = 1.0 / k
            w.loc[d] = ww.values
        w = w.ffill().fillna(0.0)
        port = (w.shift(1).fillna(0.0) * ret_df.fillna(0.0)).sum(axis=1)
        turn = w.diff().abs().sum(axis=1)
        turn.iloc[0] = w.iloc[0].abs().sum()
        net = port - turn * cost
        if mode == "fut_gate" and fut20 is not None:     # 期貨閘門:淨空時曝險打折
            scale = (fut20 > 0).astype(float).reindex(idx).fillna(1.0)
            scale = scale.where(scale > 0, 0.5)          # 淨空 -> 半倉
            net = net * scale.shift(1).fillna(1.0)
        return net

    tests = ["baseline", "foreign_filter", "foreign_combo", "margin_filter", "fut_gate"]
    names = {"baseline": "基準 skip10", "foreign_filter": "+外資濾網",
             "foreign_combo": "外資綜合排名", "margin_filter": "+融資濾網(低使用率)",
             "fut_gate": "+外資期貨閘門"}
    print(f"資料涵蓋:外資流量 {fflow.notna().any().sum()} 檔, 融資 {musage.notna().any().sum()} 檔, "
          f"期貨 {'有' if fut20 is not None else '無'}")
    print(f"{'設定':<20}{'CAGR':>8}{'Sharpe':>8}{'OOS':>7}{'MDD':>8}{'2022空頭':>9}")
    for m in tests:
        net = bt(m)
        r = evaluate(net)
        print(f"{names[m]:<20}{r['cagr']*100:>7.1f}%{r['sharpe']:>8.2f}{r['oos_sharpe']:>7.2f}"
              f"{r['mdd']*100:>7.1f}%{r['bear']*100:>+8.1f}%")


if __name__ == "__main__":
    run()
