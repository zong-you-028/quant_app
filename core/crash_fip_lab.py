# -*- coding: utf-8 -*-
"""
crash_fip_lab.py - 文獻候選驗證:
① Barroso & Santa-Clara 組合層波動目標(vol targeting):
   scale_t = clip(目標年化波動 / 策略近 60 日已實現年化波動, 0, 1)
   只降不升(不開槓桿);曝險變動以 |Δscale|×成本 近似扣費。
   文獻宣稱:幾乎消滅動能崩盤、Sharpe 接近翻倍。
② Frog-in-the-Pan 持續性(Da, Gurun & Warachka 2014;台股版 PBFJ 2023):
   資訊離散度 ID = sign(動能) × (負報酬天數% − 正報酬天數%),越負=漲得越「連續」。
   變體A:動能前 2K 強中挑「最連續」的 K 檔(同樣漲 100%,挑天天小漲的,
          不挑靠幾根大陽線跳上來的)。
   變體B:動能排名 + 連續性排名 的綜合分數。
評估與 signal_lab 同一套:去偏池(50+6 下市)、OOS(後40%)、2022 空頭、含成本。
"""
import numpy as np
import pandas as pd

import config
from core.signal_lab import load_closes, backtest_signal, evaluate, DELISTED


def vol_target(net: pd.Series, target: float = 0.15, window: int = 60,
               cost: float = None, cap: float = 1.0) -> pd.Series:
    """組合層波動目標:把策略日報酬乘上 scale(昨日已知,無前視),扣曝險變動成本。"""
    cost = config.COST_PER_TURNOVER if cost is None else cost
    rv = net.rolling(window).std() * np.sqrt(252)        # 近 window 日年化波動
    scale = (target / rv).clip(upper=cap)
    scale = scale.fillna(1.0)
    extra = scale.diff().abs().fillna(0.0) * cost        # 曝險調整的近似成本
    return scale.shift(1).fillna(1.0) * net - extra


def run():
    close_df, ret_df = load_closes(list(config.UNIVERSE) + DELISTED)
    gate = close_df.pct_change(60, fill_method=None)          # 閘門:原始動能>0
    sig = close_df.shift(10).pct_change(50, fill_method=None)  # 現行 skip10
    k = config.ROTATION_TOP_K

    rows = [("基準(現行skip10)", evaluate(backtest_signal(sig, gate, ret_df)))]
    base = backtest_signal(sig, gate, ret_df)

    # ① 波動目標(三種目標水準)
    for tgt in (0.10, 0.15, 0.20):
        rows.append((f"+波動目標{int(tgt*100)}%",
                     evaluate(vol_target(base, target=tgt))))

    # ② FIP 持續性
    pos_f = (ret_df > 0).rolling(60).mean()
    neg_f = (ret_df < 0).rolling(60).mean()
    ID = np.sign(gate) * (neg_f - pos_f)     # 越負 = 越連續(對正動能股)
    cont = -ID                                # 越大 = 越連續
    # 變體A:動能前 2K 中挑最連續的 K
    top2k = sig.rank(axis=1, ascending=False) <= 2 * k
    fip_sig = cont.where(top2k)
    fipA = backtest_signal(fip_sig, gate, ret_df)
    rows.append(("FIP過濾(前16取連續8)", evaluate(fipA)))
    # 變體B:綜合排名(動能 + 連續性)
    comb = sig.rank(axis=1, pct=True) + cont.rank(axis=1, pct=True)
    rows.append(("FIP綜合排名", evaluate(backtest_signal(comb, gate, ret_df))))
    # 變體C:A + 波動目標 15%
    rows.append(("FIP過濾+波動目標15%", evaluate(vol_target(fipA, 0.15))))

    print(f"去偏池 {close_df.shape[1]} 檔  k={k}  rebal={config.ROTATION_REBAL_DAYS}  含成本")
    print(f"{'策略':<20}{'CAGR':>8}{'Sharpe':>8}{'OOS Shp':>9}{'MDD':>8}"
          f"{'2022空頭':>10}{'空頭MDD':>9}")
    for name, m in rows:
        print(f"{name:<20}{m['cagr']*100:>7.1f}%{m['sharpe']:>8.2f}"
              f"{m['oos_sharpe']:>9.2f}{m['mdd']*100:>7.1f}%"
              f"{m['bear']*100:>+9.1f}%{m['bear_mdd']*100:>8.1f}%")
    return rows


if __name__ == "__main__":
    run()
