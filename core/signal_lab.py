# -*- coding: utf-8 -*-
"""
signal_lab.py - 輪動「選股訊號」實驗室(改輸入指標找更好的 edge)
背景:個股 ML 擇時已證明無 OOS edge(WFA 虧錢),改善方向應放在「輪動的選股訊號」。
本檔把幾個文獻上有依據的動能變體,在「去偏池(50 檔 + 6 檔下市股)」上,
用同一套機制(每 20 日選前 K 等權、絕對動能閘門、扣 0.003 成本)公平對比:

  raw60    : 60 日報酬(現行基準)
  skip10   : 跳過最近 10 日的動能 P[t-10]/P[t-60]-1
             (文獻:短期反轉,最近幾日的漲跌是雜訊 —— 也正對「老是推剛大跌的股」)
  riskadj  : 60 日報酬 / 60 日波動(風險調整動能;偏好「漲得穩」勝過「漲得猛」,
             文獻上可降動能崩盤)
  skip+ra  : 跳過 10 日 + 風險調整(兩者合併,文獻最愛的組合)
  high52   : 距 52 週高點距離(George & Hwang 2004;越接近新高越強)
  combo    : 60 日與 120 日動能「排名平均」(多視窗混合,降單一視窗運氣)
  invvol   : 選股同 raw60,但改「波動倒數加權」(波動小的多買,取代等權)

評估:全期 CAGR/MDD/Sharpe + OOS(後 40%)Sharpe/CAGR + 2022 空頭報酬/回撤。
閘門統一用「raw 60 日動能 > 0」以隔離變數(只比較『選股訊號』本身)。
誠實註記:下市股下市後報酬以 0 計(凍結),對所有變體一視同仁,不影響相互比較。
"""
import numpy as np
import pandas as pd

import config
from core.data_pipeline import ensure_data, load_ohlcv
from core import evaluation as ev

# 期間內下市的地雷股(已在 DB):康友KY/樂陞/綠悅KY/興航/誠創/陽光能源
DELISTED = ["6452", "3662", "1262", "6702", "3536", "9157"]


def load_closes(symbols, min_obs=80):
    """載入各檔還原收盤,回傳 (close_df, ret_df);資料不足/失敗者略過。"""
    closes = {}
    for s in symbols:
        try:
            ensure_data(s)
            df = load_ohlcv(s)
        except Exception:
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue
        c = df["close"].astype(float).sort_index()
        if c.notna().sum() >= min_obs:
            closes[s] = c
    if not closes:
        raise RuntimeError("無可用資料")
    close_df = pd.DataFrame(closes).sort_index()
    ret_df = close_df.pct_change(fill_method=None)
    return close_df, ret_df


# ---------------------------------------------------------------------------
# 候選訊號(回傳與 close_df 同形狀的「越大越強」分數表)
# ---------------------------------------------------------------------------
def sig_raw60(close_df, ret_df):
    return close_df.pct_change(60, fill_method=None)


def sig_skip10(close_df, ret_df):
    # P[t-10]/P[t-60] - 1:同樣回看 60 日,但跳過最近 10 日(短期反轉雜訊)
    return close_df.shift(10).pct_change(50, fill_method=None)


def sig_riskadj(close_df, ret_df):
    mom = close_df.pct_change(60, fill_method=None)
    vol = ret_df.rolling(60).std().replace(0.0, np.nan)
    return mom / vol


def sig_skip_ra(close_df, ret_df):
    mom = close_df.shift(10).pct_change(50, fill_method=None)
    vol = ret_df.rolling(60).std().replace(0.0, np.nan)
    return mom / vol


def sig_high52(close_df, ret_df):
    return close_df / close_df.rolling(252).max() - 1.0   # 0=創新高,越負越遠


def sig_combo(close_df, ret_df):
    r60 = close_df.pct_change(60, fill_method=None).rank(axis=1, pct=True)
    r120 = close_df.pct_change(120, fill_method=None).rank(axis=1, pct=True)
    return (r60 + r120) / 2.0


SIGNALS = {
    "raw60(基準)": (sig_raw60, False),
    "skip10": (sig_skip10, False),
    "riskadj": (sig_riskadj, False),
    "skip+ra": (sig_skip_ra, False),
    "high52": (sig_high52, False),
    "combo60/120": (sig_combo, False),
    "invvol加權": (sig_raw60, True),     # 選股同基準,只換加權方式
}


# ---------------------------------------------------------------------------
# 統一回測機制(只有「訊號 / 加權」是變數)
# ---------------------------------------------------------------------------
def backtest_signal(signal_df, gate_df, ret_df, top_k=None, rebal=None,
                    cost=None, inv_vol_df=None):
    """每 rebal 日依 signal 選前 K;gate<=0 的 slot 轉現金;預設等權(或波動倒數加權)。"""
    top_k = top_k or config.ROTATION_TOP_K
    rebal = rebal or config.ROTATION_REBAL_DAYS
    cost = config.COST_PER_TURNOVER if cost is None else cost
    idx, cols = ret_df.index, ret_df.columns
    weights = pd.DataFrame(np.nan, index=idx, columns=cols)
    for d in idx[::rebal]:
        row = signal_df.loc[d].dropna()
        if row.empty:
            continue
        top = row.nlargest(top_k)
        gates = gate_df.loc[d]
        passing = [s for s in top.index
                   if pd.notna(gates.get(s)) and gates.get(s) > 0]
        w = pd.Series(0.0, index=cols)
        if inv_vol_df is not None and passing:
            iv = {}
            vrow = inv_vol_df.loc[d]
            for s in passing:
                v = vrow.get(s)
                if pd.notna(v) and v > 0:
                    iv[s] = 1.0 / v
            tot = sum(iv.values())
            if tot > 0:
                scale = len(iv) / top_k          # 現金比例與等權版一致
                for s, v in iv.items():
                    w[s] = v / tot * scale
        else:
            for s in passing:
                w[s] = 1.0 / top_k
        weights.loc[d] = w.values
    weights = weights.ffill().fillna(0.0)
    port = (weights.shift(1).fillna(0.0) * ret_df.fillna(0.0)).sum(axis=1)
    turn = weights.diff().abs().sum(axis=1)
    if len(turn):
        turn.iloc[0] = weights.iloc[0].abs().sum()
    return port - turn * cost


def evaluate(net, is_frac=0.6, bear=("2021-12-01", "2022-10-31")):
    """全期 / OOS(後 40%) / 2022 空頭 三組指標。"""
    eq = (1.0 + net.fillna(0.0)).cumprod()
    split = int(len(net) * is_frac)
    oos = net.iloc[split:]
    w = net[(net.index >= pd.Timestamp(bear[0])) & (net.index <= pd.Timestamp(bear[1]))]
    beq = (1.0 + w.fillna(0.0)).cumprod()
    return {
        "cagr": ev.annualized_return(net), "mdd": ev.max_drawdown(eq),
        "sharpe": ev.annualized_sharpe(net),
        "oos_sharpe": ev.annualized_sharpe(oos), "oos_cagr": ev.annualized_return(oos),
        "bear": float(beq.iloc[-1] - 1.0) if len(beq) else 0.0,
        "bear_mdd": ev.max_drawdown(beq),
    }


def run_lab(symbols=None, verbose=True):
    symbols = symbols or (list(config.UNIVERSE) + DELISTED)
    close_df, ret_df = load_closes(symbols, getattr(config, "ROTATION_MIN_OBS", 80))
    gate_df = close_df.pct_change(60, fill_method=None)       # 閘門統一:raw mom60 > 0
    vol_df = ret_df.rolling(60).std()
    out = {}
    if verbose:
        print(f"去偏池 {close_df.shape[1]} 檔  {close_df.index[0].date()}~{close_df.index[-1].date()}"
              f"  k={config.ROTATION_TOP_K} rebal={config.ROTATION_REBAL_DAYS} 閘門=raw60>0")
        print(f"{'訊號':<14}{'全期CAGR':>9}{'MDD':>8}{'Sharpe':>8}"
              f"{'OOS Sharpe':>11}{'OOS CAGR':>10}{'2022空頭':>10}{'空頭MDD':>9}")
    for name, (fn, use_iv) in SIGNALS.items():
        sig = fn(close_df, ret_df)
        net = backtest_signal(sig, gate_df, ret_df,
                              inv_vol_df=(vol_df if use_iv else None))
        m = evaluate(net)
        out[name] = m
        if verbose:
            print(f"{name:<14}{m['cagr']*100:>8.1f}%{m['mdd']*100:>7.1f}%{m['sharpe']:>8.2f}"
                  f"{m['oos_sharpe']:>11.2f}{m['oos_cagr']*100:>9.1f}%"
                  f"{m['bear']*100:>+9.1f}%{m['bear_mdd']*100:>8.1f}%")
    return out


if __name__ == "__main__":
    run_lab()
