# -*- coding: utf-8 -*-
"""
rotation.py - 相對強弱輪動策略(cross-sectional momentum rotation)
本專案的「主策略骨幹」。職責:
  1. 對觀察清單每檔算「動能」(N 日報酬),每隔 rebal 天把資金「等權」配置到
     動能最強的前 K 檔,其餘空手 —— 永遠押當下最強的幾檔,賺相對強弱溢酬。
  2. 回測:組合報酬 = 前一日權重 × 當日報酬;成本 = |權重變動| × 每單位換手成本。
  3. 回傳:權益曲線、CAGR、最大回撤,以及「大盤代理(等權買進持有)」對照,
     並給出「本期應持有清單(今日動能排名前 K)」與相對上期的買進/賣出/續抱差異。
為什麼用它:單檔擇時(ML/均值回歸/動能)長線打不贏單檔死抱(長多股全倉只能打平
還扣成本);輪動是正統打敗指數的方式,且每月換股≈短波段(持有約 1~2 月)。
純函式、不依賴 Flet,便於單元測試。
"""
import numpy as np
import pandas as pd

import config
from core.data_pipeline import ensure_data, get_stock_name, load_ohlcv


# ---------------------------------------------------------------------------
# 小工具:CAGR / MDD
# ---------------------------------------------------------------------------
def _cagr(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    base = float(equity.iloc[-1])
    return base ** (1.0 / years) - 1.0 if (years > 0 and base > 0) else base - 1.0


def _mdd(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _aligned_performance(net_ret: pd.Series, benchmark_ret: pd.Series,
                         lookback_days=None) -> dict:
    """Recompute strategy and benchmark KPIs on identical dates and window."""
    joined = pd.concat(
        [net_ret.rename("strategy"), benchmark_ret.rename("benchmark")], axis=1
    ).dropna()
    if joined.empty:
        raise RuntimeError("strategy and benchmark have no overlapping dates")
    if lookback_days:
        start = joined.index[-1] - pd.Timedelta(days=int(lookback_days))
        joined = joined.loc[joined.index >= start]
    strat_eq = (1.0 + joined["strategy"]).cumprod()
    bench_eq = (1.0 + joined["benchmark"]).cumprod()
    years = ((joined.index[-1] - joined.index[0]).days / 365.25
             if len(joined) > 1 else 0.0)
    return {
        "equity": strat_eq,
        "benchmark_equity": bench_eq,
        "cagr": _cagr(strat_eq),
        "benchmark_cagr": _cagr(bench_eq),
        "mdd": _mdd(strat_eq),
        "benchmark_mdd": _mdd(bench_eq),
        "years": years,
        "start": joined.index[0],
        "end": joined.index[-1],
    }


# ---------------------------------------------------------------------------
# 波動式停損停利(ATR):讓停損隨個股波動自動放寬/收緊,免得被雜訊洗掉
# ---------------------------------------------------------------------------
def atr_percent(symbol, window=None):
    """近 window 日 ATR 佔現價比例(這檔「平常一天大約動幾 %」);算不出回 None。"""
    window = window or getattr(config, "ROTATION_ATR_WINDOW", 14)
    try:
        df = load_ohlcv(symbol)
        if df is None or df.empty or not {"high", "low", "close"} <= set(df.columns):
            return None
        h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        pc = c.shift(1)
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window).mean().iloc[-1]
        v = float(atr / c.iloc[-1])
        return v if (v == v and v > 0) else None
    except Exception:
        return None


def stop_take_levels(symbol, price):
    """
    回傳 (停損價, 停利價):優先用 ATR 波動式(波動大→自動放寬);
    ATR 算不出時退回固定 %。停利倍數設 0/None 則不給停利(讓贏家續抱)。
    """
    price = float(price)
    a = atr_percent(symbol)
    sm = getattr(config, "ROTATION_STOP_ATR_MULT", 3.0)
    tm = getattr(config, "ROTATION_TAKE_ATR_MULT", 6.0)
    if a:
        stop = price * (1.0 - sm * a)
        take = price * (1.0 + tm * a) if tm else None
    else:
        stop = price * (1.0 - getattr(config, "ROTATION_STOP_PCT", 0.08))
        tp = getattr(config, "ROTATION_TAKE_PCT", 0.20)
        take = price * (1.0 + tp) if tp else None
    return stop, take


# ---------------------------------------------------------------------------
# 載入觀察清單的報酬與動能
# ---------------------------------------------------------------------------
def _load_panel(symbols, mom_days, min_obs=None, skip_days=None):
    """
    對每檔 ensure_data + load_ohlcv(只用「已還原」收盤算報酬/動能),回傳:
      ret_df  : 各檔每日報酬(欄=代號)
      mom_df  : 各檔「排名用」動能 = P[t-skip]/P[t-mom_days] - 1
                (跳過最近 skip 日的短期反轉雜訊;signal_lab 實測四種視窗皆優於不跳)
      gate_df : 各檔「閘門用」原始動能(不跳;絕對動能閘門用,與實驗設定一致)
      names   : {代號: 中文名}
    只取價格(不經 build_features),避免「籌碼資料缺漏」害整檔被丟掉 —— 廣泛 PIT 池必須。
    ★ 點位即時資格(PIT,②):各檔資料起點 = 其上市/可得日,上市前 momentum 為 NaN
      -> 換股時 dropna 自動排除(=已上市夠久才可選)。
    min_obs:收盤筆數不足者略過;單檔失敗直接略過。
    """
    min_obs = min_obs or (mom_days + 20)
    skip = getattr(config, "ROTATION_SKIP_DAYS", 0) if skip_days is None else skip_days
    skip = skip if 0 < skip < mom_days else 0
    rets, moms, gates, names = {}, {}, {}, {}
    for s in symbols:
        try:
            ensure_data(s)
            df = load_ohlcv(s)
        except Exception:
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue
        close = df["close"].astype(float).sort_index()
        if close.notna().sum() < min_obs:
            continue
        rets[s] = close.pct_change()
        gates[s] = close.pct_change(mom_days)
        moms[s] = (close.shift(skip).pct_change(mom_days - skip)
                   if skip else gates[s])
        names[s] = get_stock_name(s) or ""
    if not rets:
        raise RuntimeError("觀察清單全部載入失敗(無資料)")
    ret_df = pd.DataFrame(rets).sort_index()
    mom_df = pd.DataFrame(moms).reindex(ret_df.index)
    gate_df = pd.DataFrame(gates).reindex(ret_df.index)
    return ret_df, mom_df, gate_df, names


# ---------------------------------------------------------------------------
# 主函式:跑輪動回測 + 產生本期持有清單
# ---------------------------------------------------------------------------
def _select_picks(mom_row, gate_row, top_k, abs_mom, abs_thresh,
                  defensive=False, margin_row=None, pool_mult=2,
                  fastsell_row=None, fastsell_z=-1.5):
    """
    某一換股日的選股:回傳實際進場(已過閘門)的代號 list。
      標準:動能前 top_k,其中絕對動能 > 門檻者進場(失格者留現金)。
      防禦:動能前 top_k×pool_mult 且過閘門的候選裡,挑「融資使用率最低」的 top_k。
      急賣閘門(fastsell_row 有給時):外資持股比驟降(z < fastsell_z)者剔除
      (大戶「跑得快」= 警報;該 slot 留現金)。
    """
    def _fast_selling(s):
        if fastsell_row is None:
            return False
        z = fastsell_row.get(s)
        return pd.notna(z) and z < fastsell_z

    ranked = mom_row.dropna().sort_values(ascending=False)
    if defensive and margin_row is not None:
        cand = [s for s in ranked.index
                if ((not abs_mom) or (pd.notna(gate_row.get(s))
                                      and gate_row.get(s) > abs_thresh))
                and not _fast_selling(s)]
        pool = cand[:top_k * pool_mult]
        pool = sorted(pool, key=lambda s: (margin_row.get(s)
                      if pd.notna(margin_row.get(s)) else 9e9))  # 缺融資資料排最後
        return pool[:top_k]
    sel = []
    for s in ranked.head(top_k).index:                # 標準:前 top_k + 閘門
        g = gate_row.get(s)
        if abs_mom and (pd.isna(g) or g <= abs_thresh):
            continue
        if _fast_selling(s):
            continue                                   # 外資急賣 -> 該 slot 留現金
        sel.append(s)
    return sel


def run_rotation(symbols=None, mom_days=None, top_k=None,
                 rebal_days=None, cost_per_turnover=None,
                 abs_mom=None, abs_thresh=None, defensive=None,
                 sox_gate=None) -> dict:
    """
    執行相對強弱輪動回測,並回傳「現在該持有哪幾檔」。
    參數預設讀 config.ROTATION_*。回傳 dict(見檔末 return 註解)。
    abs_mom:啟用絕對動能閘門(選中標的若絕對動能<=abs_thresh則該檔轉現金、不進場)。
    defensive:防禦模式(融資濾網)—— 從動能前 K×pool_mult 強裡挑融資使用率最低的 K 檔。
    """
    if symbols is None:                              # 預設用廣泛 PIT 池(修存活者偏差)
        use_univ = (getattr(config, "ROTATION_USE_UNIVERSE", False)
                    and getattr(config, "UNIVERSE", None))
        symbols = config.UNIVERSE if use_univ else config.WATCHLIST
    mom_days = mom_days or config.ROTATION_MOM_DAYS
    top_k = top_k or config.ROTATION_TOP_K
    rebal_days = rebal_days or config.ROTATION_REBAL_DAYS
    cost = (config.COST_PER_TURNOVER if cost_per_turnover is None
            else cost_per_turnover)
    abs_mom = getattr(config, "ROTATION_ABS_MOM", False) if abs_mom is None else abs_mom
    abs_thresh = (getattr(config, "ROTATION_ABS_THRESH", 0.0)
                  if abs_thresh is None else abs_thresh)
    defensive = (getattr(config, "ROTATION_DEFENSIVE", False)
                 if defensive is None else defensive)
    pool_mult = getattr(config, "ROTATION_DEFENSIVE_POOL_MULT", 2)
    sox_gate = (getattr(config, "ROTATION_SOX_GATE", False)
                if sox_gate is None else sox_gate)

    ret_df, mom_df, gate_df, names = _load_panel(
        symbols, mom_days, getattr(config, "ROTATION_MIN_OBS", None))
    idx = ret_df.index
    cols = ret_df.columns

    # 防禦模式:載入融資使用率面板(無快取則退回標準模式)
    margin_panel = None
    if defensive:
        try:
            from core.chip_data import margin_usage_panel
            margin_panel = margin_usage_panel(symbols, idx)
        except Exception:
            margin_panel = None
        if margin_panel is None:
            defensive = False                         # 沒籌碼資料 -> 安全退回標準

    # 外資急賣閘門:持股比驟降 z 面板(無資料 -> None,閘門自動失效)
    fastsell_panel = None
    fs_z = getattr(config, "ROTATION_FASTSELL_Z", -1.5)
    if getattr(config, "ROTATION_FASTSELL_GATE", False):
        try:
            from core.chip_data import fastsell_z_panel
            fastsell_panel = fastsell_z_panel(symbols, idx)
        except Exception:
            fastsell_panel = None

    # 換股日(每 rebal_days 取一天);排名用跳過近期的動能、閘門用原始動能
    rebal_dates = idx[::rebal_days]
    weights = pd.DataFrame(np.nan, index=idx, columns=cols)
    selections = {}                               # 換股日 -> 當日實際持有(已過閘門)的代號
    for d in rebal_dates:
        row = mom_df.loc[d].dropna()
        if row.empty:
            continue
        grow = gate_df.loc[d]
        mrow = (margin_panel.loc[d] if (margin_panel is not None
                and d in margin_panel.index) else None)
        fsrow = (fastsell_panel.loc[d] if (fastsell_panel is not None
                 and d in fastsell_panel.index) else None)
        sel = _select_picks(row, grow, top_k, abs_mom, abs_thresh,
                            defensive=defensive, margin_row=mrow, pool_mult=pool_mult,
                            fastsell_row=fsrow, fastsell_z=fs_z)
        w = pd.Series(0.0, index=cols)
        for s in sel:
            w[s] = 1.0 / top_k                         # 失格/未選 slot 留現金
        selections[d] = sel
        weights.loc[d] = w.values
    # 換股日之間沿用上一次權重
    weights = weights.ffill().fillna(0.0)

    # 組合每日報酬(前一日權重 × 當日報酬)與換手成本
    port_ret = (weights.shift(1).fillna(0.0) * ret_df.fillna(0.0)).sum(axis=1)
    turn = weights.diff().abs().sum(axis=1)
    if len(turn):
        turn.iloc[0] = weights.iloc[0].abs().sum()
    net_ret = port_ret - turn * cost

    # 費半市場燈:RISK OFF(SOX 跌破均線)時整批轉現金(逐日,無前視)
    sox = {"ok": False, "risk_on": True}
    if sox_gate:
        try:
            from core import market_regime
            reg = market_regime.sox_regime_series(idx)
            switch = reg.diff().abs().fillna(0.0) * cost   # 進出現金的換手成本
            net_ret = net_ret * reg - switch
            sox = market_regime.sox_status()
        except Exception:
            sox = {"ok": False, "risk_on": True}
    full_equity = (1.0 + net_ret).cumprod()

    # 研究用大盤代理:等權買進持有全部可用標的(不作 UI 正式基準)
    mkt_ret = ret_df.fillna(0.0).mean(axis=1)
    mkt_equity = (1.0 + mkt_ret).cumprod()

    # 正式比較:策略與 006208 使用完全相同的日期及最近 N 天視窗。
    benchmark_symbol = getattr(config, "BENCHMARK_SYMBOL", "006208")
    benchmark_label = benchmark_symbol
    try:
        ensure_data(benchmark_symbol)
        benchmark_df = load_ohlcv(benchmark_symbol)
        benchmark_ret = benchmark_df["close"].astype(float).sort_index().pct_change()
        perf = _aligned_performance(
            net_ret, benchmark_ret, getattr(config, "EVAL_LOOKBACK_DAYS", None)
        )
        full_perf = _aligned_performance(net_ret, benchmark_ret, None)
    except Exception:
        benchmark_label = "等權股票池"
        perf = _aligned_performance(
            net_ret, mkt_ret, getattr(config, "EVAL_LOOKBACK_DAYS", None)
        )
        full_perf = _aligned_performance(net_ret, mkt_ret, None)

    # --- 本期應持有清單:以「最新一日」動能排名(今天要押的就是這幾檔)---
    last_mom = mom_df.iloc[-1].dropna().sort_values(ascending=False)
    last_gate = gate_df.iloc[-1]
    last_margin = (margin_panel.iloc[-1] if (margin_panel is not None
                   and len(margin_panel)) else None)
    last_fs = (fastsell_panel.iloc[-1] if (fastsell_panel is not None
               and len(fastsell_panel)) else None)
    ranking = [(s, names.get(s, ""), float(last_mom[s])) for s in last_mom.index]
    held = _select_picks(last_mom, last_gate, top_k, abs_mom, abs_thresh,
                        defensive=defensive, margin_row=last_margin, pool_mult=pool_mult,
                        fastsell_row=last_fs, fastsell_z=fs_z)
    # 本期被「外資急賣」踢掉的標的(供 UI 標示原因)
    fastsell_symbols = ([s for s in last_mom.index
                         if pd.notna(last_fs.get(s)) and last_fs.get(s) < fs_z]
                        if last_fs is not None else [])
    candidates = ([s for s, _, _ in ranking[:top_k]]
                  if not defensive else list(held))
    cash_symbols = [s for s in candidates if s not in held]
    holdings = list(held)                              # 對外只提供通過全部閘門者
    # 市場燈 RISK OFF -> 本期整批轉現金(holdings 留作「站回時的口袋名單」)
    if sox_gate and sox.get("ok") and not sox.get("risk_on", True):
        cash_symbols = list(candidates)
        held = []
        holdings = []

    # 與「上一個換股日」的實際持有相比,算出買進/賣出/續抱(供操作建議)
    sel_dates = sorted(selections.keys())
    prev_holdings = selections[sel_dates[-1]] if sel_dates else []
    buys = [s for s in held if s not in prev_holdings]
    sells = [s for s in prev_holdings if s not in held]
    holds = [s for s in held if s in prev_holdings]

    return {
        "equity": perf["equity"],
        "market_equity": perf["benchmark_equity"],
        "cagr": perf["cagr"],
        "mdd": perf["mdd"],
        "market_cagr": perf["benchmark_cagr"],
        "market_mdd": perf["benchmark_mdd"],
        "benchmark": benchmark_label,
        "full_equity": full_equity,
        "universe_market_equity": mkt_equity,
        "full_cagr": full_perf["cagr"],
        "full_market_cagr": full_perf["benchmark_cagr"],
        "full_mdd": full_perf["mdd"],
        "full_market_mdd": full_perf["benchmark_mdd"],
        "full_start": full_perf["start"].strftime("%Y-%m-%d"),
        "full_end": full_perf["end"].strftime("%Y-%m-%d"),
        "turnover": float(turn.sum()),    # 總換手(倍)
        "years": perf["years"],
        "eval_start": perf["start"].strftime("%Y-%m-%d"),
        "eval_end": perf["end"].strftime("%Y-%m-%d"),
        "ranking": ranking,               # [(代號, 名稱, 動能)] 由強至弱(全部)
        "holdings": holdings,             # 通過全部閘門、可實際進場的標的
        "candidates": candidates,         # 閘門前候選名單(僅供診斷)
        "held": held,                     # 過絕對動能閘門、真的進場的標的
        "cash_symbols": cash_symbols,     # 前 K 中絕對動能翻負 -> 轉現金者
        "abs_mom": abs_mom, "abs_thresh": abs_thresh,  # 閘門設定(供 UI 標示)
        "defensive": defensive,           # 是否為防禦模式(融資濾網)
        "sox": sox,                       # 費半市場燈狀態(ok/risk_on/close/ma/asof)
        "fastsell_symbols": fastsell_symbols,  # 外資急賣中(被閘門剔除)的標的
        "names": names,                   # {代號: 名稱}
        "prev_holdings": prev_holdings,   # 上一換股日實際持有
        "buys": buys, "sells": sells, "holds": holds,  # 相對上期的異動
        "mom_days": mom_days, "top_k": top_k, "rebal_days": rebal_days,
        "last_date": idx[-1].strftime("%Y-%m-%d") if len(idx) else "",
    }


def analyze_stock(symbol, symbols=None, mom_days=None, top_k=None,
                  abs_thresh=None) -> dict:
    """
    「這檔符不符合輪動策略」的檢查(取代沒 edge 的 ML 燈號)。
    回傳:此檔在輪動池的動能排名、絕對動能、是否會被選/被閘門擋成現金,
          以及價格/風險(波動、近一年回撤)與判定文字。純函式、可測試。
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("代號不可空白")
    mom_days = mom_days or config.ROTATION_MOM_DAYS
    top_k = top_k or config.ROTATION_TOP_K
    abs_thresh = (getattr(config, "ROTATION_ABS_THRESH", 0.0)
                  if abs_thresh is None else abs_thresh)
    if symbols is None:
        use_univ = (getattr(config, "ROTATION_USE_UNIVERSE", False)
                    and getattr(config, "UNIVERSE", None))
        symbols = list(config.UNIVERSE) if use_univ else list(config.WATCHLIST)

    pool = list(dict.fromkeys(list(symbols) + [symbol]))   # 確保 target 在池內可排名
    ret_df, mom_df, gate_df, names = _load_panel(
        pool, mom_days, getattr(config, "ROTATION_MIN_OBS", None))
    if symbol not in mom_df.columns:
        raise RuntimeError(f"{symbol} 資料不足(上市太短或抓取失敗),無法分析")

    valid = mom_df.dropna(subset=[symbol])
    if valid.empty:
        raise RuntimeError(f"{symbol} 動能無法計算(歷史不足 {mom_days} 日)")
    asof = valid.index[-1]
    row = mom_df.loc[asof].dropna()
    ranked = row.sort_values(ascending=False)
    rank = int(list(ranked.index).index(symbol) + 1)
    n = int(len(ranked))
    mom = float(row[symbol])                      # 排名用動能(跳過近期)
    g = gate_df.loc[asof].get(symbol)             # 閘門用原始動能
    passes_abs = bool(pd.notna(g) and float(g) > abs_thresh)
    in_top_k = rank <= top_k
    selected = in_top_k and passes_abs

    # 個股價格/風險(用自己的還原收盤)
    df = load_ohlcv(symbol)
    close = df["close"].astype(float).sort_index()
    ret = close.pct_change()
    last_close = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last_close
    day_chg = last_close - prev

    def _mr(k):
        return float(close.iloc[-1] / close.iloc[-1 - k] - 1) if len(close) > k else float("nan")

    sd = ret.tail(60).std()
    vol_ann = float(sd * (252 ** 0.5)) if sd == sd else 0.0
    eq = (1.0 + ret.fillna(0.0)).cumprod()
    eqw = eq[eq.index >= (eq.index[-1] - pd.Timedelta(days=365))]
    mdd = float((eqw / eqw.cummax() - 1.0).min()) if len(eqw) else 0.0

    # 判定(台股慣例:符合/強=紅、弱=綠、轉現金=綠)
    if selected:
        vshort, color = "符合策略", "#D32F2F"
        note = f"輪動這期會選它:動能前 {top_k} 強(第 {rank}/{n})且絕對動能為正"
    elif in_top_k and not passes_abs:
        vshort, color = "會被擋成現金", "#2E7D32"
        note = f"相對排名前面(第 {rank}/{n}),但絕對動能翻負 → 閘門讓它轉現金、不進場"
    elif passes_abs and not in_top_k:
        vshort, color = "強度不足", "#9E9E9E"
        note = f"絕對動能為正,但沒進前 {top_k} 強(第 {rank}/{n})→ 輪動不會選"
    else:
        vshort, color = "不符合", "#2E7D32"
        note = f"動能偏弱(第 {rank}/{n})→ 輪動不會選它"

    return {
        "symbol": symbol, "name": names.get(symbol, "") or "",
        "asof": asof.strftime("%Y-%m-%d"),
        "rank": rank, "n": n, "mom": mom, "mom5": _mr(5), "mom20": _mr(20),
        "in_top_k": in_top_k, "passes_abs": passes_abs, "selected": selected,
        "top_k": top_k, "mom_days": mom_days,
        "last_close": last_close, "day_change": day_chg,
        "day_change_pct": (day_chg / prev) if prev else 0.0,
        "vol_annual": vol_ann, "mdd": mdd,
        "price5": close.tail(5),
        "price6": close[close.index >= (close.index[-1] - pd.Timedelta(days=180))],
        "verdict_short": vshort, "verdict_color": color, "note": note,
    }


if __name__ == "__main__":
    r = run_rotation()
    print(f"資料到 {r['last_date']} · 動能{r['mom_days']}日 · 持有前{r['top_k']}強 · 每{r['rebal_days']}日換股")
    print(f"策略 CAGR {r['cagr']*100:.1f}%  MDD {r['mdd']*100:.1f}%"
          f"  |  大盤代理 CAGR {r['market_cagr']*100:.1f}%  MDD {r['market_mdd']*100:.1f}%")
    print(f"絕對動能閘門:{'開' if r['abs_mom'] else '關'}(門檻 {r['abs_thresh']:+.0%})")
    print("本期應持有(前 K 強):")
    for s, nm, mv in r["ranking"][:r["top_k"]]:
        flag = "→ 轉現金(絕對動能翻負)" if s in r["cash_symbols"] else "→ 進場"
        print(f"  {nm} {s}  動能 {mv*100:+.1f}%  {flag}")
    print("實際進場:", r["held"], " 轉現金:", r["cash_symbols"])
    print("買進:", r["buys"], " 賣出:", r["sells"], " 續抱:", r["holds"])
