# -*- coding: utf-8 -*-
"""
rotation.py - 相對強弱輪動策略(cross-sectional momentum rotation)
本專案的「主策略骨幹」。職責:
  1. 對觀察清單每檔算「動能」(N 日報酬),每隔 rebal 天把資金「等權」配置到
     動能最強的前 K 檔,其餘空手 —— 永遠押當下最強的幾檔,賺相對強弱溢酬。
  2. 回測:訊號延遲至開盤成交，持股跨日計價，成交當日扣換手成本。
  3. 回傳:權益曲線、CAGR、最大回撤,以及「大盤代理(等權買進持有)」對照,
     並給出「本期應持有清單(最近排定選股日的前 K)」與相對上期的買進/賣出/續抱差異。
為什麼用它:單檔擇時(ML/均值回歸/動能)長線打不贏單檔死抱(長多股全倉只能打平
還扣成本);輪動是正統打敗指數的方式,且每月換股≈短波段(持有約 1~2 月)。
純函式、不依賴 Flet,便於單元測試。
"""
import numpy as np
import pandas as pd

import config
from core.data_pipeline import ensure_data, get_stock_name, load_ohlcv
from core.execution import backtest_open_execution
from core.strategy_models import build_rank_scores, data_quality, model_spec, validation_status


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
    return float((equity / equity.cummax().clip(lower=1.0) - 1.0).min())


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
                 sox_gate=None, score_mode=None) -> dict:
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
    score_mode, spec = model_spec(score_mode)
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
    prices = {s: load_ohlcv(s) for s in ret_df.columns}
    if score_mode != "production":
        benchmark_prices = load_ohlcv(config.BENCHMARK_SYMBOL)
        if benchmark_prices is None or benchmark_prices.empty:
            raise RuntimeError("新模型需要 006208 的真實行情作共同交易日曆；請先更新資料。")
        calendar = pd.DatetimeIndex(benchmark_prices["close"].dropna().index).sort_values()
        closes = pd.DataFrame({s: p["close"] for s, p in prices.items()}).reindex(calendar)
        ret_df = closes.pct_change(fill_method=None)
        mom_df = build_rank_scores(closes, score_mode)
        gate_df = closes.pct_change(mom_days, fill_method=None)
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
    selection_candidates = {}
    for d in rebal_dates:
        row = mom_df.loc[d].dropna()
        if row.empty:
            continue
        grow = gate_df.loc[d]
        mrow = (margin_panel.loc[d] if (margin_panel is not None
                and d in margin_panel.index) else None)
        fsrow = (fastsell_panel.loc[d] if (fastsell_panel is not None
                 and d in fastsell_panel.index) else None)
        if score_mode == "buffered_momentum" and not defensive:
            ranked = row.sort_values(ascending=False, kind="stable")
            previous = list(selections[next(reversed(selections))]) if selections else []
            retained = [s for s in ranked.head(2 * top_k).index if s in previous]
            picks = (retained + [s for s in ranked.index if s not in retained])[:top_k]
            row = pd.Series(np.arange(len(picks), 0, -1, dtype=float), index=picks)
        selection_candidates[d] = list(row.sort_values(ascending=False).head(top_k).index)
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

    # Targets are formed on data dates; all trades execute at delayed opens.
    sox = {"ok": False, "risk_on": True}
    if sox_gate:
        try:
            from core import market_regime
            reg = market_regime.sox_regime_series(idx, lag=0)
            weights = weights.mul(reg, axis=0)
            sox = market_regime.sox_status(asof=idx[-1])
            sox["risk_on"] = bool(reg.iloc[-1])
            sox["signal_asof"] = idx[-1].strftime("%Y-%m-%d")
        except Exception:
            sox = {"ok": False, "risk_on": True}
    lag = config.ROTATION_EXECUTION_LAG
    opens = pd.DataFrame({s: p["open"] for s, p in prices.items()})
    closes = pd.DataFrame({s: p["close"] for s, p in prices.items()})
    net_ret, turn = backtest_open_execution(
        weights, opens, closes, lag=lag, cost=cost,
        rebalance=pd.Series(idx.isin(list(selections)), index=idx))
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

    # Retain the latest scheduled selection between rebalance dates.
    sel_dates = sorted(selections)
    signal_date = sel_dates[-1] if sel_dates else idx[-1]
    last_mom = mom_df.loc[signal_date].dropna().sort_values(ascending=False)
    last_gate = gate_df.loc[signal_date]
    last_fs = fastsell_panel.loc[signal_date] if fastsell_panel is not None else None
    ranking = [(s, names.get(s, ""), float(last_mom[s])) for s in last_mom.index]
    held = list(selections.get(signal_date, []))
    # 本期被「外資急賣」踢掉的標的(供 UI 標示原因)
    fastsell_symbols = ([s for s in last_mom.index
                         if pd.notna(last_fs.get(s)) and last_fs.get(s) < fs_z]
                        if last_fs is not None else [])
    candidates = ([s for s, _, _ in ranking[:top_k]]
                  if not defensive else list(held))
    if score_mode == "buffered_momentum":
        # Raw top-K ranks can differ from retained portfolio members.
        candidates = list(selection_candidates.get(signal_date, []))
    cash_symbols = [s for s in candidates if s not in held]
    holdings = list(held)                              # 對外只提供通過全部閘門者
    # 市場燈 RISK OFF -> 本期整批轉現金(holdings 留作「站回時的口袋名單」)
    if sox_gate and sox.get("ok") and not sox.get("risk_on", True):
        cash_symbols = list(candidates)
        held = []
        holdings = []

    # 與「上一個換股日」的實際持有相比,算出買進/賣出/續抱(供操作建議)
    sel_dates = sorted(selections.keys())
    prev_holdings = selections[sel_dates[-2]] if len(sel_dates) > 1 else []
    selection_pos = idx.get_loc(signal_date)
    execution_pos = selection_pos + lag
    selection_execution = (idx[execution_pos].strftime("%Y-%m-%d")
                           if execution_pos < len(idx) else None)
    changes = weights.diff().abs().sum(axis=1).gt(1e-12)
    if len(changes):
        changes.iloc[0] = weights.iloc[0].abs().sum() > 0
    changed_dates = idx[changes.to_numpy()]
    target_change_date = changed_dates[-1] if len(changed_dates) else signal_date
    target_execution_pos = idx.get_loc(target_change_date) + lag
    target_execution_date = (idx[target_execution_pos].strftime("%Y-%m-%d")
                             if target_execution_pos < len(idx) else None)
    quality_prices = dict(prices)
    # Ranking warmup and quote freshness are different. A new listing with six
    # current bars must not make every tradable position appear stale, while a
    # genuinely absent or old cache still blocks recording actions.
    for symbol in symbols:
        if symbol not in quality_prices:
            try:
                quality_prices[symbol] = load_ohlcv(symbol)
            except Exception:
                pass
    if "benchmark_df" in locals() and benchmark_df is not None:
        quality_prices[benchmark_symbol] = benchmark_df
    quality = data_quality(symbols, prices=quality_prices)
    quality["signal_asof"] = idx[-1].strftime("%Y-%m-%d")
    if idx[-1] < pd.Timestamp(quality["target_date"]):
        quality["stale"] = True
        quality["reasons"].append("模型交易日曆未達資料目標")
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
        "score_mode": score_mode, "model_label": spec["label"],
        "model_description": spec["description"], "score_kind": spec["score_kind"],
        "validation_status": validation_status(score_mode),
        "data_quality": quality,
        "cost_per_turnover": cost,
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
        "selection_date": signal_date.strftime("%Y-%m-%d"),
        "selection_execution_date": selection_execution,
        "selection_execution_pending": selection_execution is None,
        "target_change_date": target_change_date.strftime("%Y-%m-%d"),
        "target_execution_date": target_execution_date,
        "target_execution_pending": target_execution_date is None,
        "target_asof": idx[-1].strftime("%Y-%m-%d"),
        "execution_lag": lag,
        "execution_basis": f"資料日後第 {lag} 個交易日開盤；名單按換股週期沿用",
        "net_returns": net_ret,
        "target_weights": weights,
        "model_current_weights": dict(net_ret.attrs["effective_target_weights"]),
        "model_current_holdings": list(net_ret.attrs["effective_target_weights"]),
        "model_current_asof": idx[-1].strftime("%Y-%m-%d"),
        "model_current_execution_date": net_ret.attrs["last_execution_date"],
        "model_execution_deferred": net_ret.attrs["execution_deferred"],
        "deferred_trade_days": len(net_ret.attrs.get("deferred_dates", [])),
        "last_date": idx[-1].strftime("%Y-%m-%d") if len(idx) else "",
    }


def analyze_stock(symbol, symbols=None, mom_days=None, top_k=None,
                  abs_thresh=None, score_mode=None) -> dict:
    """
    「這檔符不符合輪動策略」的檢查(取代沒 edge 的 ML 燈號)。
    回傳:此檔在輪動池的動能排名、絕對動能、是否會被選/被閘門擋成現金,
          以及價格/風險(波動、近一年回撤)與判定文字。純函式、可測試。
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("代號不可空白")
    mom_days = mom_days or config.ROTATION_MOM_DAYS
    score_mode, spec = model_spec(score_mode)
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
    prices = {s: load_ohlcv(s) for s in ret_df.columns}
    original_momentum = mom_df.copy()
    if score_mode != "production":
        benchmark_prices = load_ohlcv(config.BENCHMARK_SYMBOL)
        if benchmark_prices is None or benchmark_prices.empty:
            raise RuntimeError("新模型需要 006208 的真實行情；請先更新資料。")
        calendar = pd.DatetimeIndex(benchmark_prices["close"].dropna().index).sort_values()
        closes = pd.DataFrame({s: p["close"] for s, p in prices.items()}).reindex(calendar)
        mom_df = build_rank_scores(closes, score_mode)
        gate_df = closes.pct_change(mom_days, fill_method=None)
        original_momentum = build_rank_scores(closes, "production", mom_days=mom_days)
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
    score = float(row[symbol])
    mom = float(original_momentum.loc[asof, symbol])
    g = gate_df.loc[asof].get(symbol)             # 閘門用原始動能
    passes_abs = bool(pd.notna(g) and float(g) > abs_thresh)
    in_top_k = rank <= top_k
    selected = in_top_k and (passes_abs or not getattr(config, "ROTATION_ABS_MOM", False))
    passes_fastsell, passes_sox = True, True
    if getattr(config, "ROTATION_FASTSELL_GATE", False):
        try:
            from core.chip_data import fastsell_z_panel
            fastsell = fastsell_z_panel(pool, mom_df.index)
            z = fastsell.loc[asof].get(symbol) if fastsell is not None else None
            passes_fastsell = not (pd.notna(z) and z < config.ROTATION_FASTSELL_Z)
        except Exception:
            pass
    if getattr(config, "ROTATION_SOX_GATE", False):
        from core.market_regime import sox_regime_series
        passes_sox = bool(sox_regime_series(mom_df.index, lag=0).loc[asof])
    selected = selected and passes_fastsell and passes_sox
    scheduled_selection = None
    if score_mode == "buffered_momentum":
        scheduled_selection = run_rotation(
            symbols=symbols, mom_days=mom_days, top_k=top_k,
            abs_thresh=abs_thresh, score_mode=score_mode)
        selected = symbol in scheduled_selection["held"]

    # 個股價格/風險(用自己的還原收盤)
    df = prices[symbol]
    close = df["close"].astype(float).sort_index().loc[:asof].dropna()
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
    if not passes_sox:
        vshort, color = "市場閘門轉現金", "#2E7D32"
        note = "SOX 跌破市場均線，當日目標轉現金；仍須依資料延遲於後續開盤執行。"
    elif not passes_fastsell:
        vshort, color = "外資急賣擋成現金", "#2E7D32"
        note = "此外資持股變化觸發急賣閘門，目前不納入新倉。"
    elif selected:
        vshort, color = "符合策略", "#D32F2F"
        note = f"目前排名前 {top_k} 強(第 {rank}/{n})且通過已啟用的閘門。"
    elif in_top_k and not passes_abs:
        vshort, color = "會被擋成現金", "#2E7D32"
        note = f"相對排名前面(第 {rank}/{n}),但絕對動能翻負 → 閘門讓它轉現金、不進場"
    elif passes_abs and not in_top_k:
        vshort, color = "強度不足", "#9E9E9E"
        note = f"絕對動能為正，但目前沒進前 {top_k} 強(第 {rank}/{n})。"
    else:
        vshort, color = "不符合", "#2E7D32"
        note = f"目前動能偏弱(第 {rank}/{n})。"

    if score_mode != "production":
        note = note.replace("動能前", "模型排名前").replace("相對排名前面", "模型排名前面")
    if score_mode != "buffered_momentum":
        note += "此處為目前條件檢查；實際排定名單與成交日期請參考策略輪動頁。"
    if score_mode == "buffered_momentum":
        vshort = "本期模型入選" if selected else "本期未入選"
        color = "#D32F2F" if selected else "#9E9E9E"
        note = (f"{spec['label']}：依 {scheduled_selection['selection_date']} 排定名單，"
                + ("本期目標包含此股。" if selected else "本期目標不包含此股。")
                + "上期模型名單可保留至前 2K 名，與輪動頁一致；顯示目標不代表已成交。")
    quality_prices = dict(prices)
    quality_prices[config.BENCHMARK_SYMBOL] = load_ohlcv(config.BENCHMARK_SYMBOL)
    quality = data_quality(pool, prices=quality_prices)
    quality["signal_asof"] = str(asof.date())
    if asof < pd.Timestamp(quality["target_date"]):
        quality["stale"] = True
        quality["reasons"].append("個股模型訊號未達資料目標")

    return {
        "symbol": symbol, "name": names.get(symbol, "") or "",
        "asof": asof.strftime("%Y-%m-%d"),
        "rank": rank, "n": n, "mom": mom, "mom5": _mr(5), "mom20": _mr(20),
        "score": score, "absolute_momentum": float(g) if pd.notna(g) else float("nan"),
        "score_mode": score_mode, "model_label": spec["label"], "score_kind": spec["score_kind"],
        "model_description": spec["description"], "validation_status": validation_status(score_mode),
        "data_quality": quality,
        "passes_fastsell": passes_fastsell, "passes_sox": passes_sox,
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
