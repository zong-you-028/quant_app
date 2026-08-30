# -*- coding: utf-8 -*-
"""
backtest_engine.py - 回測層
職責:
  1. 依訊號(0~4)建立倉位:4->1.0、3->0.5、2->不變、1->砍半、0->平倉
     其中 1/2 為「路徑相依(path-dependent)」,必須用迴圈逐日建倉。
  2. 策略報酬 = 前一日倉位 × 當日報酬(用 shift(1) 避免前視偏差 look-ahead）。
  3. KPI:總報酬率、最大回撤(MDD)、勝率(以持倉回合計)。
  4. 回傳含 equity(權益曲線)Series 的結果字典。
"""
import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# 建倉(path-dependent:用迴圈,因 1/2 依賴前一日倉位)
# ---------------------------------------------------------------------------
def build_positions(signals: pd.Series, allow_short: bool = False) -> pd.Series:
    """
    依訊號序列逐日建立倉位序列(path-dependent)。支援兩種模式:
      config.POSITION_MODE == "incremental"(預設,分批進出):
        每個訊號只「加碼/減碼」一個級距(config.POSITION_STEP),
        倉位在 [POSITION_MIN, POSITION_MAX] 間漸進移動,不會一次全買或全賣。
      config.POSITION_MODE == "target"(舊行為):
        4->1.0 滿倉、3->0.5 半倉、2->不變、1->砍半、0->平倉。
    allow_short=True 時最小倉位放寬到 -1.0(可反手作空)。
    回傳與 signals 同索引的倉位 Series(float)。
    """
    mode = getattr(config, "POSITION_MODE", "target")
    # 最大持有天數(短期波段用):一筆持倉超過 N 個交易日就強制平倉,
    # 並「鎖倉」到訊號退燒(回到中性/賣出 <=2)後才允許重新進場,
    # 避免在強勢趨勢中被同一訊號一路抱成數個月(0/None = 不啟用)。
    max_hold = getattr(config, "MAX_HOLD_DAYS", 0) or 0
    pos = np.zeros(len(signals), dtype=float)
    cur = 0.0   # 當前倉位(從空手開始)
    held = 0    # 目前這筆持倉已持有的交易日數
    locked = False  # 觸發最大持有後鎖倉,等訊號退燒才解鎖
    sig_vals = signals.values

    def _apply_max_hold(cur, held, locked, s):
        """套用最大持有上限與鎖倉狀態,回傳 (cur, held, locked)。"""
        if locked:
            if int(s) <= 2:           # 訊號退燒(中性/賣出)-> 解鎖,之後可再進場
                locked = False
            return 0.0, 0, locked     # 鎖倉期間維持空手
        if cur != 0.0:
            held += 1
        else:
            held = 0
        if max_hold and held >= max_hold:   # 持有到頂 -> 強制平倉並鎖倉
            return 0.0, 0, True
        return cur, held, locked

    if mode == "incremental":
        step = config.POSITION_STEP
        hi = config.POSITION_MAX
        lo = -1.0 if allow_short else config.POSITION_MIN
        for i, s in enumerate(sig_vals):
            if locked:
                cur, held, locked = _apply_max_hold(0.0, 0, locked, s)
                pos[i] = cur
                continue
            cur = min(hi, max(lo, cur + step.get(int(s), 0.0)))   # 加減一級距並夾在上下限
            cur, held, locked = _apply_max_hold(cur, held, locked, s)
            pos[i] = cur
        return pd.Series(pos, index=signals.index, name="position")

    # --- 舊版:直接跳到目標倉位 ---
    for i, s in enumerate(sig_vals):
        if locked:
            cur, held, locked = _apply_max_hold(0.0, 0, locked, s)
            pos[i] = cur
            continue
        if s == 4:
            cur = 1.0                       # 滿倉
        elif s == 3:
            cur = 0.5                       # 半倉
        elif s == 2:
            cur = cur                       # 不變(維持前一日)
        elif s == 1:
            cur = cur * 0.5                 # 砍半(路徑相依)
        elif s == 0:
            cur = -1.0 if allow_short else 0.0   # 平倉或反手作空
        cur, held, locked = _apply_max_hold(cur, held, locked, s)
        pos[i] = cur
    return pd.Series(pos, index=signals.index, name="position")


# ---------------------------------------------------------------------------
# 訊號遲滯(hysteresis):需連續 confirm 日同向才換倉,降低過度交易
# ---------------------------------------------------------------------------
def confirm_signals(signals: pd.Series, confirm: int = 1) -> pd.Series:
    """
    把原始訊號做「連續確認」過濾:只有當某個新訊號連續出現 confirm 天,
    才真正切換為新訊號;否則沿用目前已確認的訊號。
    confirm<=1 等同不過濾(回傳原序列)。可大幅減少每日翻單的換手。
    """
    if confirm is None or confirm <= 1:
        return signals
    vals = signals.values
    out = np.empty(len(vals), dtype=vals.dtype)
    current = vals[0] if len(vals) else 0   # 目前已確認的訊號
    cand = current                          # 候選新訊號
    cand_count = 0                          # 候選連續天數
    for i, s in enumerate(vals):
        if s == current:
            cand, cand_count = s, 0          # 與現況相同 -> 無待確認
        else:
            if s == cand:
                cand_count += 1              # 候選延續,累加
            else:
                cand, cand_count = s, 1      # 換了新候選,重新計數
            if cand_count >= confirm:
                current, cand_count = s, 0    # 連續達標 -> 正式換倉
        out[i] = current
    return pd.Series(out, index=signals.index, name=signals.name)


# ---------------------------------------------------------------------------
# KPI 計算
# ---------------------------------------------------------------------------
def _max_drawdown(equity: pd.Series) -> float:
    """最大回撤(MDD):權益相對歷史高點(running peak)的最大跌幅(負值)。"""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()                 # 歷史高點
    drawdown = equity / running_peak - 1.0          # 各時點回撤
    return float(drawdown.min())                     # 最深回撤(負)


def _round_trip_winrate(position: pd.Series, strat_ret: pd.Series,
                        returns: pd.Series) -> float:
    """
    勝率(以持倉回合計、且與「買進持有」對比):
    從空手進場到回到空手算一筆 round-trip。該回合「策略累積報酬」需贏過
    「同一段期間單純抱住該股的買進持有報酬」才算勝。
    (即:勝 = 這趟進出有沒有比『同期間死抱不動』更好,而非只看賺賠。)
    回傳勝場 / 總回合數。
    """
    in_trade = False
    trade_ret = 0.0          # 此回合策略累積報酬(含成本)
    bench_ret = 0.0          # 此回合同期間買進持有累積報酬(原始股票報酬)
    wins = 0
    trades = 0
    pos_vals = position.values
    ret_vals = strat_ret.fillna(0.0).values
    bh_vals = returns.fillna(0.0).values

    for i in range(len(pos_vals)):
        prev_pos = pos_vals[i - 1] if i > 0 else 0.0
        # 進場:由空手 -> 有倉
        if not in_trade and prev_pos != 0.0:
            in_trade = True
            trade_ret = 0.0
            bench_ret = 0.0
        # 持倉中:同步累積「策略報酬」與「同期間買進持有報酬」
        if in_trade:
            trade_ret += ret_vals[i]
            bench_ret += bh_vals[i]
        # 出場:由有倉 -> 空手(本日倉位歸零)
        if in_trade and pos_vals[i] == 0.0:
            trades += 1
            if trade_ret > bench_ret:        # 贏過同期間買進持有才算勝
                wins += 1
            in_trade = False

    # 若結束時仍持倉,將最後未平倉回合也計入
    if in_trade:
        trades += 1
        if trade_ret > bench_ret:
            wins += 1

    return float(wins / trades) if trades > 0 else 0.0


# ---------------------------------------------------------------------------
# 回測主函式
# ---------------------------------------------------------------------------
def run_backtest(signals: pd.Series, returns: pd.Series,
                 allow_short: bool = False,
                 cost_per_turnover: float = None,
                 confirm_days: int = None) -> dict:
    """
    向量化回測(含交易成本 + 訊號遲滯 + 買進持有基準):
      - 訊號先經 confirm_signals 遲滯過濾(連續 confirm_days 同向才換倉)
      - 倉位由 build_positions 迴圈建立(1/2 path-dependent)
      - 毛策略報酬 = position.shift(1) × 當日報酬(避免前視偏差)
      - 交易成本 = |當日倉位變動| × cost_per_turnover,從報酬扣除
      - equity = (1 + 淨策略報酬).cumprod()
      - 同時計算「買進持有(buy & hold)」基準供對照
    參數:
      cost_per_turnover : 每單位換手成本(預設 config.COST_PER_TURNOVER)
      confirm_days      : 訊號遲滯天數(預設 config.SIGNAL_CONFIRM_DAYS)
    回傳 dict:total_return(含成本淨值)/ gross_return / buy_hold_return /
              max_drawdown / win_rate / turnover / n_changes / equity / ...
    """
    cost_per_turnover = (config.COST_PER_TURNOVER
                         if cost_per_turnover is None else cost_per_turnover)
    confirm_days = (config.SIGNAL_CONFIRM_DAYS
                    if confirm_days is None else confirm_days)

    # 對齊索引(以兩者交集為準),避免錯位
    idx = signals.index.intersection(returns.index)
    signals = signals.loc[idx]
    returns = returns.loc[idx].fillna(0.0)

    # 0) 訊號遲滯:連續確認才換倉(降低過度交易)
    signals = confirm_signals(signals, confirm_days)

    # 1) 建倉(路徑相依)
    position = build_positions(signals, allow_short=allow_short)

    # 2) 毛策略報酬:用前一日倉位 × 當日報酬(shift(1) 避免 look-ahead)
    gross_ret = position.shift(1).fillna(0.0) * returns

    # 3) 交易成本:當日換手 = |倉位變動|;首日以建倉量計
    turnover_daily = position.diff()
    turnover_daily.iloc[0] = position.iloc[0] if len(position) else 0.0
    turnover_daily = turnover_daily.abs()
    cost = turnover_daily * cost_per_turnover
    strat_ret = gross_ret - cost                      # 淨策略報酬(扣成本)

    # 4) 權益曲線(含成本淨值)+ 買進持有基準
    equity = (1.0 + strat_ret).cumprod()
    buy_hold_equity = (1.0 + returns).cumprod()

    # 5) KPI
    gross_equity = (1.0 + gross_ret).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    gross_return = float(gross_equity.iloc[-1] - 1.0) if len(gross_equity) else 0.0
    buy_hold_return = float(buy_hold_equity.iloc[-1] - 1.0) if len(buy_hold_equity) else 0.0
    mdd = _max_drawdown(equity)
    win_rate = _round_trip_winrate(position, strat_ret, returns)
    n_changes = int((position.diff().abs() > 1e-9).sum())

    return {
        "total_return": total_return,        # 含成本淨報酬(headline)
        "gross_return": gross_return,        # 未扣成本的毛報酬
        "buy_hold_return": buy_hold_return,  # 買進持有基準
        "max_drawdown": mdd,                 # 最大回撤(負值)
        "win_rate": win_rate,                # 持倉回合勝率
        "turnover": float(turnover_daily.sum()),  # 總換手(倍)
        "n_changes": n_changes,              # 倉位變動次數
        "equity": equity,                    # 權益曲線(含成本)
        "buy_hold_equity": buy_hold_equity,  # 買進持有權益曲線
        "position": position,                # 倉位序列(遲滯後)
        "strat_ret": strat_ret,              # 每日淨策略報酬
    }


# ---------------------------------------------------------------------------
# 視窗化績效:只取最近 N 天重算 KPI(對齊短線操作的時間尺度)
# ---------------------------------------------------------------------------
def window_metrics(result: dict, lookback_days: int = None) -> dict:
    """
    從 run_backtest 的完整結果中,擷取「最近 lookback_days 天」重新計算 KPI:
      - 權益曲線在視窗起點重新歸一為 1.0(避免顯示自 2015 年起的累積倍數)
      - total_return / cagr / max_drawdown / win_rate / buy_hold_return 全部
        只用視窗內的資料計算
    模型訓練仍用全歷史(在 run_backtest 之前),這裡只影響「展示用」的數字,
    讓做短期波段時看到的年化報酬是近一年的尺度,而非 5~10 年平均。
    lookback_days <=0 或 None -> 沿用全期(等同不裁切)。
    回傳 dict:total_return / cagr / max_drawdown / win_rate / buy_hold_return /
              equity / buy_hold_equity / position / years。
    """
    equity = result["equity"]
    strat_ret = result.get("strat_ret")
    position = result["position"]
    # 由買進持有權益反推每日原始報酬(供視窗內基準與勝率重算)
    bh_equity_full = result["buy_hold_equity"]
    returns = bh_equity_full.pct_change().fillna(0.0)

    if not lookback_days or lookback_days <= 0 or not len(equity):
        idx = equity.index
    else:
        cutoff = equity.index[-1] - pd.Timedelta(days=lookback_days)
        idx = equity.index[equity.index >= cutoff]

    sr = (strat_ret.reindex(idx).fillna(0.0)
          if strat_ret is not None else equity.reindex(idx).pct_change().fillna(0.0))
    rr = returns.reindex(idx).fillna(0.0)
    pos = position.reindex(idx).fillna(0.0)

    # 視窗內重新歸一(起點 = 1.0)
    eq = (1.0 + sr).cumprod()
    bh = (1.0 + rr).cumprod()

    total_return = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
    buy_hold_return = float(bh.iloc[-1] - 1.0) if len(bh) else 0.0
    mdd = _max_drawdown(eq)
    win_rate = _round_trip_winrate(pos, sr, rr)

    if len(idx) > 1:
        years = (idx[-1] - idx[0]).days / 365.25
    else:
        years = 0.0
    base = 1.0 + total_return
    cagr = base ** (1.0 / years) - 1.0 if (years > 0 and base > 0) else total_return

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "buy_hold_return": buy_hold_return,
        "equity": eq,
        "buy_hold_equity": bh,
        "position": pos,
        "years": years,
    }


if __name__ == "__main__":
    # 簡易自測
    from core.data_pipeline import seed_sample_data, build_features
    from core.ml_strategy import label_data, prepare_xy, train_model, predict_series
    seed_sample_data("TEST")
    feat = build_features("TEST")
    lab = label_data(feat)
    X, y = prepare_xy(lab)
    model, _ = train_model(X, y)
    sig = predict_series(model, feat)
    res = run_backtest(sig, feat["ret"])
    print("total_return:", round(res["total_return"], 4))
    print("max_drawdown:", round(res["max_drawdown"], 4))
    print("win_rate:", round(res["win_rate"], 4))
    print("equity tail:\n", res["equity"].tail())
