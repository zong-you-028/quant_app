# -*- coding: utf-8 -*-
"""
screener.py - 勝率掃描 / 預期點位估算(純函式,不依賴 Flet,便於單元測試)
職責:
  1. expected_levels():依「現價 + 期間波動 sigma + 訊號強度」估算預期點位
     - 預期目標價 target = 現價 × (1 + 訊號偏移 × 期間sigma)
     - 預期區間 [low_band, up_band] = 現價 × (1 ± BAND_SIGMA × 期間sigma)
     - 停損 stop:順訊號方向取反向 1σ(多單看下緣、空訊號看上緣)
  2. rank_top():依回測勝率(win_rate)由高至低排序,取前 K 支。
注意:這是「統計式期望」估算,非保證價位;期間 = config.FUTURE_N 個交易日。
"""
import numpy as np

import config


def expected_levels(last_close: float, daily_vol: float, signal: int,
                    horizon: int = None) -> dict:
    """
    估算未來 horizon 個交易日的預期點位。
      sigma_h    = 日報酬標準差 × sqrt(horizon)  (期間波動)
      exp_return = 訊號方向偏移 × sigma_h          (預期報酬)
      target     = 現價 × (1 + exp_return)         (預期目標價)
      up/low_band= 現價 × (1 ± BAND_SIGMA×sigma_h) (預期區間)
      stop       = 多方訊號取下緣、空方訊號取上緣  (停損參考)
    """
    horizon = horizon or config.FUTURE_N
    sigma_h = float(daily_vol) * np.sqrt(horizon)        # 期間波動
    bias = config.SIGNAL_BIAS.get(int(signal), 0.0)       # 訊號方向偏移
    exp_return = bias * sigma_h

    target = last_close * (1.0 + exp_return)
    up_band = last_close * (1.0 + config.BAND_SIGMA * sigma_h)
    low_band = last_close * (1.0 - config.BAND_SIGMA * sigma_h)
    stop = low_band if bias >= 0 else up_band             # 順訊號方向的停損側

    return {
        "sigma_h": sigma_h,
        "exp_return": exp_return,
        "target": target,
        "up_band": up_band,
        "low_band": low_band,
        "stop": stop,
    }


def suggest_trade_amount(signal: int, win_rate: float = 0.0) -> dict:
    """
    依「訊號強度 + 近一年勝率信心」換算成一個實際的建議單筆金額(新台幣),
    讓使用者知道這次大概該投入(買)或收回(賣)多少錢。

      strength = |訊號-2| / 2      # 0=中性、0.5=偏多/逢高、1.0=強烈買/賣
      conf     = clamp(win_rate, 0, 1)
      score    = STRENGTH_W × strength + CONF_W × conf
      amount   = MIN + (MAX-MIN) × score,四捨五入到 TRADE_AMOUNT_ROUND

    回傳 dict:
      action    : "買進" / "賣出" / "觀望"
      amount    : 建議金額(int;觀望時為 0)
      direction : +1 買 / -1 賣 / 0 觀望
      strength  : 訊號強度(0~1)
    中性訊號(2)視為觀望,金額為 0(不建議進出)。
    """
    sig = int(signal)
    strength = abs(sig - 2) / 2.0                       # 0 / 0.5 / 1.0
    if sig == 2 or strength == 0.0:
        return {"action": "觀望", "amount": 0, "direction": 0, "strength": 0.0}

    conf = min(1.0, max(0.0, float(win_rate)))
    sw = getattr(config, "TRADE_STRENGTH_W", 0.65)
    cw = getattr(config, "TRADE_CONF_W", 0.35)
    score = sw * strength + cw * conf                   # 0~1(權重和=1 時)

    lo = getattr(config, "TRADE_AMOUNT_MIN", 10000)
    hi = getattr(config, "TRADE_AMOUNT_MAX", 40000)
    step = getattr(config, "TRADE_AMOUNT_ROUND", 1000) or 1
    raw = lo + (hi - lo) * score
    amount = int(round(raw / step) * step)
    amount = max(lo, min(hi, amount))                   # 夾在 [MIN, MAX]

    direction = 1 if sig >= 3 else -1
    return {
        "action": "買進" if direction > 0 else "賣出",
        "amount": amount,
        "direction": direction,
        "strength": strength,
    }


def rank_top(results: list, k: int = None) -> list:
    """依 win_rate 由高至低排序,取前 k 筆(預設 config.SCAN_TOP_K)。"""
    k = k or config.SCAN_TOP_K
    return sorted(results, key=lambda r: r.get("win_rate", 0.0), reverse=True)[:k]
