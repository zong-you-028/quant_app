# -*- coding: utf-8 -*-
"""
regime.py - 模組四:市場狀態識別 (Regime Switching, Gaussian HMM)
對「大盤代理(觀察清單等權指數)」的 [日報酬, 滾動波動] 配適高斯 HMM,
輸出每日「多頭狀態機率 P(bull)」,作為平滑乘數動態調節模組三的倉位:
    最終倉位 = Kelly 倉位 × P(bull)
多頭時近乎不打折、震盪/空頭時自動縮手,取代原本生硬的二元順勢濾網。

★ 防 Data Leakage(關鍵):
  hmmlearn 預設的 predict_proba 是「前後向平滑」(posterior given 全序列),會看未來。
  回測/實盤必須用「前向濾波」P(state_t | obs_1..t)——只用截至 t 的觀測。
  本檔以 HMM 參數(startprob_/transmat_/高斯放射機率)自行跑 forward 演算法,
  保證每個時點的狀態機率只依賴「當下與過去」,嚴格因果、無洩漏。
  predict_bull_proba(causal=False) 僅供事後分析(會洩漏),預設 causal=True。
"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM

import config


def market_index_returns(panel: dict) -> pd.Series:
    """由特徵 panel 算「等權大盤代理」日報酬 = 各檔 'ret' 每日橫截面平均。"""
    rets = {s: panel[s]["ret"] for s in panel if "ret" in panel[s].columns}
    wide = pd.DataFrame(rets).sort_index()
    return wide.mean(axis=1).rename("mkt_ret")


def _regime_features(mkt_ret: pd.Series, vol_window: int = None) -> pd.DataFrame:
    """狀態觀測特徵:[日報酬, 滾動波動];皆為當下/過去資訊。"""
    vol_window = vol_window or config.REGIME_VOL_WINDOW
    vol = mkt_ret.rolling(vol_window).std()
    F = pd.concat([mkt_ret.rename("ret"), vol.rename("vol")], axis=1).dropna()
    return F


class RegimeHMM:
    """
    高斯 HMM 市場狀態模型。
    用法:
        rg = RegimeHMM().fit(mkt_ret_train)
        pbull = rg.predict_bull_proba(mkt_ret)          # 因果前向濾波 P(bull)
        mult  = rg.multiplier(mkt_ret)                  # 乘數(套下限 REGIME_FLOOR)
    """

    def __init__(self, n_states: int = None, vol_window: int = None,
                 cov_type: str = None):
        self.n_states = n_states or config.REGIME_N_STATES
        self.vol_window = vol_window or config.REGIME_VOL_WINDOW
        self.cov_type = cov_type or config.REGIME_COV_TYPE
        self.model = None
        self.bull_state_ = None
        self._mu = None            # 訓練集特徵均值(標準化用)
        self._sd = None            # 訓練集特徵標準差
        self._ok = False

    def _scale(self, F: pd.DataFrame) -> np.ndarray:
        """以「訓練集」統計標準化 [報酬, 波動](兩維尺度差很大,不標準化 EM 會壞)。"""
        return ((F - self._mu) / self._sd).values

    def fit(self, mkt_ret: pd.Series):
        """以訓練段大盤報酬配適 HMM,並標定「多頭狀態」(平均報酬最高者)。"""
        F = _regime_features(mkt_ret, self.vol_window)
        self._mu = F.mean()
        self._sd = F.std().replace(0.0, 1.0)
        try:
            self.model = GaussianHMM(
                n_components=self.n_states, covariance_type=self.cov_type,
                n_iter=config.REGIME_N_ITER,
                random_state=config.REGIME_RANDOM_STATE)
            self.model.fit(self._scale(F))
            # 多頭狀態 = 報酬維度(第 0 維,已標準化)平均最高的狀態
            self.bull_state_ = int(np.argmax(self.model.means_[:, 0]))
            self._ok = True
        except Exception:
            self._ok = False
        return self

    # ---- 因果前向濾波:P(state_t | obs_1..t) ----
    def _forward_filter(self, F: np.ndarray) -> np.ndarray:
        """以 HMM 參數跑 forward 演算法,回傳每個時點的「濾波後」狀態機率 (n,K)。"""
        K = self.n_states
        means, covars = self.model.means_, self.model.covars_
        start, trans = self.model.startprob_, self.model.transmat_

        # 各狀態高斯放射機率密度(數值穩定:減去每列最大 log)
        logB = np.column_stack([
            multivariate_normal(means[k], covars[k], allow_singular=True).logpdf(F)
            for k in range(K)
        ])
        B = np.exp(logB - logB.max(axis=1, keepdims=True))

        n = F.shape[0]
        alpha = np.zeros((n, K))
        a = start * B[0]
        s = a.sum()
        alpha[0] = a / s if s > 0 else np.full(K, 1.0 / K)
        for t in range(1, n):
            a = (alpha[t - 1] @ trans) * B[t]      # 前向遞推(只用到 t 為止的觀測)
            s = a.sum()
            alpha[t] = a / s if s > 0 else np.full(K, 1.0 / K)
        return alpha

    def predict_bull_proba(self, mkt_ret: pd.Series,
                           causal: bool = True) -> pd.Series:
        """
        多頭狀態機率 P(bull),對齊 mkt_ret 索引(暖機期以 1.0=不打折回填)。
        causal=True(預設):前向濾波,嚴格因果、無洩漏。
        causal=False:hmmlearn 前後向平滑,僅供事後分析(會看未來,勿用於回測)。
        """
        if not self._ok:
            return pd.Series(1.0, index=mkt_ret.index, name="p_bull")
        F = _regime_features(mkt_ret, self.vol_window)
        Fs = self._scale(F)
        if causal:
            post = self._forward_filter(Fs)
        else:
            post = self.model.predict_proba(Fs)
        p = pd.Series(post[:, self.bull_state_], index=F.index, name="p_bull")
        return p.reindex(mkt_ret.index).fillna(1.0)

    def multiplier(self, mkt_ret: pd.Series, causal: bool = True) -> pd.Series:
        """倉位平滑乘數 = max(REGIME_FLOOR, P(bull))。"""
        p = self.predict_bull_proba(mkt_ret, causal=causal)
        floor = config.REGIME_FLOOR
        return p.clip(lower=floor).rename("regime_mult")


if __name__ == "__main__":
    # 用「明確雙狀態」合成大盤(多頭:正漂移低波動;空頭:負漂移高波動)驗證:
    # 合成股價 panel 本身無 regime 結構,改造一條有 regime 的報酬序列才看得出效果。
    rng = np.random.default_rng(1)
    segs = []
    for _ in range(6):
        segs.append(rng.normal(0.0010, 0.008, 60))    # 多頭段
        segs.append(rng.normal(-0.0012, 0.022, 60))   # 空頭段
    r = np.concatenate(segs)
    idx = pd.bdate_range("2020-01-01", periods=len(r))
    mkt = pd.Series(r, index=idx, name="mkt_ret")
    truth = pd.Series(np.concatenate([np.r_[np.ones(60), np.zeros(60)]
                                      for _ in range(6)]), index=idx)

    split = int(len(mkt) * config.TRAIN_RATIO)
    rg = RegimeHMM().fit(mkt.iloc[:split])              # 只用訓練段配適
    pbull = rg.predict_bull_proba(mkt, causal=True)     # 因果前向濾波
    pbull_s = rg.predict_bull_proba(mkt, causal=False)  # 事後平滑(會看未來)

    print("HMM 配適成功 :", rg._ok, "| 多頭狀態 index:", rg.bull_state_)
    print("標準化後各狀態均值[報酬,波動]:", np.round(rg.model.means_, 3).tolist())
    print("因果 P(bull) 範圍 :", round(float(pbull.min()), 2),
          "~", round(float(pbull.max()), 2), "| 均值:", round(float(pbull.mean()), 2))
    print("真多頭段 vs 真空頭段 平均 P(bull):",
          round(float(pbull[truth == 1].mean()), 2), "vs",
          round(float(pbull[truth == 0].mean()), 2), "(應明顯高低有別)")
    print("因果 vs 平滑 平均差異:", round(float((pbull - pbull_s).abs().mean()), 4),
          "(>0 代表因果濾波確實不看未來、無洩漏)")
    print("最後一日多頭機率乘數 :", round(float(rg.multiplier(mkt).iloc[-1]), 3))
