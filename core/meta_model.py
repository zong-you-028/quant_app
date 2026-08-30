# -*- coding: utf-8 -*-
"""
meta_model.py - 模組三:Meta-Labeling(元標籤雙層)+ 改良凱利倉位
第二層模型。López de Prado 的 meta-labeling:第一層(base_model)決定「方向/側」,
第二層(本檔)決定「該不該下、下多大」—— 不調方向,只優化勝率與資金控管。

雙層分工:
  Base   : 0/2/4 方向訊號 + 連續方向分數 dir_score(模組二)
  Meta   : 只在「基層發出可執行多單(=META_TARGET_SIGNAL)」的樣本上,訓練二元分類器,
           預測「跟著做多會不會賺」 P(success)= P(ret_event > 0)。
  Sizing : 用 P(success) 配合改良版凱利公式,輸出 0~1 連續倉位(模組四再乘上市場狀態)。

特徵:ALL_FEATURE_COLS + dir_score(把第一層輸出當第二層的強特徵)。

★ 防 Data Leakage:
  - Meta 特徵全為「當下/過去」(含 base 的 dir_score,亦由當下特徵算出)。
  - 二元目標 y 與賠率 b 皆用「未來事件報酬 ret_event」,僅作為 y / 訓練統計,不進特徵。
  - 賠率 b、門檻等統計只用「訓練集」估計,套用 OOS 不洩漏。
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config


# ---------------------------------------------------------------------------
# 改良版凱利公式(binary payoff odds)
# ---------------------------------------------------------------------------
def kelly_fraction(p, b):
    """凱利最適比例 f* = p − (1−p)/b(p=勝率,b=賠率=均盈/均虧)。可為負。"""
    p = np.asarray(p, dtype=float)
    return p - (1.0 - p) / b


def kelly_position(p, b, fraction: float = None, cap: float = None):
    """倉位 = clip(分數凱利 × f*, 0, cap);f*<0(劣勢)時不進場(0)。"""
    fraction = config.KELLY_FRACTION if fraction is None else fraction
    cap = config.KELLY_CAP if cap is None else cap
    f = fraction * kelly_fraction(p, b)
    return np.clip(f, 0.0, cap)


class MetaModel:
    """
    第二層 XGBoost 二元分類器 + 凱利倉位。
    用法:
        meta = MetaModel(feature_cols).fit(X_tr, score_tr, base_pred_tr, ret_tr)
        p    = meta.predict_proba(X, score)             # P(success)
        pos  = meta.size(X, score, base_pred)           # 0~1 倉位(非多單訊號=0)
    """

    def __init__(self, feature_cols=None, params: dict = None,
                 target_signal: int = None):
        self.feature_cols = list(feature_cols or config.ALL_FEATURE_COLS)
        self.params = dict(params or config.META_PARAMS)
        self.target_signal = (config.META_TARGET_SIGNAL if target_signal is None
                              else target_signal)
        self.model = None
        self.b_ = config.KELLY_DEFAULT_B          # 訓練集估計的賠率(均盈/均虧)
        self.base_rate_ = 0.5                     # 訓練集多單樣本的勝率(備援用)

    # ---- 組裝 Meta 特徵矩陣:ALL_FEATURE_COLS + dir_score ----
    def _meta_X(self, X: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
        cols = {}
        for c in self.feature_cols:
            cols[c] = X[c].values if c in X.columns else np.zeros(len(X))
        m = pd.DataFrame(cols, index=X.index)
        m["dir_score"] = pd.Series(score).reindex(X.index).values
        return m

    def fit(self, X: pd.DataFrame, score: pd.Series,
            base_pred: pd.Series, ret_event: pd.Series):
        """只用「基層多單訊號」樣本訓練;二元 y=(ret_event>0);並估計凱利賠率 b。"""
        base_pred = pd.Series(base_pred).reindex(X.index)
        ret_event = pd.Series(ret_event).reindex(X.index)
        mask = (base_pred == self.target_signal) & ret_event.notna()

        mX = self._meta_X(X, score)[mask]
        y = (ret_event[mask] > 0).astype(int)

        # 凱利賠率 b = 平均獲利幅度 / 平均虧損幅度(僅用訓練集多單樣本)
        wins = ret_event[mask][ret_event[mask] > 0]
        losses = ret_event[mask][ret_event[mask] < 0]
        if len(wins) and len(losses) and losses.abs().mean() > 0:
            self.b_ = max(config.KELLY_MIN_B,
                          float(wins.mean() / losses.abs().mean()))
        else:
            self.b_ = config.KELLY_DEFAULT_B
        self.base_rate_ = float(y.mean()) if len(y) else 0.5

        # 樣本太少或單一類別 -> 不訓練(predict 時退回 base_rate_)
        if len(y) >= 30 and y.nunique() == 2:
            self.model = XGBClassifier(**self.params)
            self.model.fit(mX, y)
        else:
            self.model = None
        return self

    def predict_proba(self, X: pd.DataFrame, score: pd.Series) -> pd.Series:
        """P(跟著基層做多會成功)。未訓練(樣本不足)時退回訓練集勝率。"""
        if self.model is None:
            return pd.Series(self.base_rate_, index=X.index, name="p_success")
        p = self.model.predict_proba(self._meta_X(X, score))[:, 1]
        return pd.Series(p, index=X.index, name="p_success")

    def size(self, X: pd.DataFrame, score: pd.Series,
             base_pred: pd.Series) -> pd.Series:
        """
        凱利倉位(0~1):僅在基層多單訊號處依 P(success) 與賠率 b 下注,其餘為 0。
        模組四會再把此倉位乘上「市場多頭狀態機率」做平滑調節。
        """
        p = self.predict_proba(X, score)
        pos = pd.Series(kelly_position(p.values, self.b_), index=X.index)
        actionable = (pd.Series(base_pred).reindex(X.index) == self.target_signal)
        pos = pos.where(actionable, 0.0)
        return pos.rename("position")


if __name__ == "__main__":
    # 端到端:合成多檔 -> 橫截面 -> 三欄標籤 -> 基層 -> Meta -> 凱利倉位 -> 模組五
    from core.data_pipeline import seed_sample_data
    from core.features_cs import build_feature_panel
    from core.labeling import triple_barrier_labels
    from core.base_model import LGBMBaseModel, prepare_xy
    from core import evaluation as ev

    syms = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for i, s in enumerate(syms):
        seed_sample_data(s, seed=31 + i)
    panel = build_feature_panel(syms)
    feat = panel["AAA"]
    lab = triple_barrier_labels(feat)
    X, y, ret = prepare_xy(feat, lab)

    split = int(len(X) * config.TRAIN_RATIO)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr = y.iloc[:split]
    rettr, rette = ret.iloc[:split], ret.iloc[split:]

    base = LGBMBaseModel().fit(Xtr, ytr)
    score_tr, score_te = base.score(Xtr), base.score(Xte)
    pred_tr, pred_te = base.predict(Xtr), base.predict(Xte)

    meta = MetaModel().fit(Xtr, score_tr, pred_tr, rettr)
    p_te = meta.predict_proba(Xte, score_te)
    pos_te = meta.size(Xte, score_te, pred_te)

    # 用 Meta 機率算 Log-Loss(僅在驗證集的多單樣本上,目標=該多單是否獲利)
    long_mask = (pred_te == config.META_TARGET_SIGNAL) & rette.notna()
    yte_meta = (rette[long_mask] > 0).astype(int)
    print("Meta 估計賠率 b      :", round(meta.b_, 3))
    print("驗證多單樣本數        :", int(long_mask.sum()))
    if long_mask.sum() and yte_meta.nunique() == 2:
        print("Meta Log-Loss        :",
              round(ev.meta_log_loss(yte_meta, p_te[long_mask]), 4))
    print("凱利倉位>0 的天數     :", int((pos_te > 0).sum()),
          "| 平均倉位:", round(float(pos_te[pos_te > 0].mean() or 0), 3))
    print("倉位範圍              :", round(float(pos_te.min()), 3),
          "~", round(float(pos_te.max()), 3))
