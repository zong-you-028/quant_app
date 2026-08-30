# -*- coding: utf-8 -*-
"""
base_model.py - 模組二之二:LightGBM 基層模型 (Base Model, 單調約束)
第一層模型,輸出「初始方向訊號」0 / 2 / 4(對應 Triple-Barrier 的下欄/中性/上欄)。

★ 為什麼用「迴歸方向分數 + 分位門檻」而非直接多分類?
  單調性約束(Monotonic Constraints)在「多分類」LightGBM 會把同一向量套到所有類別,
  對 class 0 與 class 4 同時施加同向約束在語意上自相矛盾。
  改用「迴歸」預測一個連續方向分數:label {0→-1, 2→0, 4→+1},
  對「動能/趨勢/相對強弱/籌碼」等方向明確的特徵施加 +1 遞增約束,語意嚴謹、保證單調。
  連續分數還有兩個附帶好處:
    1) 直接餵模組五的 Rank IC(需要連續分數對未來報酬排序)。
    2) 作為模組三 Meta-Model 與 Kelly 倉位的輸入。
  最後用「訓練集分數」的分位門檻把分數切成 0/2/4(門檻在訓練段估計,套 OOS 不洩漏)。

★ 防 Data Leakage:特徵全部來自模組一/模組二(當下或過去);門檻只用訓練段分數估計。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

import config


# label(0/2/4)<-> 方向分數目標(-1/0/+1)
_LABEL_TO_DIR = {0: -1.0, 2: 0.0, 4: 1.0}


class LGBMBaseModel:
    """
    LightGBM 迴歸方向分數 + 分位門檻 -> 0/2/4。
    用法:
        m = LGBMBaseModel().fit(X_train, y_train_label)
        score = m.score(X)          # 連續方向分數(供 Rank IC / Meta / 倉位)
        sig   = m.predict(X)        # 離散訊號 0/2/4
    """

    def __init__(self, feature_cols=None, monotone: bool = True,
                 params: dict = None, low_q: float = None, high_q: float = None):
        self.feature_cols = list(feature_cols or config.ALL_FEATURE_COLS)
        self.monotone = monotone
        self.params = dict(params or config.LGBM_PARAMS)
        self.low_q = config.BASE_SIGNAL_LOW_Q if low_q is None else low_q
        self.high_q = config.BASE_SIGNAL_HIGH_Q if high_q is None else high_q
        self.model = None
        self.low_thr_ = None       # 訓練集分數低分位門檻(<= -> 0)
        self.high_thr_ = None      # 訓練集分數高分位門檻(>= -> 4)

    # ---- 單調約束向量(對齊 feature_cols 順序;無約束者為 0)----
    def _monotone_vector(self):
        if not self.monotone:
            return None
        vec = [int(config.MONOTONE_FEATURES.get(c, 0)) for c in self.feature_cols]
        return vec if any(v != 0 for v in vec) else None

    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        """取出固定順序特徵;缺漏欄補 0(例如某些橫截面欄在單檔情境不存在)。"""
        miss = [c for c in self.feature_cols if c not in X.columns]
        if miss:
            X = X.copy()
            for c in miss:
                X[c] = 0.0
        return X[self.feature_cols]

    def fit(self, X: pd.DataFrame, y_label: pd.Series):
        """以方向分數為迴歸目標訓練;並用訓練集分數分位估計 0/2/4 門檻。"""
        Xf = self._X(X)
        y_dir = pd.Series(y_label).map(_LABEL_TO_DIR).astype(float).values

        mc = self._monotone_vector()
        params = dict(self.params)
        if mc is not None:
            params["monotone_constraints"] = mc
        self.model = lgb.LGBMRegressor(objective="regression", **params)
        self.model.fit(Xf, y_dir)

        s = self.model.predict(Xf)                       # 訓練集連續分數
        self.low_thr_ = float(np.quantile(s, self.low_q))
        self.high_thr_ = float(np.quantile(s, self.high_q))
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """連續方向分數(越大越偏多);供 Rank IC / Meta-Model / 倉位 sizing。"""
        if self.model is None:
            raise RuntimeError("模型尚未訓練,請先 fit()")
        s = self.model.predict(self._X(X))
        return pd.Series(s, index=X.index, name="dir_score")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """離散訊號:分數 >= 高門檻 -> 4、<= 低門檻 -> 0、其餘 -> 2。"""
        s = self.score(X).values
        out = np.full(len(s), 2, dtype=int)
        out[s >= self.high_thr_] = 4
        out[s <= self.low_thr_] = 0
        return pd.Series(out, index=X.index, name="base_signal")

    def feature_importance(self) -> pd.Series:
        """特徵重要度(gain),由大到小。"""
        if self.model is None:
            raise RuntimeError("模型尚未訓練,請先 fit()")
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_cols).sort_values(ascending=False)


def prepare_xy(features: pd.DataFrame, labels: pd.DataFrame):
    """
    把模組二特徵與模組一 Triple-Barrier 標籤對齊成 (X, y_label, ret_event)。
    只取兩者共同的日期索引;X 僅含 ALL_FEATURE_COLS(缺欄由模型端補 0)。
    """
    idx = features.index.intersection(labels.index)
    X = features.reindex(idx)
    y = labels.reindex(idx)["label"].astype(int)
    ret = labels.reindex(idx)["ret_event"]
    return X, y, ret


if __name__ == "__main__":
    # 端到端離線驗證:多檔合成 -> 橫截面特徵 -> 三欄標籤 -> 基層模型 -> 模組五指標
    from core.data_pipeline import seed_sample_data
    from core.features_cs import build_feature_panel
    from core.labeling import triple_barrier_labels
    from core import evaluation as ev

    syms = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for i, s in enumerate(syms):
        seed_sample_data(s, seed=21 + i)
    panel = build_feature_panel(syms)

    feat = panel["AAA"]
    lab = triple_barrier_labels(feat)
    X, y, ret = prepare_xy(feat, lab)

    # 時序切分(不 shuffle):前 80% 訓練、後 20% 驗證
    split = int(len(X) * config.TRAIN_RATIO)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    rette = ret.iloc[split:]

    m = LGBMBaseModel().fit(Xtr, ytr)
    pred = m.predict(Xte)
    sc = m.score(Xte)

    print("驗證集訊號分布:\n", pred.value_counts().sort_index())
    print("Precision@4 :", round(ev.precision_at_class(yte, pred, 4), 3))
    print("Rank IC     :", round(ev.rank_ic(sc, rette), 3))
    print("單調約束     :", "ON" if m._monotone_vector() else "OFF")
    print("Top5 特徵重要度:\n", m.feature_importance().head(5).round(1))
