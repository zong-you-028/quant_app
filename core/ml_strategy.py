# -*- coding: utf-8 -*-
"""
ml_strategy.py - 機器學習策略層(五元分類 0~4)
職責:
  1. label_data():用「未來 N 日報酬 × 籌碼方向 × 價格位階」貼 0~4 標籤
     —— 未來報酬僅用於貼標籤,絕不進特徵 X(避免 data leakage)
  2. prepare_xy():組裝固定欄位順序的 X / y
  3. train_model():RandomForest(class_weight=balanced)+ 時序切分驗證
  4. predict_series() / predict_latest():整段或最新一日訊號
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import config


# ---------------------------------------------------------------------------
# 貼標籤(labeling)
# ---------------------------------------------------------------------------
def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    依下列規則貼 0~4 標籤(僅用於監督式學習的 y,未來報酬不可進 X):
      4 強烈買進:未來報酬 >= 高分位 + 籌碼增 + OSC>0
      3 偏多佈局:未來報酬 > 0   + 籌碼增 + 低位階(底部吸籌)
      1 逢高獲利:未來報酬 < 0   + 籌碼減 + 高位階(高檔出貨)
      0 強烈賣出:未來報酬 <= 低分位 + 籌碼減 + OSC<0
      2 中性觀望:其餘
    回傳含 'label' 與輔助欄位的新 DataFrame(已去除尾端 N 日無未來報酬者)。
    """
    out = df.copy()

    # --- 未來 N 日報酬(forward return);僅供貼標籤 ---
    fwd_ret = out["close"].shift(-config.FUTURE_N) / out["close"] - 1.0

    # --- 價格位階:120 日 rolling 百分位(0~1,愈高代表愈接近近期高點)---
    def _rolling_rank(s: pd.Series) -> float:
        # 回傳最後一個值在視窗內的百分位
        return s.rank(pct=True).iloc[-1]

    price_pos = out["close"].rolling(config.RANK_WINDOW).apply(_rolling_rank, raw=False)

    # --- 籌碼方向:用集中度週變化率判斷增/減 ---
    chip_up = out["chip_ratio_chg"] > 0      # 籌碼增(大戶加碼)
    chip_dn = out["chip_ratio_chg"] < 0      # 籌碼減(大戶調節)

    # --- 報酬分位門檻(以全樣本分位數為基準)---
    valid = fwd_ret.dropna()
    q_high = valid.quantile(config.RET_HIGH_Q) if len(valid) else 0.0
    q_low = valid.quantile(config.RET_LOW_Q) if len(valid) else 0.0

    osc = out["osc"]
    label = pd.Series(2, index=out.index, dtype=int)   # 預設中性=2

    # 條件(注意順序:強訊號最後覆蓋,確保 4/0 優先級最高)
    cond3 = (fwd_ret > 0) & chip_up & (price_pos <= config.POS_LOW)        # 偏多佈局
    cond1 = (fwd_ret < 0) & chip_dn & (price_pos >= config.POS_HIGH)       # 逢高獲利
    cond4 = (fwd_ret >= q_high) & chip_up & (osc > 0)                       # 強烈買進
    cond0 = (fwd_ret <= q_low) & chip_dn & (osc < 0)                        # 強烈賣出

    label[cond3] = 3
    label[cond1] = 1
    label[cond4] = 4     # 強訊號覆蓋弱訊號
    label[cond0] = 0

    out["fwd_ret"] = fwd_ret           # 保留供檢視(非特徵)
    out["price_pos"] = price_pos       # 保留供檢視(非特徵)
    out["label"] = label

    # 去除尾端 N 日(無未來報酬)與位階暖機期 NaN
    out = out.dropna(subset=["fwd_ret", "price_pos"]).copy()
    out["label"] = out["label"].astype(int)
    return out


# ---------------------------------------------------------------------------
# 組裝 X / y
# ---------------------------------------------------------------------------
def prepare_xy(labeled: pd.DataFrame):
    """
    依 config.FEATURE_COLS 的固定順序取出特徵 X,標籤 y。
    回傳 (X: DataFrame, y: Series)。X 絕不包含任何未來資訊。
    """
    X = labeled[config.FEATURE_COLS].copy()
    y = labeled["label"].copy()
    return X, y


# ---------------------------------------------------------------------------
# 訓練(時序切分,不 shuffle)
# ---------------------------------------------------------------------------
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    時序切分(time-series split):前 TRAIN_RATIO 訓練、其餘驗證,絕不打散順序。
    RandomForestClassifier(class_weight='balanced' 處理類別不平衡)。
    回傳 (model, val_accuracy)。
    """
    n = len(X)
    split = int(n * config.TRAIN_RATIO)
    # 保底:資料太少時至少留 1 筆驗證
    split = min(max(split, 1), n - 1) if n > 1 else n

    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        class_weight="balanced",          # 平衡各訊號類別的權重
        random_state=config.RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 驗證準確率(若驗證集為空則回傳訓練準確率)
    if len(X_val) > 0:
        val_acc = accuracy_score(y_val, model.predict(X_val))
    else:
        val_acc = accuracy_score(y_train, model.predict(X_train))

    return model, float(val_acc)


# ---------------------------------------------------------------------------
# 預測
# ---------------------------------------------------------------------------
def apply_trend_filter(signals: pd.Series, feat_df: pd.DataFrame) -> pd.Series:
    """
    順勢濾網(trade with the trend)—— 解決「多頭趨勢中過早出場、長期輸給買進持有」。
    依 config.TREND_STYLE 取對照表,把原始訊號在「多頭/空頭排列」下重新映射:
      多頭排列(SMA_fast > SMA_slow,feat['trend_up']==1):整體偏多、充分參與上漲。
      空頭排列(trend_up==0):封頂在續抱、不加碼,讓 0/1 得以減倉避開下跌。
    可由 config.USE_TREND_FILTER 關閉做 A/B 對照。回傳過濾後的訊號 Series。
    """
    if not getattr(config, "USE_TREND_FILTER", False) or "trend_up" not in feat_df:
        return signals
    up = feat_df["trend_up"].reindex(signals.index).fillna(0).astype(int).values
    s = signals.values.copy()
    # 依風格取對照表:多頭排列充分參與上漲,空頭排列封頂續抱、讓 0/1 得以減倉。
    maps = config.TREND_MAPS.get(config.TREND_STYLE, config.TREND_MAPS["defensive"])
    up_map, dn_map = maps["up"], maps["dn"]
    out = [up_map[v] if up[i] else dn_map[v] for i, v in enumerate(s)]
    return pd.Series(out, index=signals.index, name=signals.name)


def predict_series(model, feat_df: pd.DataFrame, trend_filter: bool = True) -> pd.Series:
    """
    整段歷史訊號:回傳與 feat_df 同索引的訊號碼 Series(0~4)。
    trend_filter=True 時套用 apply_trend_filter(順勢濾網)。
    """
    X = feat_df[config.FEATURE_COLS]
    preds = model.predict(X)
    sig = pd.Series(preds, index=feat_df.index, name="signal").astype(int)
    if trend_filter:
        sig = apply_trend_filter(sig, feat_df)
    return sig


def predict_latest(model, feat_df: pd.DataFrame, trend_filter: bool = True) -> int:
    """最新一日訊號:回傳最後一列的訊號碼(int 0~4),預設套用順勢濾網。"""
    sig = predict_series(model, feat_df, trend_filter=trend_filter)
    return int(sig.iloc[-1])


if __name__ == "__main__":
    # 簡易自測
    from core.data_pipeline import seed_sample_data, build_features
    seed_sample_data("TEST")
    feat = build_features("TEST")
    lab = label_data(feat)
    X, y = prepare_xy(lab)
    m, acc = train_model(X, y)
    print("label dist:\n", y.value_counts().sort_index())
    print("val_acc:", round(acc, 4))
    print("latest signal:", predict_latest(m, feat))
