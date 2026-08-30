# -*- coding: utf-8 -*-
"""
cpcv.py - Prong B:Combinatorial Purged Cross-Validation(把 Rank IC 推離 0)
針對「個股方向預測」的 base 訊號,用 CPCV 做最嚴謹的樣本外評估:
  - 把時間切成 N 個連續區塊,每次取 k 個區塊當「驗證」-> 共 C(N,k) 條獨立路徑,
    得到 OOS 表現的「分布」(而非單一 WFA 路徑),對過擬合與運氣最不留情。
  - Purge + 拉長 Embargo:剔除「標籤水平會探進驗證區」與「驗證後禁制帶」的訓練樣本。
  - 每個 fold 在「訓練段」用單變量 |Rank IC| 選前 N 個特徵(29 -> 8),避免雜訊特徵。
  - 強正規化 LightGBM(淺樹/大葉樣本/L1L2/子採樣)對抗過擬合。
目標:看「精簡+正規化」能不能把 OOS Rank IC 從 ≈0 推到穩定為正,而非追 In-Sample 分數。

★ 防 Data Leakage:特徵選擇、模型、門檻全在「訓練段」完成;Purge/Embargo 隔離標籤水平。
"""
import itertools

import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from core.wfa_backtest import build_wfa_dataset
from core import evaluation as ev

_LABEL_TO_DIR = {0: -1.0, 2: 0.0, 4: 1.0}


# ---------------------------------------------------------------------------
# 單變量特徵選擇:訓練段 |Rank IC| 最高的前 N 個
# ---------------------------------------------------------------------------
def select_features(X: pd.DataFrame, ret_event: pd.Series,
                    n: int = None, candidates=None):
    """回傳 (選中的特徵 list, {特徵: |IC|})。只用訓練段資料,無洩漏。"""
    candidates = candidates or config.ALL_FEATURE_COLS
    ics = {}
    r = ret_event.values
    for c in candidates:
        if c in X.columns:
            ics[c] = abs(ev.rank_ic(X[c].values, r))
    top = sorted(ics, key=ics.get, reverse=True)[: (n or config.N_SELECTED_FEATURES)]
    return top, ics


# ---------------------------------------------------------------------------
# CPCV 區塊與 Purge/Embargo
# ---------------------------------------------------------------------------
def _group_bounds(n: int, n_groups: int):
    """把 0..n-1 切成 n_groups 個連續區塊,回傳 [(lo, hi)](半開)。"""
    return [(int(b[0]), int(b[-1]) + 1) for b in np.array_split(np.arange(n), n_groups)]


def _train_positions(n, test_blocks, purge_before, embargo_after):
    """訓練位置 = 全部 - 驗證區塊 - (各驗證區塊前 purge / 後 embargo 的禁制帶)。"""
    forbidden = np.zeros(n, dtype=bool)
    for (lo, hi) in test_blocks:
        forbidden[lo:hi] = True                                   # 驗證區本身
        forbidden[max(0, lo - purge_before):lo] = True            # 前向 purge(標籤水平)
        forbidden[hi:min(n, hi + embargo_after)] = True           # 後向 embargo
    return ~forbidden


def _stack_by_dates(data, symbols, date_index):
    """把各股在 date_index 上的 (X, y, ret_event) 跨股堆疊。"""
    Xs, ys, rs = [], [], []
    for s in symbols:
        idx = data[s]["X"].index
        m = idx.isin(date_index)
        if m.any():
            Xs.append(data[s]["X"][m]); ys.append(data[s]["y"][m])
            rs.append(data[s]["ret_event"][m])
    if not Xs:
        return None, None, None
    return pd.concat(Xs), pd.concat(ys), pd.concat(rs)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_cpcv(data=None, mkt=None, symbols=None, n_groups=None, test_groups=None,
             select_n=None, params=None, label="", verbose=True):
    """
    執行 CPCV。select_n=None 表示用全部特徵;否則每 fold 選前 select_n 個。
    回傳 dict:每條路徑的 test Rank IC / Precision@4 分布 + 選中特徵頻率。
    """
    if data is None:
        _, data, mkt = build_wfa_dataset(symbols)
    symbols = list(data.keys())
    n_groups = n_groups or config.CPCV_N_GROUPS
    test_groups = test_groups or config.CPCV_TEST_GROUPS
    params = dict(params or config.LGBM_PARAMS)
    purge_before = config.FUTURE_N
    embargo_after = config.CPCV_EMBARGO_DAYS

    dates = pd.DatetimeIndex(sorted(set().union(
        *[data[s]["X"].index for s in symbols])))
    n = len(dates)
    bounds = _group_bounds(n, n_groups)

    ics, precs, base4, feat_count = [], [], [], {}
    combos = list(itertools.combinations(range(n_groups), test_groups))
    for gi in combos:
        test_blocks = [bounds[g] for g in gi]
        train_mask = _train_positions(n, test_blocks, purge_before, embargo_after)
        train_dates = dates[train_mask]
        test_pos = np.zeros(n, dtype=bool)
        for (lo, hi) in test_blocks:
            test_pos[lo:hi] = True
        test_dates = dates[test_pos]

        Xtr, ytr, rtr = _stack_by_dates(data, symbols, train_dates)
        Xte, yte, rte = _stack_by_dates(data, symbols, test_dates)
        if Xtr is None or Xte is None or len(Xtr) < config.WFA_MIN_TRAIN_SAMPLES:
            continue

        # 特徵選擇(訓練段)
        if select_n:
            feats, _ = select_features(Xtr, rtr, n=select_n)
        else:
            feats = [c for c in config.ALL_FEATURE_COLS if c in Xtr.columns]
        for f in feats:
            feat_count[f] = feat_count.get(f, 0) + 1

        # 訓練(迴歸方向分數)+ OOS 預測
        ydir = ytr.map(_LABEL_TO_DIR).astype(float)
        model = lgb.LGBMRegressor(objective="regression", **params)
        model.fit(Xtr[feats], ydir)
        s_tr = model.predict(Xtr[feats])
        lo_thr = np.quantile(s_tr, config.BASE_SIGNAL_LOW_Q)
        hi_thr = np.quantile(s_tr, config.BASE_SIGNAL_HIGH_Q)

        s_te = model.predict(Xte[feats])
        pred = np.full(len(s_te), 2)
        pred[s_te >= hi_thr] = 4
        pred[s_te <= lo_thr] = 0

        ics.append(ev.rank_ic(s_te, rte.values))                 # OOS Rank IC
        precs.append(ev.precision_at_class(yte.values, pred, 4))  # OOS Precision@4
        base4.append(float((yte.values == 4).mean()))             # 驗證集 4 的基率

    ics = np.array(ics); precs = np.array(precs)
    n_paths = len(ics)
    mean_ic, std_ic = float(ics.mean()), float(ics.std())
    tstat = mean_ic / (std_ic / np.sqrt(n_paths)) if std_ic > 0 else 0.0
    out = {
        "label": label, "n_paths": n_paths,
        "mean_ic": mean_ic, "std_ic": std_ic, "tstat": tstat,
        "pct_ic_pos": float((ics > 0).mean()),
        "mean_precision4": float(precs.mean()), "base_rate4": float(np.mean(base4)),
        "ics": ics, "feat_count": feat_count,
    }
    if verbose:
        print(f"[{label}] 路徑 {n_paths}　"
              f"OOS Rank IC 平均={mean_ic:+.4f} ± {std_ic:.4f}"
              f"（粗略 t={tstat:+.2f}, IC>0 比例={out['pct_ic_pos']*100:.0f}%)")
        print(f"          Precision@4={out['mean_precision4']:.3f}"
              f"（基率 {out['base_rate4']:.3f}）")
        top_feats = sorted(feat_count, key=feat_count.get, reverse=True)[:8]
        if select_n:
            print(f"          最常被選中的特徵：{top_feats}")
    return out


if __name__ == "__main__":
    # 一次載入資料,兩種設定跑同一套 CPCV 直接對比是否把 Rank IC 推離 0
    print("載入資料(快取)...")
    _, data, mkt = build_wfa_dataset()
    print(f"標的 {len(data)} 檔　CPCV {config.CPCV_N_GROUPS} 區塊取 "
          f"{config.CPCV_TEST_GROUPS} = C(N,k) 條路徑　"
          f"purge={config.FUTURE_N} embargo={config.CPCV_EMBARGO_DAYS}\n")

    base = run_cpcv(data=data, mkt=mkt, label="基準:全 29 特徵 + 輕正規化",
                    select_n=None, params=config.LGBM_PARAMS)
    print()
    impr = run_cpcv(data=data, mkt=mkt, label="改良:選 8 特徵 + 強正規化",
                    select_n=config.N_SELECTED_FEATURES, params=config.LGBM_PARAMS_REG)

    print("\n[結論]")
    d = impr["mean_ic"] - base["mean_ic"]
    print(f"  Rank IC 平均：{base['mean_ic']:+.4f} -> {impr['mean_ic']:+.4f}"
          f"（{'改善' if d > 0 else '未改善'} {d:+.4f}）")
    verdict = ("OOS Rank IC 仍貼近 0,個股方向預測難有穩定 edge"
               if abs(impr["mean_ic"]) < 0.02 or impr["pct_ic_pos"] < 0.6
               else "OOS Rank IC 穩定為正,精簡+正規化確有幫助")
    print(f"  判定：{verdict}")
