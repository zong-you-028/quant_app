# -*- coding: utf-8 -*-
"""
wfa_backtest.py - 收口:Walk-Forward 回測引擎(串接模組一~四,出模組五報表)
把五個模組串成一條嚴謹的滾動時間序列管線(WFA):

  每個 fold:
    訓練段(已 Purge 淨化) → 三欄標籤(模組一)
      → 基層 LGBM(模組二,跨股池化訓練,用橫截面特徵)
      → Meta XGBoost(模組三)
      → HMM 市場狀態(模組四,只用訓練段配適)
    驗證段(OOS) → 基層分數/訊號 → Meta P(success)
      → 凱利倉位 × P(bull) → 每檔部位 → 扣 0.003 成本
  串接所有 fold 的 OOS 淨報酬 → 模組五三階段報表 + vs 006208 對比圖。

★ 防 Data Leakage(三道防線):
  1) 特徵全為當下/過去(模組一~四已保證);
  2) 模型/HMM/凱利賠率/門檻只用「訓練段」配適,套 OOS;
  3) Purge:三欄標籤用到未來 FUTURE_N 天,故訓練段與驗證段之間隔離 WFA_PURGE_DAYS
     天(剔除尾端標籤會探進驗證段的訓練樣本),這是 WFA 最常見也最致命的洩漏點。

組合假設:多檔等權(各檔凱利倉位 0~1,組合日報酬 = 各檔策略淨報酬的橫截面平均)。
"""
import numpy as np
import pandas as pd

import config
from core.features_cs import build_feature_panel
from core.labeling import triple_barrier_labels
from core.base_model import LGBMBaseModel, prepare_xy
from core.meta_model import MetaModel
from core.regime import RegimeHMM, market_index_returns
from core import evaluation as ev


# ---------------------------------------------------------------------------
# 資料準備:panel(含橫截面)→ 每檔 (X, y, ret_event, daily_ret)
# ---------------------------------------------------------------------------
def build_wfa_dataset(symbols=None):
    """組裝 WFA 所需資料:特徵 panel、每檔的標籤/特徵/事件報酬/日報酬、大盤報酬。"""
    panel = build_feature_panel(symbols, with_cs=True)
    data = {}
    for s, feat in panel.items():
        lab = triple_barrier_labels(feat)
        X, y, ret_event = prepare_xy(feat, lab)
        if len(X) == 0:
            continue
        data[s] = {
            "X": X, "y": y, "ret_event": ret_event,
            "daily_ret": feat["ret"].reindex(X.index).fillna(0.0),
        }
    if not data:
        raise RuntimeError("無可用資料(標籤後全空)")
    mkt = market_index_returns(panel)
    return panel, data, mkt


# ---------------------------------------------------------------------------
# 切分:在「所有股票的聯集時間軸」上產生 (train, test) 區段(含 Purge)
# ---------------------------------------------------------------------------
def make_folds(dates: pd.DatetimeIndex, train_min: int, test_days: int,
               purge: int, expanding: bool, train_window: int):
    """回傳 [(train_lo, train_hi, test_lo, test_hi)] 的整數位置區段(半開區間)。"""
    n = len(dates)
    folds = []
    start = train_min
    while start < n:
        test_lo, test_hi = start, min(start + test_days, n)
        train_hi = test_lo - purge                      # Purge:隔離標籤水平
        train_lo = 0 if expanding else max(0, train_hi - train_window)
        if train_hi - train_lo > 0:
            folds.append((train_lo, train_hi, test_lo, test_hi))
        start += test_days
    return folds


def _mask(idx: pd.DatetimeIndex, lo_date, hi_date) -> pd.Series:
    """日期落在 [lo_date, hi_date] 的布林遮罩(含端點)。"""
    return (idx >= lo_date) & (idx <= hi_date)


def _stack(data, symbols, lo_date, hi_date):
    """把各股在 [lo,hi] 區間的 (X, y, ret_event) 跨股堆疊成池化訓練集。"""
    Xs, ys, rs = [], [], []
    for s in symbols:
        d = data[s]
        m = _mask(d["X"].index, lo_date, hi_date)
        if m.any():
            Xs.append(d["X"][m]); ys.append(d["y"][m]); rs.append(d["ret_event"][m])
    if not Xs:
        return None, None, None
    return pd.concat(Xs), pd.concat(ys), pd.concat(rs)


def _backtest_symbol(pos: pd.Series, daily_ret: pd.Series, cost: float):
    """單檔:部位 pos 於 d 決定、earns ret 於 d+1;扣換手成本。回傳淨報酬序列。"""
    pos = pos.fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * daily_ret.reindex(pos.index).fillna(0.0)
    turn = pos.diff().abs()
    if len(turn):
        turn.iloc[0] = abs(float(pos.iloc[0]))
    return strat - turn * cost, turn


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_legacy_wfa(symbols=None, train_min=None, test_days=None, purge=None,
            expanding=None, benchmark=None, make_plot=False, verbose=True):
    """執行 Walk-Forward 回測,回傳含三階段報表與 OOS 權益曲線的 dict。"""
    symbols = symbols or config.WATCHLIST
    train_min = train_min or config.WFA_TRAIN_MIN_DAYS
    test_days = test_days or config.WFA_TEST_DAYS
    purge = config.WFA_PURGE_DAYS if purge is None else purge
    expanding = config.WFA_EXPANDING if expanding is None else expanding
    benchmark = benchmark or config.BENCHMARK_SYMBOL
    cost = config.COST_PER_TURNOVER

    panel, data, mkt = build_wfa_dataset(symbols)
    syms = list(data.keys())

    # 聯集時間軸
    all_dates = pd.DatetimeIndex(sorted(set().union(
        *[data[s]["X"].index for s in syms])))
    folds = make_folds(all_dates, train_min, test_days, purge,
                       expanding, config.WFA_TRAIN_WINDOW)
    if not folds:
        raise RuntimeError(f"資料長度不足以產生 fold(共 {len(all_dates)} 天,"
                           f"需 > train_min={train_min})")

    # 累積容器
    oos_net = {s: [] for s in syms}        # 各 fold 的 OOS 淨報酬(每檔)
    ml = {"y": [], "pred": [], "score": [], "ret": [],
          "meta_y": [], "meta_p": []}       # 池化 ML 指標素材
    is_sr, oos_sr, turn_tot = [], [], 0.0

    for fi, (tl, th, sl, sh) in enumerate(folds):
        tr_lo, tr_hi = all_dates[tl], all_dates[th - 1]      # 訓練段(已 purge)
        te_lo, te_hi = all_dates[sl], all_dates[sh - 1]      # 驗證段(OOS)

        # --- 1) 池化訓練基層 + Meta ---
        Xtr, ytr, rtr = _stack(data, syms, tr_lo, tr_hi)
        if Xtr is None or len(Xtr) < config.WFA_MIN_TRAIN_SAMPLES:
            continue
        base = LGBMBaseModel().fit(Xtr, ytr)
        score_tr, pred_tr = base.score(Xtr), base.predict(Xtr)
        meta = MetaModel().fit(Xtr, score_tr, pred_tr, rtr)

        # --- 2) HMM 只用訓練段配適;因果 P(bull) 套全序列(每點僅看過去)---
        rg = RegimeHMM().fit(mkt[_mask(mkt.index, tr_lo, tr_hi)])
        pbull = rg.predict_bull_proba(mkt, causal=True)

        # --- 3) 逐檔在 OOS 預測 → 凱利倉位 × P(bull) → 回測 ---
        fold_oos, fold_is = [], []
        for s in syms:
            d = data[s]
            mte = _mask(d["X"].index, te_lo, te_hi)
            if mte.any():
                Xte = d["X"][mte]
                sc, pr = base.score(Xte), base.predict(Xte)
                pos = meta.size(Xte, sc, pr) * pbull.reindex(Xte.index).fillna(1.0)
                net, turn = _backtest_symbol(pos, d["daily_ret"], cost)
                oos_net[s].append(net)
                fold_oos.append(net)
                turn_tot += float(turn.sum())
                # 池化 ML 素材
                ml["y"].append(d["y"][mte]); ml["pred"].append(pr)
                ml["score"].append(sc); ml["ret"].append(d["ret_event"][mte])
                lm = (pr == config.META_TARGET_SIGNAL) & d["ret_event"][mte].notna()
                if lm.any():
                    ml["meta_y"].append((d["ret_event"][mte][lm] > 0).astype(int))
                    ml["meta_p"].append(meta.predict_proba(Xte, sc)[lm])
            # In-Sample(訓練段)策略報酬,供穩健性衰減比較
            mtr = _mask(d["X"].index, tr_lo, tr_hi)
            if mtr.any():
                Xi = d["X"][mtr]
                pos_i = (meta.size(Xi, base.score(Xi), base.predict(Xi))
                         * pbull.reindex(Xi.index).fillna(1.0))
                net_i, _ = _backtest_symbol(pos_i, d["daily_ret"], cost)
                fold_is.append(net_i)

        # 每 fold 的 IS / OOS 組合夏普(等權跨股平均)
        if fold_is:
            is_sr.append(ev.annualized_sharpe(
                pd.concat(fold_is, axis=1).mean(axis=1)))
        if fold_oos:
            oos_sr.append(ev.annualized_sharpe(
                pd.concat(fold_oos, axis=1).mean(axis=1)))
        if verbose:
            print(f"  fold {fi+1}/{len(folds)}: train<={tr_hi.date()} "
                  f"test {te_lo.date()}~{te_hi.date()} "
                  f"({len(Xtr)} 訓練樣本)")

    # --- 組合 OOS 權益:各檔串接 → 橫截面等權平均 ---
    per_sym = {s: pd.concat(oos_net[s]).sort_index()
               for s in syms if oos_net[s]}
    port_ret = pd.DataFrame(per_sym).mean(axis=1).sort_index()

    # --- benchmark:006208;抓不到退回等權大盤代理 ---
    bench = ev.load_benchmark_returns(benchmark)
    bench_label = benchmark
    if bench.empty:
        bench = mkt.rename("equal_weight_mkt")
        bench_label = "等權大盤代理"
    bench = bench.reindex(port_ret.index).fillna(0.0)

    # --- 模組五三階段 ---
    y = pd.concat(ml["y"]); pred = pd.concat(ml["pred"])
    score = pd.concat(ml["score"]); ret = pd.concat(ml["ret"])
    ml_report = {
        "precision_at_4": ev.precision_at_class(y, pred, config.META_TARGET_SIGNAL),
        "rank_ic": ev.rank_ic(score, ret),
        "meta_log_loss": (ev.meta_log_loss(pd.concat(ml["meta_y"]),
                                           pd.concat(ml["meta_p"]))
                          if ml["meta_y"] else float("nan")),
    }
    fin_report = ev.financial_report(port_ret, bench)
    rob_report = {
        "is_sharpe": float(np.mean(is_sr)) if is_sr else 0.0,
        "oos_sharpe": float(np.mean(oos_sr)) if oos_sr else 0.0,
        "sharpe_decay": ev.sharpe_decay(float(np.mean(is_sr)) if is_sr else 0.0,
                                        float(np.mean(oos_sr)) if oos_sr else 0.0),
        "turnover": turn_tot / (len(port_ret) / ev.PERIODS_PER_YEAR)
        if len(port_ret) else 0.0,
    }
    fin_report["turnover"] = rob_report["turnover"]

    fig = None
    if make_plot:
        fig = ev.plot_equity_vs_benchmark(
            fin_report["strat_equity"], fin_report["bench_equity"],
            title=f"WFA Strategy (net) vs {bench_label}")

    result = {
        "n_folds": len(folds), "symbols": syms, "benchmark": bench_label,
        "ml": ml_report, "fin": fin_report, "rob": rob_report,
        "oos_equity": fin_report["strat_equity"],
        "bench_equity": fin_report["bench_equity"],
        "report_text": ev.format_report(ml_report, fin_report, rob_report),
        "figure": fig,
    }
    if verbose:
        print("\n" + result["report_text"])
    return result

# Legacy pipeline note: run_legacy_wfa is retained only for retrospective comparison.
# All normal callers use the conservative, gate-enforced implementation below.
def run_wfa(symbols=None, train_min=None, test_days=None, purge=None,
            expanding=None, benchmark=None, make_plot=False, verbose=True):
    """預設 WFA：保守 AI，未通過四道 OOS 門檻即不可視為交易模型。

    舊版 Meta/HMM 管線因嚴重 IS→OOS 衰減不再是預設。為防止以舊參數悄悄
    重啟它，僅保留 symbols 與 verbose；其他 legacy 參數會明確拒絕。
    """
    if any(value is not None for value in (train_min, test_days, purge, expanding, benchmark)) or make_plot:
        raise ValueError("舊版 WFA 參數已停用；請使用 run_legacy_wfa 做純研究對照。")
    from core.conservative_ai import format_relative_alpha_report, run_relative_alpha_wfa
    result = run_relative_alpha_wfa(symbols=symbols, verbose=verbose)
    result["oos_equity"] = result["fin"]["strat_equity"]
    result["bench_equity"] = result["fin"]["bench_equity"]
    result["report_text"] = format_relative_alpha_report(result)
    if verbose:
        print("\n" + result["report_text"])
    return result


if __name__ == "__main__":
    # 端到端離線驗證:多檔合成(較長序列以產生多個 fold)→ 完整 WFA 管線 → 報表
    from core.data_pipeline import seed_sample_data
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    for i, s in enumerate(syms):
        seed_sample_data(s, n_days=900, seed=51 + i)
    res = run_wfa(symbols=syms, verbose=True)
    print("\nfold 數:", res["n_folds"], "| 對標:", res["benchmark"])
    print("OOS 權益末值:", round(float(res["oos_equity"].iloc[-1]), 3),
          "| benchmark 末值:", round(float(res["bench_equity"].iloc[-1]), 3))
