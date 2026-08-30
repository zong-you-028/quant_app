# -*- coding: utf-8 -*-
"""
evaluation.py - 模組五:三階段績效評估管道 (Performance Evaluation Pipeline)
全部為純函式,吃 pandas Series/array,可直接套在任一 Walk-Forward fold 上。
所有金融指標一律以「淨報酬」計算(扣 0.003 / 單位換手摩擦成本)。

三階段:
  ① 機器學習訊號:Precision@(Label=4)、Rank IC(Spearman)、Meta Log-Loss
  ② 金融回測 vs 006208(富邦台50):年化/Sharpe/MDD/Alpha/Information Ratio + 對比圖
  ③ 穩健性:IS→OOS 夏普衰減率、換手率(Turnover)

★ 防 Data Leakage:本檔只「評估」已產生的訊號/報酬,不回看未來;
  指標計算嚴格按時間索引對齊,benchmark 一律 reindex 到策略索引再比較。
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sklearn.metrics import log_loss

import config

PERIODS_PER_YEAR = 252        # 台股年交易日約 252


# ===========================================================================
# 成本 / 報酬 / 權益 基礎工具
# ===========================================================================
def apply_turnover_costs(gross_ret: pd.Series, position: pd.Series,
                         cost: float = None):
    """
    依倉位變動扣換手成本:net = gross − |Δposition| × cost。
    回傳 (net_ret, turnover_series)。首日成本以「初始建倉」|pos0| 計。
    """
    cost = config.COST_PER_TURNOVER if cost is None else cost
    pos = position.reindex(gross_ret.index).fillna(0.0)
    turn = pos.diff().abs()
    if len(turn):
        turn.iloc[0] = abs(float(pos.iloc[0]))
    net = gross_ret.fillna(0.0) - turn * cost
    return net, turn


def turnover_rate(position: pd.Series, annualize: bool = True) -> float:
    """換手率 = Σ|Δposition|;annualize=True 時換算成「每年」平均換手。"""
    pos = position.fillna(0.0)
    turn = pos.diff().abs()
    if len(turn):
        turn.iloc[0] = abs(float(pos.iloc[0]))
    total = float(turn.sum())
    if annualize and len(pos) > 1:
        years = len(pos) / PERIODS_PER_YEAR
        return total / years if years > 0 else total
    return total


def equity_curve(net_ret: pd.Series) -> pd.Series:
    """淨報酬 → 累積權益曲線(起點隱含 1.0)。"""
    return (1.0 + net_ret.fillna(0.0)).cumprod()


def annualized_return(net_ret: pd.Series) -> float:
    """幾何年化報酬(CAGR)。"""
    r = net_ret.dropna()
    if len(r) < 2:
        return 0.0
    base = float((1.0 + r).prod())
    years = len(r) / PERIODS_PER_YEAR
    return base ** (1.0 / years) - 1.0 if (years > 0 and base > 0) else base - 1.0


def annualized_sharpe(net_ret: pd.Series, rf: float = 0.0,
                      periods: int = PERIODS_PER_YEAR) -> float:
    """年化夏普 = (日超額報酬均值 / 標準差) × sqrt(periods)。"""
    r = net_ret.dropna() - rf / periods
    sd = r.std()
    if sd == 0 or not np.isfinite(sd) or len(r) < 2:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤(負值)。"""
    if equity is None or len(equity) == 0:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def active_return(strat_ret: pd.Series, bench_ret: pd.Series) -> float:
    """超額(Alpha)= 策略年化報酬 − 標的年化報酬(以對齊後區段計)。"""
    s, b = _align(strat_ret, bench_ret)
    return annualized_return(s) - annualized_return(b)


def information_ratio(strat_ret: pd.Series, bench_ret: pd.Series,
                      periods: int = PERIODS_PER_YEAR) -> float:
    """
    資訊比率 = 年化超額報酬 / 追蹤誤差。
    以日頻主動報酬 a=策略−標的:IR = mean(a)/std(a) × sqrt(periods)。
    """
    s, b = _align(strat_ret, bench_ret)
    a = (s - b).dropna()
    te = a.std()
    if te == 0 or not np.isfinite(te) or len(a) < 2:
        return 0.0
    return float(a.mean() / te * np.sqrt(periods))


def tracking_error(strat_ret: pd.Series, bench_ret: pd.Series,
                   periods: int = PERIODS_PER_YEAR) -> float:
    """年化追蹤誤差 = std(策略−標的日報酬) × sqrt(periods)。"""
    s, b = _align(strat_ret, bench_ret)
    a = (s - b).dropna()
    return float(a.std() * np.sqrt(periods)) if len(a) > 1 else 0.0


def _align(a: pd.Series, b: pd.Series):
    """把兩條報酬序列對齊到共同索引(交集),避免錯期比較。"""
    idx = a.index.intersection(b.index)
    return a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0)


# ===========================================================================
# ① 機器學習訊號指標
# ===========================================================================
def precision_at_class(y_true, y_pred, target: int = 4) -> float:
    """
    指定類別的 Precision = TP / (TP + FP)。
    預設 target=4(強烈買進):在所有「被預測為 4」中,真的是 4 的比例。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred_pos = (y_pred == target)
    tp = int(np.sum(pred_pos & (y_true == target)))
    fp = int(np.sum(pred_pos & (y_true != target)))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def rank_ic(pred_score, fwd_ret, method: str = "spearman") -> float:
    """
    Rank IC = 預測分數與「實際未來報酬」的等級相關(Spearman)。
    衡量模型「排序能力」:>0 代表分數越高、實際報酬越好。
    """
    s = pd.Series(np.asarray(pred_score, dtype=float)).reset_index(drop=True)
    r = pd.Series(np.asarray(fwd_ret, dtype=float)).reset_index(drop=True)
    d = pd.concat([s, r], axis=1).dropna()
    if len(d) < 3:
        return 0.0
    ic = d.iloc[:, 0].corr(d.iloc[:, 1], method=method)
    return float(ic) if np.isfinite(ic) else 0.0


def meta_log_loss(y_true, prob_success, eps: float = 1e-15) -> float:
    """
    第二層 Meta-Model 的 Log-Loss(二元:1=基層訊號獲利、0=失敗)。
    機率先夾在 [eps, 1−eps];單一類別時手動計算避免 sklearn 報錯。
    """
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(prob_success, dtype=float), eps, 1 - eps)
    if len(np.unique(y)) < 2:
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return float(log_loss(y, p, labels=[0, 1]))


# ===========================================================================
# ② 金融回測 vs 006208 + 對比圖
# ===========================================================================
def load_benchmark_returns(symbol: str = "006208") -> pd.Series:
    """載入對標 ETF(預設 006208 富邦台50)日報酬;無資料回空 Series。"""
    try:
        from core.data_pipeline import ensure_data, load_ohlcv
        try:
            ensure_data(symbol)
        except Exception:
            pass
        df = load_ohlcv(symbol)
        if df is not None and not df.empty:
            return df["close"].pct_change().dropna().rename(symbol)
    except Exception:
        pass
    return pd.Series(dtype=float, name=symbol)


def financial_report(strat_net_ret: pd.Series, bench_ret: pd.Series,
                     position: pd.Series = None) -> dict:
    """策略(已扣成本的淨報酬)對標 benchmark 的整組金融指標。"""
    s, b = _align(strat_net_ret, bench_ret)
    s_eq, b_eq = equity_curve(s), equity_curve(b)
    return {
        "strat_cagr": annualized_return(s),
        "bench_cagr": annualized_return(b),
        "strat_sharpe": annualized_sharpe(s),
        "bench_sharpe": annualized_sharpe(b),
        "strat_mdd": max_drawdown(s_eq),
        "bench_mdd": max_drawdown(b_eq),
        "alpha": active_return(s, b),
        "tracking_error": tracking_error(s, b),
        "information_ratio": information_ratio(s, b),
        "turnover": turnover_rate(position) if position is not None else None,
        "strat_equity": s_eq,
        "bench_equity": b_eq,
    }


def plot_equity_vs_benchmark(strat_equity: pd.Series, bench_equity: pd.Series,
                             title: str = "Strategy (net) vs 006208 Buy&Hold",
                             save_path: str = None) -> Figure:
    """把策略淨權益曲線與 006208 買進持有畫在同一張圖(Agg,thread-safe)。"""
    se, be = _align(strat_equity, bench_equity)
    fig = Figure(figsize=(7.2, 3.4), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(se.index, se.values, color="#1565C0", lw=1.7, label="Strategy (net)")
    ax.plot(be.index, be.values, color="#9E9E9E", lw=1.4, ls="--", label="006208 B&H")
    ax.axhline(1.0, color="#BDBDBD", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Equity (×)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    if save_path:
        FigureCanvasAgg(fig).print_png(save_path)
    return fig


# ===========================================================================
# ③ 穩健性:IS→OOS 夏普衰減 + 換手
# ===========================================================================
def sharpe_decay(is_sharpe: float, oos_sharpe: float) -> float:
    """
    夏普衰減率 = 1 − OOS_Sharpe / IS_Sharpe。
    0=無衰減;接近 1=樣本外幾乎失效(過擬合警訊);可能 >1(OOS 轉負)。
    """
    if is_sharpe == 0 or not np.isfinite(is_sharpe):
        return 0.0
    return float(1.0 - oos_sharpe / is_sharpe)


def robustness_report(is_net_ret: pd.Series, oos_net_ret: pd.Series,
                      is_position: pd.Series = None,
                      oos_position: pd.Series = None) -> dict:
    """訓練段(IS)與驗證段(OOS)的夏普與衰減率 + 各自換手率。"""
    is_sr = annualized_sharpe(is_net_ret)
    oos_sr = annualized_sharpe(oos_net_ret)
    return {
        "is_sharpe": is_sr,
        "oos_sharpe": oos_sr,
        "sharpe_decay": sharpe_decay(is_sr, oos_sr),
        "is_turnover": turnover_rate(is_position) if is_position is not None else None,
        "oos_turnover": turnover_rate(oos_position) if oos_position is not None else None,
    }


# ===========================================================================
# 彙整列印
# ===========================================================================
def format_report(ml: dict = None, fin: dict = None, rob: dict = None) -> str:
    """把三階段結果排成可讀字串(供 CLI / log)。"""
    lines = []
    if ml:
        lines.append("【① ML 訊號】"
                     f" Precision@4={ml.get('precision_at_4', 0):.3f}"
                     f"  RankIC={ml.get('rank_ic', 0):+.3f}"
                     f"  MetaLogLoss={ml.get('meta_log_loss', 0):.4f}")
    if fin:
        lines.append("【② 金融 vs 006208】"
                     f" 策略 CAGR={fin['strat_cagr']*100:+.1f}%"
                     f" (006208 {fin['bench_cagr']*100:+.1f}%)"
                     f"  Sharpe={fin['strat_sharpe']:.2f}"
                     f" (vs {fin['bench_sharpe']:.2f})")
        lines.append(f"            MDD={fin['strat_mdd']*100:.1f}%"
                     f" (vs {fin['bench_mdd']*100:.1f}%)"
                     f"  Alpha={fin['alpha']*100:+.1f}%"
                     f"  IR={fin['information_ratio']:.2f}"
                     + (f"  Turnover={fin['turnover']:.1f}x/yr"
                        if fin.get('turnover') is not None else ""))
    if rob:
        lines.append("【③ 穩健性】"
                     f" IS Sharpe={rob['is_sharpe']:.2f}"
                     f"  OOS Sharpe={rob['oos_sharpe']:.2f}"
                     f"  衰減={rob['sharpe_decay']*100:.0f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    # 煙霧測試:用合成報酬驗證指標管線可運作
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-01", periods=500)
    strat = pd.Series(rng.normal(0.0007, 0.012, 500), index=idx)
    bench = pd.Series(rng.normal(0.0004, 0.011, 500), index=idx)
    pos = pd.Series(rng.integers(0, 2, 500).astype(float), index=idx)
    net, _ = apply_turnover_costs(strat, pos)
    fin = financial_report(net, bench, pos)
    ml = {
        "precision_at_4": precision_at_class([4, 2, 4, 0], [4, 4, 4, 0], 4),
        "rank_ic": rank_ic([4, 3, 2, 1, 0], [0.05, 0.02, 0.0, -0.01, -0.03]),
        "meta_log_loss": meta_log_loss([1, 0, 1, 1], [0.8, 0.3, 0.6, 0.9]),
    }
    rob = robustness_report(net.iloc[:400], net.iloc[400:], pos.iloc[:400], pos.iloc[400:])
    print(format_report(ml, fin, rob))
    plot_equity_vs_benchmark(fin["strat_equity"], fin["bench_equity"],
                             save_path=None)
    print("plot OK; precision@4 =", ml["precision_at_4"])
