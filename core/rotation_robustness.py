# -*- coding: utf-8 -*-
"""
rotation_robustness.py - Prong A:輪動策略「參數穩健性」嚴謹驗證
輪動(相對強弱動能)本身是因果的(每次換股只用過去動能),沒有 look-ahead;
真正的風險是「參數過擬合」—— 在歷史上挑到剛好最好的 (動能視窗, K, 換股週期)。
本檔用 In-Sample / Out-of-Sample 切分,掃整個參數網格,回答:
  1. 預設參數在 OOS 還站得住嗎?(IS->OOS 夏普衰減)
  2. 「用 IS 選最佳參數」拿到 OOS 會不會崩?(選參過擬合檢驗)
  3. 有沒有「穩健區」——一大片參數都打贏大盤,而不是孤峰?

指標一律用模組五(扣 0.003 成本)。資料用快取 panel(不重抓)。
"""
import itertools

import numpy as np
import pandas as pd

import config
from core.features_cs import build_feature_panel
from core import evaluation as ev


def _panel_to_frames(panel: dict):
    """panel -> (ret_df, close_df):欄=代號,列=日期。"""
    ret_df = pd.DataFrame({s: panel[s]["ret"] for s in panel}).sort_index()
    close_df = pd.DataFrame({s: panel[s]["close"] for s in panel}).reindex(ret_df.index)
    return ret_df, close_df


def rotation_returns(ret_df, close_df, mom_days, top_k, rebal_days, cost=None,
                     abs_mom=False, abs_thresh=0.0, regime_pbull=None):
    """
    純函式輪動回測(與 core/rotation.py 同邏輯):每 rebal_days 取動能前 top_k 等權。
    新增兩道「現金閘門」(預設關,維持原行為):
      abs_mom=True:選中標的若「絕對動能 <= abs_thresh」則該 slot 轉現金(雙動能)。
        ★ 失格 slot 權重除以 top_k(非除以存活數)→ 真的留現金,不硬集中接刀。
      regime_pbull=Series:換股日把該期權重整體乘上 P(bull)(空頭機率高 -> 降曝險)。
    回傳 (策略淨日報酬, 等權大盤日報酬)。因果:換股權重 shift(1) 才乘當日報酬。
    """
    cost = config.COST_PER_TURNOVER if cost is None else cost
    mom_df = close_df.pct_change(mom_days, fill_method=None)
    idx, cols = ret_df.index, ret_df.columns
    weights = pd.DataFrame(np.nan, index=idx, columns=cols)
    for d in idx[::rebal_days]:
        row = mom_df.loc[d].dropna()
        if row.empty:
            continue
        top = row.nlargest(top_k)
        w = pd.Series(0.0, index=cols)
        for s, mv in top.items():
            if abs_mom and mv <= abs_thresh:
                continue                         # 絕對動能不足 -> 該 slot 留現金
            w[s] = 1.0 / top_k
        if regime_pbull is not None:             # regime 現金乘數(整體降曝險)
            w = w * max(0.0, min(1.0, float(regime_pbull.get(d, 1.0))))
        weights.loc[d] = w.values
    weights = weights.ffill().fillna(0.0)

    port = (weights.shift(1).fillna(0.0) * ret_df.fillna(0.0)).sum(axis=1)
    turn = weights.diff().abs().sum(axis=1)
    if len(turn):
        turn.iloc[0] = weights.iloc[0].abs().sum()
    net = port - turn * cost
    mkt = ret_df.fillna(0.0).mean(axis=1)
    return net, mkt


def run_rotation_robustness(symbols=None, mom_grid=None, k_grid=None,
                            rebal_grid=None, is_fraction=None, verbose=True):
    """掃參數網格,輸出每組的 IS/OOS 夏普與衰減,並做選參過擬合檢驗。"""
    symbols = symbols or config.WATCHLIST
    mom_grid = mom_grid or config.ROT_ROBUST_MOM_GRID
    k_grid = k_grid or config.ROT_ROBUST_K_GRID
    rebal_grid = rebal_grid or config.ROT_ROBUST_REBAL_GRID
    is_fraction = is_fraction or config.ROT_ROBUST_IS_FRACTION

    panel = build_feature_panel(symbols, with_cs=False)   # 輪動只要 ret/close
    ret_df, close_df = _panel_to_frames(panel)
    idx = ret_df.index
    split = int(len(idx) * is_fraction)
    is_idx, oos_idx = idx[:split], idx[split:]

    # 大盤(等權)OOS 基準
    mkt_full = ret_df.fillna(0.0).mean(axis=1)
    mkt_oos_sr = ev.annualized_sharpe(mkt_full.reindex(oos_idx))
    mkt_oos_cagr = ev.annualized_return(mkt_full.reindex(oos_idx))

    rows = []
    for mom, k, rebal in itertools.product(mom_grid, k_grid, rebal_grid):
        net, _ = rotation_returns(ret_df, close_df, mom, k, rebal)
        is_net, oos_net = net.reindex(is_idx), net.reindex(oos_idx)
        is_sr = ev.annualized_sharpe(is_net)
        oos_sr = ev.annualized_sharpe(oos_net)
        rows.append({
            "mom": mom, "k": k, "rebal": rebal,
            "is_sharpe": is_sr, "oos_sharpe": oos_sr,
            "decay": ev.sharpe_decay(is_sr, oos_sr),
            "oos_cagr": ev.annualized_return(oos_net),
            "beat_mkt_oos": ev.annualized_sharpe(oos_net) > mkt_oos_sr,
        })
    res = pd.DataFrame(rows)

    # 選參過擬合檢驗:用 IS 選最佳 -> 看它 OOS
    best_is = res.loc[res["is_sharpe"].idxmax()]
    best_oos = res.loc[res["oos_sharpe"].idxmax()]    # 事後最佳(僅參考)
    default = res[(res["mom"] == config.ROTATION_MOM_DAYS) &
                  (res["k"] == config.ROTATION_TOP_K) &
                  (res["rebal"] == config.ROTATION_REBAL_DAYS)]

    if verbose:
        print(f"資料 {idx[0].date()}~{idx[-1].date()}　IS {len(is_idx)} 天 / "
              f"OOS {len(oos_idx)} 天　大盤 OOS Sharpe={mkt_oos_sr:.2f} "
              f"CAGR={mkt_oos_cagr*100:+.1f}%")
        print(f"網格 {len(res)} 組　|　OOS Sharpe 中位數={res['oos_sharpe'].median():.2f}"
              f"　打贏大盤比例={res['beat_mkt_oos'].mean()*100:.0f}%"
              f"　OOS Sharpe>0 比例={ (res['oos_sharpe']>0).mean()*100:.0f}%")
        print("\n[選參過擬合檢驗]")
        print(f"  IS 最佳參數 mom={int(best_is['mom'])} k={int(best_is['k'])} "
              f"rebal={int(best_is['rebal'])}：IS Sharpe={best_is['is_sharpe']:.2f}"
              f" -> OOS Sharpe={best_is['oos_sharpe']:.2f}"
              f"（衰減 {best_is['decay']*100:.0f}%, OOS CAGR {best_is['oos_cagr']*100:+.1f}%）")
        print(f"  事後 OOS 最佳 mom={int(best_oos['mom'])} k={int(best_oos['k'])} "
              f"rebal={int(best_oos['rebal'])}：OOS Sharpe={best_oos['oos_sharpe']:.2f}（僅參考）")
        if len(default):
            dd = default.iloc[0]
            print(f"  預設參數 mom={int(dd['mom'])} k={int(dd['k'])} rebal={int(dd['rebal'])}："
                  f"IS Sharpe={dd['is_sharpe']:.2f} -> OOS Sharpe={dd['oos_sharpe']:.2f}"
                  f"（衰減 {dd['decay']*100:.0f}%, OOS CAGR {dd['oos_cagr']*100:+.1f}%）")
        # 動能視窗敏感度(固定其餘取中位)
        piv = res.pivot_table(index="mom", values="oos_sharpe", aggfunc="median")
        print("\n[動能視窗 OOS Sharpe 中位數]")
        for m, v in piv["oos_sharpe"].items():
            print(f"  mom={int(m):>3}: {v:+.2f}")

    return {"results": res, "best_is": best_is, "best_oos": best_oos,
            "default": default, "mkt_oos_sharpe": mkt_oos_sr,
            "mkt_oos_cagr": mkt_oos_cagr,
            "is_idx": is_idx, "oos_idx": oos_idx}


def _window_metrics(r: pd.Series) -> dict:
    """單一報酬序列在某視窗的:總報酬 / 年化 / 夏普 / 最大回撤。"""
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return {"total": float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
            "cagr": ev.annualized_return(r), "sharpe": ev.annualized_sharpe(r),
            "mdd": ev.max_drawdown(eq)}


def evaluate_period(symbols=None, start="2021-12-01", end="2022-10-31",
                    combos=None, benchmark=None, grid=True, verbose=True):
    """
    在指定期間(預設 2021/12~2022/10 空頭)測輪動 vs 等權大盤 vs 006208。
    輪動永遠滿倉(無現金閘門),重點看「持有最強的幾檔」是否比大盤『跌得少』。
    部位由「進入該期前」的動能決定(因果);僅擷取該期間實現的報酬。
    """
    symbols = symbols or config.WATCHLIST
    benchmark = benchmark or config.BENCHMARK_SYMBOL
    panel = build_feature_panel(symbols, with_cs=False)
    ret_df, close_df = _panel_to_frames(panel)
    idx = ret_df.index
    win = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(win) == 0:
        raise RuntimeError(f"資料不含 {start}~{end} 區間")

    mkt = _window_metrics(ret_df.fillna(0.0).mean(axis=1).reindex(win))   # 等權大盤
    bench_ret = ev.load_benchmark_returns(benchmark).reindex(win).fillna(0.0)
    bench = _window_metrics(bench_ret) if bench_ret.abs().sum() > 0 else None

    default = (config.ROTATION_MOM_DAYS, config.ROTATION_TOP_K,
               config.ROTATION_REBAL_DAYS)
    combos = combos or [default]
    rows = []
    for (mom, k, reb) in combos:
        net, _ = rotation_returns(ret_df, close_df, mom, k, reb)
        rows.append((mom, k, reb, _window_metrics(net.reindex(win))))

    if verbose:
        print(f"=== 期間測試 {win[0].date()} ~ {win[-1].date()}（{len(win)} 交易日）===")
        print(f"等權大盤　　 總報酬 {mkt['total']*100:+6.1f}%　MDD {mkt['mdd']*100:6.1f}%"
              f"　Sharpe {mkt['sharpe']:+.2f}")
        if bench:
            print(f"006208　　　 總報酬 {bench['total']*100:+6.1f}%　MDD {bench['mdd']*100:6.1f}%"
                  f"　Sharpe {bench['sharpe']:+.2f}")
        for (mom, k, reb, m) in rows:
            tag = "（預設）" if (mom, k, reb) == default else ""
            print(f"輪動 m{mom} k{k} r{reb}{tag} 總報酬 {m['total']*100:+6.1f}%"
                  f"　MDD {m['mdd']*100:6.1f}%　Sharpe {m['sharpe']:+.2f}"
                  f"　vs大盤 {(m['total']-mkt['total'])*100:+.1f}pp")

    grid_summary = None
    if grid:
        g = []
        import itertools as _it
        for mom, k, reb in _it.product(config.ROT_ROBUST_MOM_GRID,
                                       config.ROT_ROBUST_K_GRID,
                                       config.ROT_ROBUST_REBAL_GRID):
            net, _ = rotation_returns(ret_df, close_df, mom, k, reb)
            m = _window_metrics(net.reindex(win))
            g.append({"mom": mom, "k": k, "rebal": reb, **m})
        gdf = pd.DataFrame(g)
        grid_summary = gdf
        if verbose:
            beat_ret = (gdf["total"] > mkt["total"]).mean()
            beat_mdd = (gdf["mdd"] > mkt["mdd"]).mean()      # mdd 負數,> 表回撤較淺
            print(f"\n[全網格 {len(gdf)} 組在此空頭期]")
            print(f"  總報酬贏過大盤比例：{beat_ret*100:.0f}%"
                  f"　回撤比大盤淺比例：{beat_mdd*100:.0f}%")
            print(f"  輪動總報酬 中位數 {gdf['total'].median()*100:+.1f}%"
                  f"（大盤 {mkt['total']*100:+.1f}%）"
                  f"　最佳 {gdf['total'].max()*100:+.1f}% / 最差 {gdf['total'].min()*100:+.1f}%")

    return {"window": (win[0], win[-1]), "market": mkt, "benchmark": bench,
            "rows": rows, "grid": grid_summary}


def compare_gates_period(symbols=None, start="2021-12-01", end="2022-10-31",
                         benchmark=None, full_check=True, verbose=True):
    """
    在指定空頭期比較四種版本:原始滿倉 / +絕對動能 / +regime / +兩者。
    對「預設參數」逐項列出,並對「全網格」算各版本『跌得比大盤少』的比例(穩健度)。
    regime 的 HMM 只用「視窗開始日之前」的資料配適 -> 對該視窗無洩漏。
    full_check=True 時另列各版本「全期(含多頭)」總報酬,確認閘門沒殺掉多頭 edge。
    """
    from core.regime import RegimeHMM
    symbols = symbols or config.WATCHLIST
    benchmark = benchmark or config.BENCHMARK_SYMBOL
    panel = build_feature_panel(symbols, with_cs=False)
    ret_df, close_df = _panel_to_frames(panel)
    idx = ret_df.index
    win = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(win) == 0:
        raise RuntimeError(f"資料不含 {start}~{end}")

    # regime:HMM 只用視窗開始前的大盤資料配適(無洩漏),causal P(bull)
    mkt_ret = ret_df.fillna(0.0).mean(axis=1)
    rg = RegimeHMM().fit(mkt_ret[mkt_ret.index < pd.Timestamp(start)])
    pbull = rg.predict_bull_proba(mkt_ret, causal=True)

    mkt = _window_metrics(mkt_ret.reindex(win))
    bench_ret = ev.load_benchmark_returns(benchmark).reindex(win).fillna(0.0)
    bench = _window_metrics(bench_ret) if bench_ret.abs().sum() > 0 else None

    variants = {
        "原始(滿倉)": dict(),
        "+絕對動能": dict(abs_mom=True, abs_thresh=0.0),
        "+regime": dict(regime_pbull=pbull),
        "+兩者": dict(abs_mom=True, abs_thresh=0.0, regime_pbull=pbull),
    }
    default = (config.ROTATION_MOM_DAYS, config.ROTATION_TOP_K,
               config.ROTATION_REBAL_DAYS)
    combos = list(__import__("itertools").product(
        config.ROT_ROBUST_MOM_GRID, config.ROT_ROBUST_K_GRID,
        config.ROT_ROBUST_REBAL_GRID))

    out = {}
    if verbose:
        print(f"=== 空頭期 {win[0].date()}~{win[-1].date()}（{len(win)} 日)===")
        print(f"基準　等權大盤 {mkt['total']*100:+.1f}%  MDD {mkt['mdd']*100:.1f}%"
              + (f"　|  006208 {bench['total']*100:+.1f}%  MDD {bench['mdd']*100:.1f}%"
                 if bench else ""))
        print(f"\n{'版本':<12}{'預設參數總報酬':>14}{'MDD':>9}{'Sharpe':>9}"
              f"{'  全網格跌得比大盤少':>0}")
    for name, kw in variants.items():
        net_d, _ = rotation_returns(ret_df, close_df, *default, **kw)
        md = _window_metrics(net_d.reindex(win))
        beat = np.mean([
            _window_metrics(rotation_returns(ret_df, close_df, m, k, r, **kw)[0]
                            .reindex(win))["total"] > mkt["total"]
            for (m, k, r) in combos])
        full = None
        if full_check:
            full = _window_metrics(net_d)        # 全期(含多頭)
        out[name] = {"default": md, "beat_grid": beat, "full": full}
        if verbose:
            print(f"{name:<12}{md['total']*100:>+13.1f}%{md['mdd']*100:>8.1f}%"
                  f"{md['sharpe']:>+9.2f}{beat*100:>14.0f}%")

    if verbose and full_check:
        print("\n[全期含多頭:預設參數總報酬 / Sharpe(確認閘門沒殺掉多頭 edge)]")
        for name in variants:
            f = out[name]["full"]
            print(f"  {name:<10} 總報酬 {f['total']*100:+8.1f}%　Sharpe {f['sharpe']:+.2f}"
                  f"　MDD {f['mdd']*100:.1f}%")
    return out


if __name__ == "__main__":
    run_rotation_robustness(verbose=True)
