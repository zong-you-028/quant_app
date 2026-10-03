# -*- coding: utf-8 -*-
"""正式輪動策略：nested WFA、擴大參數穩健性、閘門消融。

所有候選策略都使用前一日權重乘當日報酬並扣實際換手成本。Nested WFA 每折
只用 expanding IS 選參，緊接的 OOS 完全凍結；OOS 結果不參與該折選擇。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

import config
from core import evaluation as ev
from core.chip_data import fastsell_z_panel
from core.data_pipeline import ensure_data, load_ohlcv
from core.market_regime import sox_regime_series
from core.rotation import _load_panel, _select_picks


@dataclass(frozen=True)
class RotationParams:
    mom: int
    skip: int
    k: int
    rebal: int
    sox_ma: int


@dataclass(frozen=True)
class GateConfig:
    abs_gate: bool
    fastsell_gate: bool
    sox_gate: bool


GATE_GRID = tuple(GateConfig(*v) for v in itertools.product((False, True), repeat=3))


PRODUCTION = RotationParams(
    config.ROTATION_MOM_DAYS, config.ROTATION_SKIP_DAYS,
    config.ROTATION_TOP_K, config.ROTATION_REBAL_DAYS, config.ROTATION_SOX_MA,
)

DEFAULT_GRID = {
    "mom": (40, 60, 80),
    "skip": (0, 10),
    "k": (6, 8, 10),
    "rebal": (15, 20, 30),
    "sox_ma": (150, 200, 250),
}

# Nested 必須聯合搜尋閘門，為控制多重比較與運算量，只用包住正式值與前次
# expanding WFA 常勝區的聚焦網格；擴大 162 組仍由 robustness 單獨完整檢驗。
NESTED_GRID = {
    "mom": (60, 80), "skip": (0, 10), "k": (6, 8),
    "rebal": (20, 30), "sox_ma": (150, 200, 250),
}


def parameter_grid(grid=None):
    g = grid or DEFAULT_GRID
    values = itertools.product(g["mom"], g["skip"], g["k"], g["rebal"], g["sox_ma"])
    return [RotationParams(*v) for v in values if v[1] < v[0]]


class ProductionResearchEngine:
    """一次載入資料，快速重播正式策略的不同參數與閘門版本。"""

    def __init__(self, symbols=None):
        self.symbols = list(symbols or config.UNIVERSE)
        self._panels = {}
        # 先用正式設定取得共同日曆；其他動能組合再對齊它。
        ret, _, _, _ = _load_panel(self.symbols, PRODUCTION.mom,
                                    config.ROTATION_MIN_OBS, PRODUCTION.skip)
        self.index = ret.index
        self.fastsell = fastsell_z_panel(self.symbols, self.index)
        ensure_data(config.BENCHMARK_SYMBOL)
        self.benchmark = (load_ohlcv(config.BENCHMARK_SYMBOL)["close"].astype(float)
                          .sort_index().pct_change().reindex(self.index))
        self._sox = {}

    def panel(self, mom, skip):
        key = (mom, skip)
        if key not in self._panels:
            ret, rank_mom, gate_mom, _ = _load_panel(
                self.symbols, mom, config.ROTATION_MIN_OBS, skip)
            self._panels[key] = (ret.reindex(self.index),
                                 rank_mom.reindex(self.index),
                                 gate_mom.reindex(self.index))
        return self._panels[key]

    def sox(self, ma):
        if ma not in self._sox:
            self._sox[ma] = sox_regime_series(
                self.index, ma=ma, lag=config.ROTATION_SOX_LAG)
        return self._sox[ma]

    def returns(self, params: RotationParams, *, abs_gate=True,
                fastsell_gate=True, sox_gate=True, cost=None,
                execution_delay=0):
        ret, rank_mom, gate_mom = self.panel(params.mom, params.skip)
        weights = pd.DataFrame(0.0, index=self.index, columns=ret.columns)
        current = pd.Series(0.0, index=ret.columns)
        fs = self.fastsell if fastsell_gate else None
        for i, date in enumerate(self.index):
            if i % params.rebal == 0:
                fsrow = fs.loc[date] if fs is not None and date in fs.index else None
                selected = _select_picks(
                    rank_mom.loc[date], gate_mom.loc[date], params.k,
                    abs_gate, config.ROTATION_ABS_THRESH,
                    fastsell_row=fsrow, fastsell_z=config.ROTATION_FASTSELL_Z,
                )
                current = pd.Series(0.0, index=ret.columns)
                for symbol in selected:
                    current[symbol] = 1.0 / params.k
            weights.loc[date] = current
        execution_delay = int(execution_delay)
        if execution_delay < 0:
            raise ValueError("成交延遲不可小於 0")
        # delay=0 完全等同正式策略：決策權重自下一個交易日承擔報酬。
        executed = weights.shift(execution_delay).fillna(0.0)
        gross = (executed.shift(1).fillna(0.0) * ret.fillna(0.0)).sum(axis=1)
        turnover = executed.diff().abs().sum(axis=1)
        if len(turnover):
            turnover.iloc[0] = weights.iloc[0].abs().sum()
        net = gross - turnover * (config.COST_PER_TURNOVER if cost is None else cost)
        if sox_gate:
            regime = self.sox(params.sox_ma).shift(execution_delay).fillna(1.0)
            net = net * regime - regime.diff().abs().fillna(0.0) * (
                config.COST_PER_TURNOVER if cost is None else cost)
        return net.rename("strategy")


def _metrics(strategy, benchmark):
    report = ev.financial_report(strategy, benchmark)
    return {k: report[k] for k in (
        "strat_cagr", "bench_cagr", "strat_sharpe", "bench_sharpe",
        "strat_mdd", "bench_mdd", "alpha", "information_ratio")}


def _folds(index, train_days=756, test_days=252):
    idx = pd.DatetimeIndex(index)
    out, start = [], int(train_days)
    while start < len(idx):
        stop = min(start + int(test_days), len(idx))
        out.append((idx[:start], idx[start:stop]))
        start = stop
    return out


def run_nested_wfa(engine=None, grid=None, train_days=756, test_days=252,
                   select_gates=True, verbose=True):
    """每折以 expanding IS Sharpe 選參，再串接完全凍結的下一段 OOS。"""
    engine = engine or ProductionResearchEngine()
    params = parameter_grid(NESTED_GRID if grid is None else grid)
    gates = GATE_GRID if select_gates else (GateConfig(True, True, True),)
    candidates = list(itertools.product(params, gates))
    cache = {(p, g): engine.returns(
        p, abs_gate=g.abs_gate, fastsell_gate=g.fastsell_gate, sox_gate=g.sox_gate)
             for p, g in candidates}
    rows, oos_parts = [], []
    common = engine.index.intersection(engine.benchmark.dropna().index)
    for number, (train_idx, test_idx) in enumerate(
            _folds(common, train_days, test_days), start=1):
        scored = [(ev.annualized_sharpe(cache[c].reindex(train_idx)), c)
                  for c in candidates]
        is_sharpe, (selected, selected_gates) = max(scored, key=lambda item: item[0])
        test = cache[(selected, selected_gates)].reindex(test_idx)
        bench = engine.benchmark.reindex(test_idx)
        met = _metrics(test, bench)
        rows.append({"fold": number, "train_end": train_idx[-1],
                     "test_start": test_idx[0], "test_end": test_idx[-1],
                     "is_sharpe": is_sharpe, "params": selected,
                     "gates": selected_gates, **met})
        oos_parts.append(test)
    table = pd.DataFrame(rows)
    oos = pd.concat(oos_parts).sort_index()
    summary = _metrics(oos, engine.benchmark.reindex(oos.index))
    if verbose:
        print("=== Nested Walk-Forward（每折只用 IS 選參）===")
        for row in rows:
            p = row["params"]
            g = row["gates"]
            print(f"Fold {row['fold']:02d} OOS {row['test_start'].date()}~{row['test_end'].date()} "
                  f"選 m{p.mom}/s{p.skip}/k{p.k}/r{p.rebal}/sox{p.sox_ma} | "
                  f"gate A{int(g.abs_gate)}F{int(g.fastsell_gate)}S{int(g.sox_gate)} | "
                  f"IS Shp {row['is_sharpe']:+.2f} -> OOS Shp {row['strat_sharpe']:+.2f}, "
                  f"alpha {row['alpha']*100:+.1f}pp")
        print(f"合併 OOS: CAGR {summary['strat_cagr']*100:+.1f}% vs "
              f"006208 {summary['bench_cagr']*100:+.1f}% | Sharpe "
              f"{summary['strat_sharpe']:.2f} vs {summary['bench_sharpe']:.2f} | "
              f"MDD {summary['strat_mdd']*100:.1f}% vs {summary['bench_mdd']*100:.1f}%")
    return {"folds": table, "oos": oos, "summary": summary, "engine": engine}


def run_execution_stress(nested, costs=(0.003, 0.005, 0.008, 0.010),
                         delays=(0, 1, 2), verbose=True):
    """凍結 baseline nested 每折所選規則，只改成本與額外成交延遲。"""
    engine = nested["engine"]
    rows = []
    for cost, delay in itertools.product(costs, delays):
        parts = []
        for _, fold in nested["folds"].iterrows():
            p, g = fold["params"], fold["gates"]
            ret = engine.returns(
                p, abs_gate=g.abs_gate, fastsell_gate=g.fastsell_gate,
                sox_gate=g.sox_gate, cost=cost, execution_delay=delay)
            parts.append(ret.loc[fold["test_start"]:fold["test_end"]])
        oos = pd.concat(parts).sort_index()
        met = _metrics(oos, engine.benchmark.reindex(oos.index))
        rows.append({"cost": cost, "delay": delay, **met})
    table = pd.DataFrame(rows)
    if verbose:
        print("\n=== Nested OOS 成本／成交延遲壓力測試 ===")
        for _, r in table.iterrows():
            print(f"cost {r.cost*100:.1f}% delay +{int(r.delay)}d | "
                  f"CAGR {r.strat_cagr*100:+6.1f}% Sharpe {r.strat_sharpe:.2f} "
                  f"MDD {r.strat_mdd*100:6.1f}% alpha {r.alpha*100:+6.1f}pp")
    return table


def run_parameter_robustness(engine=None, grid=None, is_fraction=0.6, verbose=True):
    """擴大正式網格，報告整個鄰域而非只報最佳參數。"""
    engine = engine or ProductionResearchEngine()
    idx = engine.index.intersection(engine.benchmark.dropna().index)
    split = int(len(idx) * is_fraction)
    is_idx, oos_idx = idx[:split], idx[split:]
    rows = []
    for p in parameter_grid(grid):
        ret = engine.returns(p)
        rows.append({"params": p, "mom": p.mom, "skip": p.skip, "k": p.k,
                     "rebal": p.rebal, "sox_ma": p.sox_ma,
                     "is_sharpe": ev.annualized_sharpe(ret.reindex(is_idx)),
                     "oos_sharpe": ev.annualized_sharpe(ret.reindex(oos_idx)),
                     "oos_cagr": ev.annualized_return(ret.reindex(oos_idx)),
                     "oos_alpha": ev.active_return(ret.reindex(oos_idx),
                                                   engine.benchmark.reindex(oos_idx))})
    table = pd.DataFrame(rows)
    if verbose:
        prod = table[(table.mom == PRODUCTION.mom) & (table.skip == PRODUCTION.skip) &
                     (table.k == PRODUCTION.k) & (table.rebal == PRODUCTION.rebal) &
                     (table.sox_ma == PRODUCTION.sox_ma)]
        print("\n=== 正式參數鄰域穩健性 ===")
        print(f"組合 {len(table)} | OOS Sharpe 中位數 {table.oos_sharpe.median():.2f} | "
              f"OOS Sharpe>0 {(table.oos_sharpe > 0).mean()*100:.0f}% | "
              f"OOS alpha>0 {(table.oos_alpha > 0).mean()*100:.0f}%")
        if len(prod):
            r = prod.iloc[0]
            print(f"正式參數: IS Sharpe {r.is_sharpe:.2f} -> OOS Sharpe {r.oos_sharpe:.2f}, "
                  f"OOS CAGR {r.oos_cagr*100:+.1f}%, alpha {r.oos_alpha*100:+.1f}pp")
    return table


def run_gate_ablation(engine=None, split_fraction=0.6, verbose=True):
    """固定正式參數，逐項移除與加入閘門，僅比較後 40% OOS。"""
    engine = engine or ProductionResearchEngine()
    idx = engine.index.intersection(engine.benchmark.dropna().index)
    oos_idx = idx[int(len(idx) * split_fraction):]
    variants = {
        "base": (False, False, False),
        "+abs": (True, False, False),
        "+abs+fastsell": (True, True, False),
        "+abs+sox": (True, False, True),
        "full": (True, True, True),
        "full-no-abs": (False, True, True),
        "full-no-fastsell": (True, False, True),
        "full-no-sox": (True, True, False),
    }
    rows = []
    for name, gates in variants.items():
        ret = engine.returns(PRODUCTION, abs_gate=gates[0],
                             fastsell_gate=gates[1], sox_gate=gates[2]).reindex(oos_idx)
        rows.append({"variant": name, **_metrics(ret, engine.benchmark.reindex(oos_idx))})
    table = pd.DataFrame(rows)
    if verbose:
        print("\n=== 閘門消融（固定正式參數，後 40% OOS）===")
        for _, r in table.iterrows():
            print(f"{r['variant']:<18} CAGR {r.strat_cagr*100:+6.1f}% "
                  f"Sharpe {r.strat_sharpe:+.2f} MDD {r.strat_mdd*100:6.1f}% "
                  f"alpha {r.alpha*100:+6.1f}pp")
    return table


def run_all(verbose=True):
    engine = ProductionResearchEngine()
    nested = run_nested_wfa(engine, verbose=verbose)
    robustness = run_parameter_robustness(engine, verbose=verbose)
    ablation = run_gate_ablation(engine, verbose=verbose)
    stress = run_execution_stress(nested, verbose=verbose)
    return {"nested": nested, "robustness": robustness,
            "ablation": ablation, "stress": stress}


if __name__ == "__main__":
    run_all()
