# 固定模型：歷史因果性與壓力驗證

**全部結果是已看過歷史的回溯診斷，不能排除過擬合，沒有依結果調參。**

公開行情快照截至 2026-10-02；實際使用 `core.rotation.run_rotation` 的排名、保留名單、風險閘門與開盤成交引擎。
資料來自 public seed 的暫存副本；不下載新資料、不存取正式 DB、不寫帳本。

## 因果性

30 項前綴／未來污染檢查通過：過去目標、換股排程和成交狀態一致；成本後報酬、權益與換手採1e-12浮點容差，逐項差值保存。
早期前綴未滿80筆的3711被app略過，全資料保留其當時全零目標欄；補零後目標逐筆一致，但加總維度造成約機器精度的數值差。未因此修改production。
這證明所檢查輸入下的程式因果性，不代表消除資料發布時點、選型或存活者偏誤。

## 固定壓力情境

比較區段 2018-09-18～2026-10-02；沿用 app 全歷史暖機及起始持倉，截取同期报酬，不重設歷史持倉。
原策略基準保留60日／跳過10日公式，重新在006208共同日曆計算；稱 `production_common_calendar`，不宣稱與舊 app 的股票日期聯集流程逐筆一致。
成本按每單位買賣總換手計；外部來源延遲以額外台股交易日的已計算閘門可得性模擬，仍另加成交 lag。

| 情境 | 模型 CAGR | 原策略 CAGR | 模型 Sharpe | 模型回撤 | 模型年換手 |
|---|---:|---:|---:|---:|---:|
| cost_0.003_lag_2 | 30.42% | 24.31% | 1.28 | -29.24% | 8.96 |
| cost_0.003_lag_3 | 28.82% | 25.46% | 1.23 | -27.51% | 8.98 |
| cost_0.003_lag_4 | 24.12% | 19.31% | 1.06 | -28.11% | 8.99 |
| cost_0.005_lag_2 | 28.10% | 20.86% | 1.20 | -32.18% | 8.96 |
| cost_0.005_lag_3 | 26.52% | 21.98% | 1.15 | -30.53% | 8.98 |
| cost_0.005_lag_4 | 21.90% | 15.99% | 0.98 | -30.42% | 8.99 |
| cost_0.008_lag_2 | 24.70% | 15.87% | 1.08 | -36.37% | 8.95 |
| cost_0.008_lag_3 | 23.15% | 16.93% | 1.03 | -34.81% | 8.97 |
| cost_0.008_lag_4 | 18.65% | 11.19% | 0.86 | -34.72% | 8.98 |
| chip_delay_1 | 31.00% | 25.89% | 1.29 | -28.32% | 9.09 |
| chip_delay_2 | 30.39% | 25.66% | 1.29 | -28.95% | 9.02 |
| sox_delay_1 | 30.94% | 26.98% | 1.30 | -28.48% | 8.96 |
| sox_delay_2 | 28.49% | 21.48% | 1.21 | -26.51% | 8.93 |
| both_sources_delay_2 | 27.95% | 22.46% | 1.19 | -31.37% | 9.02 |

所有情境固定列出；不將最佳成本、延遲或報酬當作策略新設定。

## 單一年度集中度

固定採成本0.3%、lag2。相對均值是日報酬差算術平均乘252，並非CAGR差；首末年可能不足完整一年。

| 年 | 日數 | 模型 CAGR／Sharpe | 共日日曆原策略 CAGR／Sharpe | 006208 CAGR／Sharpe | 相對原策略年化平均日差 |
|---|---:|---|---|---|---:|
| 2018 | 73 | -22.46%／-2.80 | -36.65%／-3.57 | -47.02%／-3.13 | 19.80% |
| 2019 | 242 | 16.31%／1.12 | 23.00%／1.29 | 30.27%／2.24 | -6.05% |
| 2020 | 245 | 32.92%／1.49 | 24.03%／1.02 | 29.18%／1.35 | 6.09% |
| 2021 | 244 | 43.48%／1.48 | 34.38%／1.29 | 19.74%／1.12 | 6.97% |
| 2022 | 246 | -8.32%／-1.57 | -13.96%／-2.29 | -24.64%／-1.21 | 6.28% |
| 2023 | 239 | 35.62%／1.61 | 44.08%／1.76 | 25.71%／1.76 | -6.47% |
| 2024 | 242 | 3.34%／0.26 | -7.85%／-0.29 | 50.30%／1.85 | 11.31% |
| 2025 | 243 | 42.37%／1.97 | 45.86%／1.81 | 32.89%／1.24 | -3.12% |
| 2026 | 181 | 198.87%／2.61 | 144.16%／2.39 | 112.92%／2.59 | 22.44% |

全期相對原策略年化平均日差 4.71%；
排除年度差最高的 2026 年後為 2.90%。
這是事後集中度診斷，不能稱為該年度未見樣本外測試；全部逐年排除結果都保存，不以排除結果挑策略。

| 排除年 | 剩餘日數 | 相對原策略年化平均日差 | 相對006208年化平均日差 |
|---|---:|---:|---:|
| 2018 | 1882 | 4.13% | 4.56% |
| 2019 | 1713 | 6.23% | 8.12% |
| 2020 | 1710 | 4.52% | 6.16% |
| 2021 | 1711 | 4.39% | 3.69% |
| 2022 | 1709 | 4.49% | 4.07% |
| 2023 | 1716 | 6.27% | 5.34% |
| 2024 | 1713 | 3.78% | 11.98% |
| 2025 | 1712 | 5.83% | 5.80% |
| 2026 | 1774 | 2.90% | 2.27% |

## 同日期配對不確定性

固定模型成本0.3%、lag2的配對循環區塊bootstrap：20／40／60日，各3000次，seed314159；主比較為共日日曆原策略，006208為次要參考。
以下是已看過歷史的單項95%區間，未校正先前選型；不能與舊報告+5.16%的點估計及11候選校正p=0.389混用，也不取代該診斷。

| 比較基準 | 區塊日數 | 年化平均日差 | 單項95%區間 | 單候選單尾p |
|---|---:|---:|---|---:|
| production_common_calendar | 20 | 4.71% | -2.10%～11.88% | 0.100 |
| production_common_calendar | 40 | 4.71% | -1.55%～11.69% | 0.080 |
| production_common_calendar | 60 | 4.71% | -1.58%～11.67% | 0.090 |
| 006208 | 20 | 5.75% | -7.82%～20.08% | 0.216 |
| 006208 | 40 | 5.75% | -8.57%～19.83% | 0.208 |
| 006208 | 60 | 5.75% | -8.01%～20.40% | 0.201 |

## 仍未解決

- All dates were already available before this audit; no new chronological out-of-sample evidence.
- Price perturbations start after pipeline adjustment; historical raw-price revisions/corporate-action reconstruction are not validated.
- The existing +/-11% heuristic is not official total-return or corporate-action data; dividends, splits and missing-bar jumps remain unresolved.
- The current 50-stock universe is not historical point-in-time membership and omits delisted holdings.
- Date-only SOX/chip data do not certify publication timestamps; gate delays are stress assumptions, not verified release times.
- The app chip reader reindexes before forward-filling, whereas older research unions source dates first; this audit tests actual app behavior and does not reconcile non-calendar releases.
- The benchmark calendar omits dates present in the stock-source union; it is not independently certified as the exchange trading calendar.
- Flat costs do not model asymmetric taxes, spread, liquidity capacity, locked limits, cash interest or brokerage fills.
- The 006208 reference uses cached close-to-close returns without brokerage entry/exit costs, not a fill-verified ETF paper account.
- Shorter/deferred missing-open cases are covered by synthetic execution tests, not guaranteed by complete historical bars.
- Actual app initialization/calendar differs from the original research cash start; these metrics do not reproduce prior report values.
- Passing causal checks cannot remove selection bias, earlier undisclosed experiments or statistical overfitting.

重播：`python -m core.robustness_audit`；測試：`python -m pytest -q test_robustness_audit.py`。

來源檔案 SHA-256、逐項檢查與所有情境逐日報酬見同目錄 `results.json`、`causal_checks.csv`、`stress_daily_returns.csv`。
