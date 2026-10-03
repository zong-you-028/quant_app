# 固定50檔與可取得收盤價普通股的估算市值150檔：歷史比較

**目前選池之後的歷史診斷含選股及存活者偏誤，不是新樣本外，也不證明最佳化成功。**

模型保持低換手多視窗、8個等權名額、前16名續留、20交易日換股；未變更app設定或既有紙上驗證manifest。

名單市值日期：2026-10-02；資料上限：2026-10-02。
兩池使用同一006208日曆、費半来源、外資急賣閘門、成本與成交延遲，全部情境原樣保留。

排名範圍：same-day priced TWSE/TPEx classified ordinary shares。這是依普通股收盤與股數來源估算的候選池，不稱官方完整全市場市值榜。 名單仍暫定；7檔普通股缺同日有效價格被明列排除，交易狀態／完整涵蓋尚未確認。
## 資料覆蓋

| 股票池 | 宣告檔數 | 有行情 | 有籌碼 | 最新排名資格 | 完整來源 |
|---|---:|---:|---:|---:|---|
| fixed_50 | 50 | 50 | 50 | 48 | True |
| market_cap_150 | 150 | 150 | 150 | 142 | True |

短上市歷史保留於宣告池，依252筆連續行情與動能錨點暖機，不補假價；coverage明示短歷史與實際可排名檔數。完全缺行情／外資或来源過期時拒絕比較，不縮池冒稱完整150檔。

新增行情採『日期不早於官方current-market ISIN上市／櫃日期』的保守下限，排除興櫃及其他更早資料。
本次修剪20檔的早期價格；逐檔日期及移除筆數保存於results.json的acquisition_provenance。
此規則也可能移除轉板前合法上市櫃行情，不能稱為完整歷史PIT資格；原50檔公開seed保持原價格。

共同區段：2018-09-18～2026-10-02，1955筆。
沿用app歷史暖機與持倉，截取同期成本後報酬，不在評估起點重設持倉。

## 固定成本與成交壓力

| 成本／lag | 50 CAGR | 150 CAGR | 50 Sharpe | 150 Sharpe | 50回撤 | 150回撤 | 50年換手 | 150年換手 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3%／t+2 | 30.42% | 60.03% | 1.28 | 1.60 | -29.24% | -36.92% | 8.96 | 11.70 |
| 0.3%／t+3 | 28.82% | 58.38% | 1.23 | 1.58 | -27.51% | -37.26% | 8.98 | 11.72 |
| 0.3%／t+4 | 24.12% | 47.13% | 1.06 | 1.34 | -28.11% | -38.26% | 8.99 | 11.71 |
| 0.5%／t+2 | 28.10% | 56.33% | 1.20 | 1.53 | -32.18% | -37.06% | 8.96 | 11.69 |
| 0.5%／t+3 | 26.52% | 54.70% | 1.15 | 1.50 | -30.53% | -37.40% | 8.98 | 11.71 |
| 0.5%／t+4 | 21.90% | 43.72% | 0.98 | 1.26 | -30.42% | -39.00% | 8.99 | 11.70 |
| 0.8%／t+2 | 24.70% | 50.93% | 1.08 | 1.42 | -36.37% | -39.52% | 8.95 | 11.68 |
| 0.8%／t+3 | 23.15% | 49.35% | 1.03 | 1.39 | -34.81% | -37.59% | 8.97 | 11.70 |
| 0.8%／t+4 | 18.65% | 38.75% | 0.86 | 1.16 | -34.72% | -42.72% | 8.98 | 11.69 |

## 逐年（成本0.3%、t+2）

首末年可能不足一年；年化平均日差不是CAGR差。

| 年／日數 | 50 CAGR／Sharpe | 150 CAGR／Sharpe | 006208 CAGR／Sharpe | 年化平均日差150−50 |
|---|---|---|---|---:|
| 2018／73 | -22.46%／-2.80 | -21.34%／-1.84 | -47.02%／-3.13 | 1.82% |
| 2019／242 | 16.31%／1.12 | 37.75%／1.58 | 30.27%／2.24 | 18.27% |
| 2020／245 | 32.92%／1.49 | 62.94%／1.63 | 29.18%／1.35 | 23.89% |
| 2021／244 | 43.48%／1.48 | 192.92%／2.88 | 19.74%／1.12 | 75.95% |
| 2022／246 | -8.32%／-1.57 | -14.80%／-1.63 | -24.64%／-1.21 | -7.01% |
| 2023／239 | 35.62%／1.61 | 109.55%／2.80 | 25.71%／1.76 | 45.43% |
| 2024／242 | 3.34%／0.26 | 60.16%／1.55 | 50.30%／1.85 | 47.76% |
| 2025／243 | 42.37%／1.97 | 56.40%／1.78 | 32.89%／1.24 | 11.35% |
| 2026／181 | 198.87%／2.61 | 91.25%／1.40 | 112.92%／2.59 | -38.06% |

## 配對不確定性

20／40／60日循環區塊，3000次、seed314159；單項回溯區間未校正選池／選型，不取代先前11候選PBO。

| 區塊日數 | 年化平均日差150−50 | 單項95%區間 | 單候選單尾p |
|---|---:|---|---:|
| 20 | 23.27% | 6.67%～40.36% | 0.004 |
| 40 | 23.27% | 5.53%～40.43% | 0.005 |
| 60 | 23.27% | 5.66%～41.77% | 0.008 |

## 限制

- Current market-cap membership was selected after observing historical markets; it retains selection/survivorship bias and is not historical PIT membership.
- All comparison dates are retrospective; neither universe nor favorable stress setting is automatically deployed or selected.
- Ranking/market-cap methodology is supplied in source JSON, not independently certified historical membership.
- The supplied candidate ranking estimates priced TWSE/TPEx ordinary-share capitalization; issued-common and listed-share capitalizations can differ, and unpriced official-classified stocks are explicitly excluded.
- Prices retain the same unverified +/-11% corporate-action heuristic; neither dataset is certified total-return data.
- Added-stock prices use a conservative current-market ISIN listing-date floor that can exclude legitimate older OTC transfers as well as emerging history; this is not complete PIT eligibility.
- Historical chip dates are not publication timestamps; absent rolling Z observations retain existing gate warmup behavior.
- The benchmark calendar is the existing app calendar, not an independently verified exchange-session calendar.
- Flat costs omit spreads, capacity, locked limits, asymmetric taxes and corporate-action cash/quantity accounting.
- 006208 is a close-to-close reference without brokerage entry/exit costs, not a fill-verified paper account.
- The frozen forward paper model and original 50-stock manifest are unchanged by this research.

重播：`python research/universe_150_2026_10_03/compare_universes.py --db <獨立DB> --sox <SOX.csv> --universe <排名JSON>`。
