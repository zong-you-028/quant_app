# 固定候選策略與模型研究

行情快照：2015-01-05～2026-09-01，50 檔。
開發區段：2018-09-18～2024-08-05。
本次選型排除區段：2024-08-06～2026-09-01。
只依開發區段選出的候選：`trend_quality`；正式策略未變更。

下表為扣成本、延遲開盤成交的末段結果；所有候選使用相同交易日與股票池。

| 候選 | 開發 Sharpe | 末段 CAGR | 末段 Sharpe | 末段最大回撤 | 年換手 |
|---|---:|---:|---:|---:|---:|
| production | 0.77 | 53.84% | 1.61 | -30.65% | 19.1 |
| multi_momentum | 1.13 | 62.01% | 1.74 | -26.30% | 16.5 |
| trend_quality | 1.13 | 67.77% | 1.95 | -17.63% | 19.3 |
| buffered_momentum | 1.00 | 70.21% | 1.88 | -25.58% | 13.0 |
| vol_control | 1.05 | 54.54% | 1.85 | -17.58% | 14.3 |
| lgb_rank | 0.46 | 43.27% | 1.57 | -20.96% | 16.2 |
| lambda_rank | 0.46 | 33.36% | 1.31 | -21.01% | 15.6 |
| xgb_rank | 0.47 | 39.71% | 1.51 | -20.41% | 15.5 |
| ridge_rank | 0.76 | 59.50% | 1.82 | -18.06% | 16.8 |
| rank_ensemble | 0.43 | 51.21% | 1.77 | -21.67% | 15.8 |
| ai_momentum_blend | 0.88 | 55.20% | 1.70 | -24.53% | 17.5 |

同區段 006208 買進持有 CAGR：62.80%。

選定候選相對正式策略的年化平均日超額報酬，20 日區塊 bootstrap 95% 區間：
-9.31%～27.79%。這不是 CAGR 差異的信賴區間，也未做多重比較校正。

固定候選在完整共同區段的比較：

| 策略 | CAGR | Sharpe | 最大回撤 |
|---|---:|---:|---:|
| production | 23.72% | 1.04 | -33.63% |
| trend_quality | 32.27% | 1.39 | -26.65% |

壓力測試（仍只比較原先選定候選，不依結果換策略）：

| 成本／成交延遲 | 候選 CAGR | 正式策略 CAGR | 候選最大回撤 |
|---|---:|---:|---:|
| 0.5% / t+2 | 61.42% | 48.08% | -21.56% |
| 0.8% / t+2 | 52.34% | 39.83% | -27.10% |
| 0.3% / t+3 | 63.31% | 50.14% | -18.96% |

換股排程位移敏感度（0、5、10、15 日；不選最佳排程）：

| 位移 | 候選 CAGR | 正式策略 CAGR | 候選 Sharpe |
|---|---:|---:|---:|
| 0 | 67.77% | 53.84% | 1.95 |
| 5 | 34.73% | 54.91% | 1.17 |
| 10 | 51.19% | 74.16% | 1.57 |
| 15 | 62.26% | 58.46% | 1.96 |

候選只在 2/4 種排程的 CAGR 勝過正式策略；換股日期敏感，尚未具備穩健替換證據。

後續探索：四批分散換股，降低單一排程依賴。這是在看過末段後新增的研究，
未納入原先選型；下列數字屬探索結果，不是新的未見資料驗證。

| 分批策略 | 末段 CAGR | 末段 Sharpe | 末段最大回撤 | 年換手 |
|---|---:|---:|---:|---:|
| trend_quality | 53.46% | 1.72 | -19.12% | 19.3 |
| production | 60.28% | 1.74 | -29.59% | 20.3 |

限制：

- Historical snapshot ends at price_asof; these are not current trading signals.
- Final split is retrospective; earlier research has already seen this snapshot.
- Current 50-stock universe retains selection/survivorship bias; no delisted execution model.
- Cached adjusted prices and chip dates are used as provided; release times not verified.
- No limit-lock, volume/capacity, slippage beyond flat costs, or cash interest modeling.
- Bootstrap interval is exploratory and not corrected for multiple strategy comparisons.

固定模型：LightGBM 排名迴歸、LambdaRank、XGBoost、Ridge；集成採等權百分位排名。
所有模型僅使用標籤結束日早於當折預測日的樣本，固定每 126 日重訓；不依末段選模型或調參。
