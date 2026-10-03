# 150 檔整合唯讀審查（2026-10-04）

審查對象為本輪 working tree 的 `core/tpex_prices.py`、`core/data_pipeline.py`、`config.py`、`main.py`、`research/universe_150_2026_10_03/fetch_history.py`，以及 old50 compatibility command。未連網，未開啟正式 `data/market.db`，未修改任何模型或 pinned source。僅用 TPEx 純解析函式與一列合成資料確認邊界。

## 需先補的設定與前瞻防線

1. **150 metadata 必須完整且與 symbols 一致。** `config.py` 目前驗證 symbols 的 150／唯一數、scope 與 completeness flag，卻未驗證 `top150` 的 symbol 集合、重複、合法 market、有效上市日期與 `listed_date <= asof`。若 symbols 完整而某列 metadata 缺失，`UNIVERSE_LISTING_DATES.get(symbol)` 將省略掛牌日限制，`_is_tpex` 將退成 TWSE。建議啟動時檢查兩份名單完全同集合且各有 150 唯一列；每列 market 只能 twse/tpex；日期必須 ISO 日、不得晚於名單日。此屬設定 fail-closed，不能用缺失值猜市場。

2. **新的 150 前瞻實驗要凍結外部 pool metadata。** 現有 `frozen_identity` 會凍結 UNIVERSE symbols 和 config.py 原始碼，卻未凍結新載入的 `data_seed/universe_150.json` 檔案、UNIVERSE_MARKETS、UNIVERSE_LISTING_DATES 與 UNIVERSE_ASOF。symbols 不變而 metadata 改掛牌日期／市場會改資格與載入結果，identity 仍可能相同。新增 150 實驗前，將該固定 JSON 的 SHA 及實際使用的上述欄位納入身分；舊50用 inline config 的 pinned runtime 保持原樣，不能重寫舊 protocol。

## 行情與恢復下載

3. **TPEx daily 最新分支缺資料上限，數字解析接受 infinity。** `parse_daily`／`parse_month` 只 dropna、檢查大於零，`float('inf')` 會通過。`fetch_tpex_latest_data` 的 latest 分支也未在 merge 前限制到 expected 日期。純合成重現：`Date=1151005`、OHLCV 全為 `inf` 的一列被 parser 接受（本輪資料目標為 2026-10-02）。建議拒絕非有限 OHLCV、確認 high/low 合理，最新列不得晚於本次資料上限；超出日期應拒絕而非推進 MAX(date)。monthly 最近區段已有 end filter，應維持。

4. **fetch_history resume 的 base seed provenance 會漂移。** `manifest.update(base_seed_sha256=digest(ROOT/data_seed/market.db.gz))` 每次執行都覆寫，但已有研究 DB 並未重新從現在 seed 初始化。公開 seed 擴大到150後，舊研究 DB 會被誤標為新 seed 的後裔。只在首次建 DB 時記錄 base seed SHA；其後保留原值，缺少舊證據則明示 unknown，不推定現在 seed。

5. **HTTP 拒絕狀態應先於 JSON 解析。** `fetch` 先 `response.json()` 才判 401／402／403／429。若拒絕頁是 HTML，JSONDecodeError 會走一般逐檔錯誤，批次繼續請求，而非立即停止。先檢查 HTTP denial，再解析 JSON 及 API-body denial；這不改模型，只避免無效長時間重試。

6. **來源範圍應用實際措辭。** 名單 JSON 已明示 `scope=priced_ordinary`、`provisional=true`、缺價格股票的交易狀態未确认；UI 的「可交易普通股」比「同日有有效收盤報價普通股」更強。建議顯示估算／暫定／固定名單日，避免把有效價格當成已驗證可成交或完整全市場排序。零成交／缺可成交 open 與真實容量仍須成交驗證，不能由此名單推導。

## 已確認較早的主問題已處理

- 最新 `load_ohlcv` 已把舊 cache 行情讀取限制在官方 current-market 掛牌日以後；最新 `fetch_real_data` 也在公司行動清理前篩掛牌日並拒絕空結果。因此較早「重新完整下載帶回興櫃並提早滿252筆」主路徑已有讀端＋完整下載端防線。只要上一項 metadata 完整性成立，這條路徑不再直接餵入模型。若還要 cache 全表語義也完全一致，可於 merge 的同一交易內刪除低於 floor 的價格；讀端篩選仍須保留。
- TPEx routine routing 已避免送到 TWSE；缺完整 OTC 歷史時明確失敗，未將幾天近期價格冒稱252日歷史。長缺口讀取所有月份，某月失敗時尚未 merge，因此不先推進更新 cursor。
- `compare_universes.py` 凍結原50符號、共同006208報價日曆、固定8／20／lag／cost，同時把 current membership／存活者偏誤、未驗證公司行動、回溯比較明列限制；不能把此結果稱新增樣本外。
- 根 compatibility `capture_latest.run(offline=False)` 現已轉為 `pinned_run(offline=offline, refresh=not offline)`，與實際 helper API 相符。根檔不 import 現版 config／dp；helper 另起清除 PYTHONPATH、指定独立 APP_DATA_DIR／空 journal URL 的 subprocess，實際載入 f49b4ca 的50檔 source。因此同一 root Python process 曾 import 新150 config 不會污染 child 的舊50身分。
- 舊50實際離線接續已測 `already_sealed`、events1→1、原 protocol/event/snapshot hash 不變、無 formal DB／外部網路／journal SQL；公開舊 seed 與封存 snapshot 精確一致。這不代表新150已完成前瞻或已產生 verified fills。
