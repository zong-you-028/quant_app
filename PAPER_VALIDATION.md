# 前瞻紙上驗證

本次先建立**固定規則的訊號封存**，不是已完成的券商模擬成交盤。歷史回測已被研究過；從封存之後累積的紀錄，才可能提供新的前瞻證據。現在沒有新的前瞻報酬，也不能宣布排除過擬合。

## 固定比較與檢查時點

- App 的低換手多視窗模型，與「原 60 日規則、共同日曆重播」及 006208 同期比較。基準標識為 `production_common_calendar`，只作研究用途，不恢復 app 多模式。
- 固定目前程式、參數、股票池、排程與閘門；8 個名額、20 交易日換股、t+2 開盤時序、研究換手成本 0.3%。保存版本與資料雜湊。
- 三個紙上組合都從封存後的現金起點開始，不匯入模型過往持倉，也不匯入個人帳本。初始目標另記為紙上盤啟動配置；後續遵循凍結的策略排程。
- 126 個實際交易日、至少 6 次排定換股後檢查資料與執行流程；這是營運檢查，不是統計認證。
- 預先固定 252 個有效前瞻交易日再作一次主要比較：低換手相對原模型的成本後每日報酬差，20 日配對區塊 bootstrap 95% 區間；006208 為另列的市場比較。若屆時資料品質或成交核對未完成，不宣稱通過。區間包含零則仍缺乏明確正向證據；即使下界為正也不代表保證未來獲利或完全排除過擬合。
- 不依途中輸贏更換模型、股票池或檢查終點；需要修改時建立新實驗，保留舊紀錄，不能延續同一驗證標籤。

## 封存必須保留的資訊

每筆保存真實 UTC 紀錄時間、行情日期、公開行情／籌碼／SOX 快照、目標權重、資料品質、程式與參數指紋。原始封存只追加；同日重跑不覆蓋。缺漏日、歷史資料修訂、程式或參數變更須明確標示，不能補算成當時已記錄的訊號。

也固定 Python／NumPy／pandas／SQLite 版本。Hash chain 可偵測檔案意外改寫；它不是外部公證，若操作者同時重算全部雜湊，不能提供不可否認的證明。

成交只可能發生在封存後，且必須有訊號日之後第 2 個**實際**交易日的可用開盤資料。未來交易日尚未收錄時保持 pending，不用一般工作日猜成交日期；缺必要開盤則整組延後。封存時已過去的開盤不得倒填成前瞻成交。

目前工具只保存目標與待執行狀態，沒有送單、虛構成交或計算已驗證 NAV。後續取得原始行情與可核對的公司行動／成交紀錄，再新增獨立的成交核對；不得改寫既有訊號。

## 已建立的實驗與使用方法

初始紀錄已在台北 **2026-10-03 20:58:42** 封存，來源截至 2026-10-02。三臂同一日曆、同日重跑未覆蓋、快照重播一致；目前待未來開盤核對，沒有成交或績效。[初始化驗證](research/paper_forward_2026_10_03/initial_verification.json)

本機原始封存位置為 `D:\quant_app\data\paper_validation\forward_2026_10_03_v1`，不在正式 `market.db`，也不推送個人資料。公開驗證報告另外保存在 research 目錄。

```powershell
cd D:\quant_app
# 驗證原始封存雜湊；不下載、不追加
python -m core.paper_validation --output data/paper_validation/forward_2026_10_03_v1 --verify
# 由封存快照重算目標；不認定成交或績效
python -m core.paper_validation --output data/paper_validation/forward_2026_10_03_v1 --replay
# 使用獨立公開行情快取，補最新資料後封存；不碰正式帳本
python research/paper_forward_2026_10_03/capture_latest.py
```

`capture_latest.py` 的快取為 `data/paper_validation/market_cache`，更新只處理公開股票／006208／籌碼／SOX。`--offline` 可檢查既有快取而不下載。請在資料發布後使用；更新失敗、來源過期或歷史修訂時不產生可執行的新訊號。既有紀錄的 `already_sealed` 只表示沒有覆蓋；仍須查看 `current_block_reasons`，不能解讀為今天資料已最新。工具沒有券商金鑰或下單能力。

若有其他獨立市場快取，可明確指定來源，工具只讀三個公開資料表、拒絕正式 app DB 路徑：

```powershell
python -m core.paper_validation --output data/paper_validation/forward_2026_10_03_v1 --market-db D:/quant_app/data/paper_validation/market_cache/market.db --sox-csv D:/quant_app/data/paper_validation/market_cache/sox.csv
```

程式／參數／套件版本變更時同一實驗會拒絕接續；先審查原因，建立新的實驗名稱，保留舊資料。不要把初始化驗證腳本當作每日更新器；它使用隨附 seed，日後可能過期。本機原始封存不會隨 GitHub／Render 發布移到伺服器，伺服器也可能使用不同 Python／套件版本；不要把同名資料夾直接當同一實驗。

## 資料與成本限制

現有行情 SQLite 混有自行還原價格，尚未分離原始報價與還原價。以 ±11% 收盤跳動推估公司行動不等於官方除權息資料；也可能漏掉一般現金股利。紙上成交應保存首次取得的原始開盤／報價，另核對官方股利、分割及減資，不能用未來改寫後的價格重算先前績效。官方除權息計算有事件與参考價資料，但該表也明示不含部分減資事件。[證交所計算結果表](https://www.twse.com.tw/exchangeReport/TWT49U?response=html)

籌碼與 SOX 日期不是發布時間戳；資料完整、沒有程式前視，仍不能直接證明訊號當時所有資料均已公布。50 檔固定股票池有選股與存活者偏差，前瞻驗證只適用此凍結股票池，不能外推成全台股有效。

另外，新模型用 006208 報價日曆，舊 app 原策略用股票日期聯集。公開 seed 中早期有 152 個股票交易日沒有 ETF 報價，最後一日為 2018-04-12；三個月份的官方抽查確認其中 5 日是 ETF 零成交、無收盤，不能叫休市日。本次基準在共同 ETF 日曆上重算原 60 日／跳 10 日規則與閘門，排程起點與目前模型一致；它不是舊 app 全部歷史結果的逐筆重現。實際模型日曆未修改。[抽查結果](research/data_fidelity_2026_10_03/summary.md)

0.3% 每單位換手是既有研究假設，不是券商實際費稅。台股普通股票賣出稅為價金 0.3%，佣金由券商訂定；006208 等 ETF 的賣出稅另為 0.1%。實際成交盤須分買／賣、商品種類與最低手續費，另計滑價。[證交所交易制度](https://www.twse.com.tw/en/products/system/trading.html)、[ETF 投資問答](https://investoredu.twse.com.tw/pages/TWSE_InvestmentQA.aspx?ID=11&Page=2)

日線開盤價格存在不表示一定能成交；漲跌停、掛單優先、容量、整股／零股與現金交割仍要驗證。整股市場以 1,000 股為交易單位，低資本的 8 檔等權組合可能需要零股；不能把可分割的模型權重直接視為可成交股數。[證交所交易制度](https://www.twse.com.tw/en/products/system/trading.html)

## 外部模擬環境查核（2026-10-03）

| 選項 | 適合驗證什麼 | 限制與入口 |
|---|---|---|
| 永豐 Shioaji simulation | 台股委託、狀態、持倉與損益 API 流程 | 需永豐帳戶與 API Key／Secret；不支援零股、興櫃。官方公開頁未充分交代模擬撮合與費稅，所以不能單憑其報酬排除 overfit。[模擬模式](https://sinotrade.github.io/tutor/simulation/)、[帳戶](https://sinotrade.github.io/tutor/prepare/open_account/)、[金鑰](https://sinotrade.github.io/tutor/prepare/token/) |
| Fugle 模擬環境 | 下單 API 串接 | 官方明示接單後不會模擬成交，不能作策略績效驗證。[事前準備](https://developer.fugle.tw/docs/trading/prerequisites/) |
| 富邦 Neo 測試環境 | 證券 API 功能串接 | 有測試帳號；尚無足夠公開證據確認長期模擬撮合、費稅與帳務保存。[快速上手](https://www.fbs.com.tw/TradeAPI/docs/welcome/) |
| TradingView Paper Trading | 手動保存紙上委託與核對台股操作 | 台股頁可使用紙上盤；TWSE 免費行情延遲 15 分鐘，因此不直接拿畫面報價驗證精確開盤成交。[台股頁](https://www.tradingview.com/markets/stocks-taiwan/)、[行情覆蓋](https://www.tradingview.com/data-coverage/) |

優先使用本地封存記錄；已有永豐帳戶時，Shioaji 可作第二階段的成交流程對照，須使用禁止正式環境的模擬專用金鑰及 `simulation=True`。本次沒有登入、取得金鑰、開戶或送出任何券商委託。

## 150池與原50分開封存（2026-10-04）

App的新150池不會替換原50實驗。原50程式與公開seed已逐檔SHA還原在 `research/paper_forward_2026_10_03/pinned_runtime/`，同一環境可用其真正舊規則verify/replay；原封存protocol、event與snapshot保持不變。原命令 `python research/paper_forward_2026_10_03/capture_latest.py` 已轉向pinned50，`--offline`不更新網路。

新150另存 `data/paper_validation/forward_150_2026_10_04_v1`，獨立cache為 `data/paper_validation/market_cache_150`，只封存訊號，沒有fills/NAV。名單JSON SHA、估值日期、市場別、掛牌下限、TPEx程式及其餘模型來源一起凍結。同日不覆蓋；source/runtime或歷史資料修訂時停止，不強行更改manifest。固定252個已核實共同前瞻交易日比較150與原50的低換手模型，使用20日配對區塊95%區間；任一方缺及時訊號或未核實成交就不能作績效結論。原各自三臂比較計畫保留。

```powershell
# 150每次接續（預設只更新獨立公開cache）
python research/universe_150_2026_10_03/capture_latest.py
# 無網路檢查與同日冪等
python research/universe_150_2026_10_03/capture_latest.py --offline
# 原50精確重播
python research/paper_forward_2026_10_03/pinned_runtime/launch.py replay
```

排程接續不用 `--initialize`；缺初始protocol或身份不一致必須停止。新的候選池不代表過擬合已排除。尚無新的未來交易日或可核對成交，不能把封存初始歷史資料日冒稱新的樣本外報酬。
