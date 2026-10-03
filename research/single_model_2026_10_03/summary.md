# 單一模式驗證

2026-10-03 最後源碼版本：完整 pytest 116 項通過；正式 DB 存取嘗試 0 次。
518 次連線全為 22 個臨時測試 DB，證據見 `test_verification.json`。

Chrome 153／Selenium 實際驗證 1440px 桌面與 390px 窄版：固定低換手模型、
「過擬合未排除」狀態、使用流程展開、輪動計算、個股計算及過期推薦停用均通過。
輪動結果不再重複顯示完整狀態卡，上方共用卡隨計算更新。
詳見 `ui_browser_result.json` 與 `ui_*.png`。

UI 預覽使用 `APP_DATA_DIR` 指向本目錄的 `preview_data`，
`QUANT_APP_OFFLINE=1`，並清空 `JOURNAL_DATABASE_URL`；僅載入隨附真實行情種子
與既有 SOX 快取。未按資料更新或帳本操作。隔離帳本五類紀錄皆為 0 筆，
預覽服務與暫存副本已在驗證後移除。

`verify_ui.py` 只應針對以上配置的本機隔離預覽執行。它不啟動預覽，也不更新
行情；重新執行前需先建立隔離預覽服務，不能直接指向日常使用的 app。

本次過擬合診斷見 [報告](../../OVERFITTING_REPORT.md)；操作方法見
[使用手冊](../../USAGE.md)。
