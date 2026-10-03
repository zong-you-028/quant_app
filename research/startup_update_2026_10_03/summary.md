# Startup update verification — 2026-10-03

- Real TWSE/FinMind update on a temporary copy of the previous public market seed: 51 symbols updated in 21.26 seconds; zero stale, failed or pending symbols. Formal database access is forbidden by the probe.
- All 51 prices and required shareholding data reached 2026-10-02. A second current-cache check took 0.3 seconds. Exported seed contains only `ohlcv`, `chip_weekly`, `stock_info`; SQLite integrity check passed. SOX public cache also ends at 2026-10-02.
- Full regression suite: 131 passed, formal DB connection attempts 0, 516 isolated connections across 22 temporary databases. See `test_verification.json`.
- Live yfinance 0.2.65 incremental SOX refresh also passed using a temporary public cache ending 2026-09-28: preserved the history from 2014-06-02 and updated through 2026-10-02 (3,104 rows). See `live_sox_result.json`; the deployed dependency is pinned to this tested version.
- Chrome 153 / Flet 0.85.2 desktop 1440 and mobile 390px: actual fresh-cache startup succeeds with external market downloads forbidden; simulated partial/failure scenarios show shared progress, disabled analysis, continuation, visible source failure and successful retry. Both journals remained empty across all five record tables. See `ui_browser_result.json` and screenshots.
- Partial/failure screenshots are controlled UI scenarios, not evidence of an actual network timeout. Live timing is local and does not establish Render egress connectivity or guarantee total update duration. Login gating remains in the deployed app; the isolated QA entrypoint bypasses login only on localhost.
- Model parameters, selection/execution timing and prior overfitting evidence are unchanged. Updating market data is not new evidence against overfitting.
