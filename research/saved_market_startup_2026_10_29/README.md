# Restore saved market data before showing the application

The authenticated screen previously read the bundled October 2 data before the
background daily-update workflow restored cloud bundles. Move public-cache
restoration ahead of screen construction. A process-wide lock and cache ensure
cold startup and simultaneous/repeated sessions share one restore. Failed cloud
restoration shows a retry screen; it cannot expose seed dates as loaded data.
Once restored, a bounded SOX catch-up uses the existing public-index updater.
SOX source failures cannot hide already-restored stocks. The regular daily
updater still handles new dates and existing daily asset snapshots.

Label the fixed component-list snapshot separately from quote freshness. The
stock-pool snapshot, strategy parameters and frozen research remain unchanged.

Upgrade yfinance from 0.2.65 to 1.7.0, including the upstream curl_cffi >=0.16
compatibility fix. Use single-ticker history with explicit errors and an end
date after the last completed US session. A successful SOX save logs only its
public date, target and persistence status. The Yahoo chart endpoint was also
tested read-only but returned 429; no unreliable fallback was added.

New tests cover concurrent/repeated opens, first-screen quote dates, newer
local quotes on subsequent opens, SOX failure isolation, restoration failure
UI and snapshot-date labelling. All use synthetic public bundles and temporary
SQLite. The regression harness blocks formal SQLite and live HTTP and excludes
the historical identity test that needs the separate original Python runtime.

Source: https://github.com/ranaroussi/yfinance/blob/main/CHANGELOG.rst

Verification: 370 tests passed, one historical-runtime identity test deselected
in 54.85 seconds. Formal-database and live-HTTP attempts were both zero in this
suite. The temporary-cache HTTP/WebSocket startup check also passed. The first
run exposed a test fixture that changed its SQLite adapter without changing its
cache-key path; the fixture now supplies its real disposable path, preserving
the production cache-key and restoration guards.
