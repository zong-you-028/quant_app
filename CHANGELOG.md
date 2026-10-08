# Changelog

## 2026-10-08

- Automatically upsert a Taipei daily total-assets snapshot after held quotes are saved, and recheck at batch end. Same-day retries and concurrent sessions retain one automatic row; manual snapshots and earlier days are preserved. Editing an automatic snapshot converts it to a manual record.
- Refresh portfolio values, asset history and the line chart while the update continues. The chart is visible independently of collapsed history details and supports a single first-day point. Unheld quotes skip redundant remote valuation reads.
- Store held-quote valuation dates and freshness, preserve valid snapshots when a held quote is unavailable, and report asset-write failures without redownloading successful quotes. Existing missing-valued snapshots no longer plot as zero.
- Passed 283 isolated regression tests and 17 desktop/mobile browser checks, with zero formal database attempts; prospective source identities and both original archive heads remain unchanged. Browser QA uses synthetic holdings and controlled quotes, never real investment records.

## 2026-10-07

- Added an app-only Taipei daily update policy backed by SQLite: committed successful downloads are reused across clicks and processes; failures and unfinished symbols can resume without refetching completed symbols. Same-day newer targets remain visibly stale until the next day; current-cache checks do not consume the daily allowance.
- Persisted SOX closes in `market_index_prices` and synchronized the existing CSV; added transactional DB leases for concurrent app sessions. Frozen model, research acquisition sources and both prospective archive identities are unchanged.
- Updated the UI and usage guide to show database persistence, daily reuse and the recommended post-18:00 update time. Added isolated persistence/restart/resume/concurrency tests.
- Passed 271 isolated regression tests and 34 desktop/mobile checks; the controlled gap was committed once and reused across a fresh browser session, with no formal DB access. Read-only archive identity checks matched both existing experiment heads.

## 2026-10-04

- Expanded the single app pool to a fixed snapshot of 150 quoted ordinary shares by estimated issued-share market cap: 126 TWSE and 24 TPEx. Archived official responses, explicit unpriced exclusions, private-share/listed-cap differences and membership bias limitations. Holdings remain capped at eight.
- Bundled complete public prices/shareholding through 2026-10-02, added TPEx daily/monthly updates, rejected non-finite/unpublished quotes, and filtered current-market pre-listing history at download and read time. Recent IPO warmup is separated from freshness so it cannot falsely disable all reconciliation actions.
- Compared unchanged rules on nine fixed cost/lag conditions. Normal historical CAGR is 60.03% versus 30.42%, but drawdown worsens to -36.92% versus -29.24%; current membership selection and survivorship bias remain. No overfitting claim is made.
- Pinned the exact old50 sources/public seed and kept its original prospective archive unchanged. Added a separate 150-pool signal recorder with metadata/source fingerprints and a predeclared parallel old50 comparison; neither archive implements fills/NAV.
- Passed 259 regression tests with zero formal-database access, plus 35 focused prospective/integrity checks after extending the protocol. Actual isolated app calculation reproduced the normal 150 comparison result and all 151 source targets were current.

## 2026-10-03

- Added retrospective end-to-end prefix/future-contamination checks, fixed cost/execution/source-delay stresses, year-concentration diagnostics and an explicitly common-calendar original-rule baseline. Results remain retrospective and do not rule out overfitting; active app rules were not changed.
- Added an isolated prospective signal archive with source/config fingerprints, real timestamps, public snapshot hashes, append-only records and visible stale/revision/gap states. It does not submit orders, backfill fills or claim realized paper-trading returns. Documented Taiwan paper-trading options and a frozen evaluation protocol.
- Added read-only journal reconciliation showing the currently executable model portfolio, kept/missing/outside stocks and separate pending targets. Eight model slots remain fixed; the top-16 buffer never creates a sixteen-stock portfolio. Early purchases and unheld upcoming removals are identified to avoid buy/sell churn.
- Rechecks refresh journal positions without rerunning the model; journal refreshes update the comparison and market updates invalidate it. Missing execution metadata, stale dates and journal errors never produce live action lists. Candidate labels now explicitly describe model-list changes.
- Verified reconciliation with 149 regression tests and 18 desktop/mobile checks on isolated public market data and synthetic positions; real journal records stayed untouched. Return values exactly matched the prior execution engine on the same snapshot.
- Made startup updates visible and resumable, removed duplicate holdings downloads, bounded nested retries and prevented simultaneous shared-cache updates. Journal reads during updates run off the UI loop.
- Fixed recursive SOX failure recovery, added incremental atomic SOX caching and shipped public market/SOX seeds through 2026-10-02. An isolated 51-symbol live update completed in 21.26 seconds; the next cache check took 0.3 seconds. These timings are local, not Render guarantees.
- Verified the update changes with 131 regression tests, a formal-database access guard and desktop/mobile browser checks for partial completion, source errors and retry recovery.
- Unified stock analysis and rotation under the buffered multi-horizon momentum model, with clearer candidate labels and an in-app usage guide.
- Added delayed open execution, shared-calendar model ranking, stale-data guards, atomic cache updates and independent shareholding refreshes.
- Added corporate-action journal tools and documented their accounting behavior.
- Published strategy comparisons, schedule sensitivity checks and retrospective overfitting diagnostics. Overfitting remains unexcluded; the model awaits forward validation.
- Added the usage manual and regression coverage: 116 tests passed with no access to the live database, plus isolated desktop and 390px UI validation.

## 2026-08-30

- Added Render deployment configuration for the Flet web application.
- Added persistent cloud storage support through the `APP_DATA_DIR` setting.
- Added Docker build files and deployment instructions.
