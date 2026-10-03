# Changelog

## 2026-10-03

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
