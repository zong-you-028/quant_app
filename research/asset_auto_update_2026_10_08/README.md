# Automatic daily assets and chart verification

App change requested October 8: after stored daily quotes, automatically revalue
holdings plus cash and save/display the daily assets point. Taipei `auto_day` is
unique; repeated saves update the same row atomically. Existing/manual records
are retained. Editing an automatic row converts it to a manual row. There is no
backfill and no change to trades, cash, strategy rules or prospective archives.

To avoid slowing the 150-stock update with unnecessary remote journal reads,
the app revalues the first saved quote, each held quote, and the batch end.
An unheld quote cannot change the portfolio value. Worker threads save/read the
journal; the UI loop renders queued views. Missing held prices do not generate
zero-value snapshots; legacy missing-valued points are excluded from the plot.

Validation commands from the repository root:

```powershell
python research/asset_auto_update_2026_10_08/verify_tests.py
python research/asset_auto_update_2026_10_08/verify_ui.py
python research/asset_auto_update_2026_10_08/verify_identity.py
```

The full suite passed **283 tests**. The SQLite guard covered plain paths and
file URIs, rejecting access to `data/market.db`: **0 attempts**, 1,183 isolated
connections across 54 temporary databases. Tests include legacy schema upgrades,
Taipei rollover on a UTC server, concurrent daily saves, preserving manual and
past-day rows, missing/empty values, committed-quote callbacks, save failures,
single-point charts and a UI refresh before batch completion.

Browser QA uses a disposable copy of the public seed with explicitly synthetic
holdings (10 shares of 2330, 20 shares of 2317, cash 1,000) and synthetic price
changes (100 to 110, 50 to 55). It observes assets **3,000 → 3,100 → 3,200**,
the intermediate committed row/UI before the batch ends, and the unchanged
previous-day row. Same-day repeated updates download neither quote again.
These are test values, not real current market prices or investment performance.
All **17 desktop/mobile checks passed**, including scrolling the 390px mobile
viewport to the visible chart and history controls. No horizontal overflow was
present; saved screenshots were visually inspected.
Python market/journal HTTP and external sockets, formal/outside-cache database
access are blocked; browser Flet frontend CDN assets are permitted. Screenshots
and JSON results are stored here. No model/backtest is rerun in this preview.

SQLite behavior is exercised with real disposable databases; the production
PostgreSQL migration/upsert uses its existing journal adapter and compatible SQL.
Production journal access is deliberately not used in QA. Publication verification
is limited to deployment status and a fresh anonymous login page.

Read-only archive checks confirm the original 50- and 150-stock experiment heads,
event counts and frozen 150 source/config/runtime identity are unchanged. There
is no capture, replay of historical returns, source retuning or backfilled fill.
