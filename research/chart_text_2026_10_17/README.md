# Render chart Chinese text

The asset-history PNG used Windows font family names which are absent from the
minimal Linux image. Bundle the unmodified, licensed Noto Sans CJK TC regular
font and register it before the app creates charts. No runtime font download,
system font installation, journal writes or strategy changes are required.
The Docker build verifies registration and the asset-chart glyphs on Linux.

`verify_chart.py --output <folder>` disables operating-system font discovery,
removes all system font entries and forbids every SQLite/HTTP request. Synthetic
single-point and seven-point histories exercise Agg PNG rendering and the Flet
image path. The baseline reproduces 42 missing-glyph warnings; the corrected
charts have zero, resolve to the shipped font and keep title/legend inside the
image. A lone point displays its actual date instead of a multi-year auto axis.

`before.png`, `single.png` and `multiple.png` are synthetic previews. They contain
no real holdings, balances or investment performance. The font's upstream
revision, checksum and license are included under `resources/fonts/`.

The tracked release suite passed **357 tests** (355 existing and 2 new), with
zero formal-database or live-HTTP attempts. Frozen source/archive identities
were checked read-only and still match. The preview images were visually
inspected; actual production balances and journals were not opened.
