# Original 50-stock prospective experiment runtime

The model code here is the actual `f49b4cae3c0f5c55acfd609aae5ce7abf6e672a0` source, byte-for-byte matched to the original experiment's recorded SHA256. The exporter restores the Windows CRLF checkout only for files whose frozen hash requires it. `.gitattributes` preserves those exact source bytes across future Git checkouts.

It computes the real original identity and verifies it against the unchanged original protocol. It does not overwrite `config.UNIVERSE` or fake `frozen_identity()`. The ordinary app can independently change its stock pool and code. The original public Git market/SOX seed is retained separately under `source/data_seed`, with its own SHA manifest, so expanding the ordinary seed does not affect it.

Run from `D:\quant_app`:

```powershell
python research\paper_forward_2026_10_03\pinned_runtime\launch.py verify
python research\paper_forward_2026_10_03\pinned_runtime\launch.py replay
python research\paper_forward_2026_10_03\pinned_runtime\launch.py capture-copy
```

`verify` and `replay` read the original archive. `capture-copy` copies its public protocol/events/snapshots to a temporary directory, reconstructs a public market cache from the last sealed snapshot, and runs the actual old capture function on that disposable copy. It never appends to the original experiment. Snapshot SHA checks, original inventory checks, isolated `APP_DATA_DIR`, cleared journal credentials, and blocked network protect the original records.

Only independent market caches may be supplied with `capture-copy --market-db <path> --sox-csv <path>`. The real app `data\market.db` is refused. That command captures new inputs only on a disposable copy.

The user's existing two-hour automation also authorizes appending future dates to the original 50-stock experiment. Its separate continuation helper uses the pinned real source and old public seed, and the independent `data\paper_validation\market_cache`:

```powershell
python research\paper_forward_2026_10_03\pinned_runtime\capture_latest.py --offline
python research\paper_forward_2026_10_03\pinned_runtime\capture_latest.py --refresh
```

Offline is the default. `--refresh` explicitly permits only public market endpoints with a 90-second request budget checked before each request; response parsing/commits may finish later. The source/runtime identity and old protocol are verified before HTTP is enabled. Existing protocol/event/snapshot hashes must remain unchanged; new files may only be appended. Locks are never force-deleted. A stale, revised, missing or incompatible source stops executable signal recording. This remains the original 50-stock experiment, independently of the ordinary app's new pool.

The original runtime versions are Python 3.13.14 / CPython, NumPy 2.2.6, pandas 2.3.2, SQLite 3.50.4. A version mismatch fails identity verification. This directory preserves model source, not a redistributable Python binary or virtual environment. The launcher does not install or downgrade dependencies automatically.

No orders or performance are produced. The original signal targets remain pending future verified executable quotes; this is a decision archive and replay tool.
