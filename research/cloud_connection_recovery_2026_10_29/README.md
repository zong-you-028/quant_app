# Neon pooled market-cache connection recovery

The updater stops before downloads if shared-cache restoration fails. The
PostgreSQL gateway passed `statement_timeout` and `lock_timeout` as startup
`options`, which Neon/PgBouncer does not support. The previous adapter test
accepted every connect argument and therefore did not exercise this failure.

Connect without startup options, use explicit transactions with `SET LOCAL`
timeouts for schema creation, reads and each resource save, and disable automatic
prepared statements. Autocommit keeps the connection idle between transactions;
reads finish their transaction before restoring the local SQLite cache. Existing
conflict inserts, row locks and per-resource commits remain in place.

Add a pooler regression that rejects startup options and resets transaction
settings between operations. It verifies saves, merges and reads through the
actual gateway with the same simulated protocol restriction. Error reporting
classifies failures without logging driver text, URLs, credentials or source
rows. The service runs a read-only connectivity probe using its existing
environment: SELECT 1, cache-table existence/access checks, resource count and
validation of one public cache payload. It never reads journal tables or changes
authentication. Unknown update dates display as unconfirmed rather than None.

The full regression harness blocks production SQLite access and external HTTP.
It excludes the one historic research identity check which requires a separate
Python 3.13 runtime; no frozen source, protocol or archive is changed.

Verification: 364 tests passed, 1 historical-runtime test deselected in 59.21s.
Production-database and external-HTTP attempts were both zero. Real local HTTP
200 and WebSocket startup also passed. The retained harness was run with
`PYTEST_ADDOPTS=--deselect=test_pinned_paper_runtime.py::test_actual_identity_verify_ignores_host_import_paths_and_journal_credentials`.

References:
- https://neon.com/docs/connect/connection-errors (unsupported startup parameters)
- https://www.psycopg.org/psycopg3/docs/advanced/prepare.html
