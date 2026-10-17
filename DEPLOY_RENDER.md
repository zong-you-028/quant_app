# Deploy to Render

**Version update date:** 2026-10-17

This project is ready to run as a Render web service. Trading-journal data is
stored in Neon PostgreSQL. Public market bundles also persist in PostgreSQL;
SQLite is a disposable local read cache, restored before updates start.

1. Create a private [GitHub repository](https://github.com/new) and push this project to it. Do not add
   a FinMind token to the repository.
2. In [Render](https://dashboard.render.com/), select **New** > **Blueprint**, then connect that repository.
   Render reads `render.yaml` and creates the `quant-app` web service.
3. In the service environment settings, set `JOURNAL_DATABASE_URL` to the Neon
   PostgreSQL connection string. Set `APP_PASSWORD` to a strong, unique password;
   the web app stays locked if this variable is missing. Enter `FINMIND_TOKEN`
   too if you use one.
4. Deploy. Render displays the public `onrender.com` URL after the build
   finishes. Open that URL from any device.

Journal entries survive Render restarts in Neon. The Docker image includes a
journal-free market/name seed. A fresh instance then restores previously saved
market bundles from Neon before checking for missing updates.
The current seed includes public TWSE/FinMind market rows and SOX through
2026-10-02, with no journal records. SOX is also initialized for custom
`APP_DATA_DIR` locations; the tested yfinance dependency is included.

Startup and manual updates share one progress display and update lock. Batches
have a 90-second request budget and automatically continue pending work. SOX
and 006208 are prioritized; failed front-of-queue symbols cannot repeatedly
starve later work. Historical months commit in chronological order. Upstream
failures remain visible; requests are not
repeated separately for journal holdings. Request budgets are checked between
calls, so parsing, commits and cloud persistence can extend total elapsed time.
There is a 64-batch cap, finite source retries, and a no-progress stop. A cloud
write failure pauses further downloads and leaves an upload-retry marker.

## Durable market cache on the existing free service

Render's free filesystem is ephemeral: local SQLite changes are lost on restart,
redeploy or idle spin-down. Setting `APP_DATA_DIR` alone does not make that path
durable. See [Render's filesystem documentation](https://render.com/docs/free#local-files-lost-on-redeploy).

The app uses `MARKET_DATABASE_URL` when set, otherwise the existing
`JOURNAL_DATABASE_URL`. No paid Render disk or plan change is needed. Its role
must be allowed to create/read/write `quant_market_cache_v1`. The app creates
this separate table automatically on first access; it never copies journal
tables into market bundles or changes their schema.

Bundles explicitly allow only public OHLCV, foreign-shareholding, stock names,
SOX and validated update/backfill metadata. They are gzip JSON, keyed by resource,
with row-locked merges that retain newer cloud history. A cold instance restores
these resources into SQLite; SOX also restores its atomic CSV cache. Successful
download metadata is checked against actual saved dates before reuse.

Cloud errors are visible and never count as durable success. If restoration
fails, downloading does not start. If saving fails after a quote commit, the
local quote remains available; retry uploads it without another market request.
Until a cloud save succeeds, an instance restart can still lose that local-only
progress. First use after upgrading may need one catch-up from the bundled seed.

## Existing local data

The current local `data/market.db` is intentionally excluded from Docker
builds so personal data is not published by accident. A new Neon database starts
with empty journal tables; an existing configured Neon journal is preserved.
Existing SQLite journal rows are not migrated automatically. Only explicitly
allowed public market fields are adopted into the cloud cache.

## Security

The Render URL is publicly reachable. Do not store sensitive information in
the journal without a strong application password. The optional "remember me"
setting stores a derived login token in that browser, never the plaintext
password. Changing `APP_PASSWORD` invalidates remembered logins. Anyone who
knows the password can view and modify the shared journal, so use a unique password.
