# Deploy to Render

**Version update date:** 2026-08-30

This project is ready to run as a Render web service. Its SQLite database and
market-data cache live under `APP_DATA_DIR`; in Render, that directory is the
persistent disk mounted at `/var/data`.

1. Create a private [GitHub repository](https://github.com/new) and push this project to it. Do not add
   a FinMind token to the repository.
2. In [Render](https://dashboard.render.com/), select **New** > **Blueprint**, then connect that repository.
   Render reads `render.yaml` and creates the `quant-app` web service.
3. In the service environment settings, enter `FINMIND_TOKEN` if you use one.
   Leave it blank to use FinMind's unauthenticated quota.
4. Deploy. Render displays the public `onrender.com` URL after the build
   finishes. Open that URL from any device.

The attached 1 GB persistent disk is required: without it, trades, journal
entries, and downloaded price data are lost when the service restarts. A
persistent disk requires a paid Render web-service plan. The default `starter`
plan in `render.yaml` keeps the service available without your computer on.

## Existing local data

The current local `data/market.db` is intentionally excluded from Docker
builds so personal data is not published by accident. A new deployment starts
with an empty database and downloads price history when needed. Keep a backup
of the existing SQLite file if you need the current journal migrated.

## Security

The Render URL is publicly reachable. Do not store sensitive information in
the journal until access control has been added. For personal use, add a login
gate before sharing the URL.
