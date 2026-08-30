# Deploy to Render

**Version update date:** 2026-08-30

This project is ready to run as a Render web service. Trading-journal data is
stored in Neon PostgreSQL; the market-data cache remains in local SQLite.

1. Create a private [GitHub repository](https://github.com/new) and push this project to it. Do not add
   a FinMind token to the repository.
2. In [Render](https://dashboard.render.com/), select **New** > **Blueprint**, then connect that repository.
   Render reads `render.yaml` and creates the `quant-app` web service.
3. In the service environment settings, set `JOURNAL_DATABASE_URL` to the Neon
   PostgreSQL connection string. Enter `FINMIND_TOKEN` too if you use one.
4. Deploy. Render displays the public `onrender.com` URL after the build
   finishes. Open that URL from any device.

Journal entries survive Render restarts in Neon. The Docker image includes a
journal-free market/name seed, so a fresh Render instance starts from the same
real-data baseline as local runs and then applies daily market updates.

## Existing local data

The current local `data/market.db` is intentionally excluded from Docker
builds so personal data is not published by accident. A new deployment starts
with empty Neon journal tables and downloads price history when needed. Existing
SQLite journal rows are not migrated automatically.

## Security

The Render URL is publicly reachable. Do not store sensitive information in
the journal until access control has been added. For personal use, add a login
gate before sharing the URL.
