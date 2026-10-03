"""Old50 continuation bootstrap; the pinned calculation sources remain untouched."""
import argparse
from contextlib import closing, redirect_stdout
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import time
from urllib.parse import urlsplit

import bootstrap
import launch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--host-root", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--time-budget", type=float, default=90)
    args = parser.parse_args()
    cache, archive = args.cache.resolve(), args.archive.resolve()
    if not 0 < args.time_budget <= 90:
        raise ValueError("Refresh request budget must be positive and at most 90 seconds")
    if Path(os.environ.get("APP_DATA_DIR", "")).resolve() != cache or os.environ.get("JOURNAL_DATABASE_URL"):
        raise AssertionError("Independent old50 cache and cleared journal credentials are required")
    formal = (args.host_root / "data/market.db").resolve()
    if (cache / "market.db").resolve() == formal:
        raise ValueError("Formal host app cache is forbidden")
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    bootstrap.source_verify(manifest)
    seeds = json.loads((HERE / "public_seed_manifest.json").read_bytes())
    for name, item in seeds["public_seed_files"].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Pinned public seed was modified: " + name)
    counters = {"formal_database_attempts": 0, "outside_database_attempts": 0,
                "journal_sql_attempts": 0, "external_network_attempts": 0}
    ready, deadline = [False], [None]
    original_connect = sqlite3.connect

    def authorize(action, first, second, database, trigger):
        if action == sqlite3.SQLITE_READ and first and "journal_" in first.lower():
            counters["journal_sql_attempts"] += 1
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def connect(database, *positional, **keyword):
        if str(database) != ":memory:":
            candidate = bootstrap.database_path(database)
            if candidate == formal:
                counters["formal_database_attempts"] += 1
                raise AssertionError("Formal host database is forbidden")
            if not candidate.is_relative_to(cache):
                counters["outside_database_attempts"] += 1
                raise AssertionError("Only the independent old50 public cache may be opened")
        conn = original_connect(database, *positional, **keyword)
        conn.set_authorizer(authorize)
        return conn

    def no_network(*positional, **keyword):
        counters["external_network_attempts"] += 1
        raise AssertionError("Offline old50 continuation forbids network access")

    sqlite3.connect = sqlite3.dbapi2.connect = connect
    import requests
    real_request = requests.sessions.Session.request
    if not args.refresh:
        socket.create_connection = socket.socket.connect = no_network
        requests.sessions.Session.request = no_network
    else:
        def public_request_wrapper(original_request):
            def public_request(session, method, url, **keyword):
                if not ready[0]:
                    raise AssertionError("No public request is permitted before frozen identity validation")
                hostname = (urlsplit(url).hostname or "").lower()
                if urlsplit(url).scheme != "https" or hostname not in {
                    "api.finmindtrade.com", "www.twse.com.tw", "openapi.twse.com.tw",
                    "query1.finance.yahoo.com", "query2.finance.yahoo.com", "fc.yahoo.com",
                    "guce.yahoo.com", "consent.yahoo.com"}:
                    raise AssertionError("Only pinned public market data endpoints may be refreshed")
                remaining = deadline[0] - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Old50 public refresh request budget exhausted")
                supplied = keyword.get("timeout", 8)
                if isinstance(supplied, (tuple, list)):
                    supplied = min(float(value) for value in supplied if value is not None)
                keyword["timeout"] = min(float(supplied or 8), 8., remaining)
                counters["external_network_attempts"] += 1
                return original_request(session, method, url, **keyword)
            return public_request
        requests.sessions.Session.request = public_request_wrapper(real_request)
        try:
            from curl_cffi.requests import Session as CurlSession
            CurlSession.request = public_request_wrapper(CurlSession.request)
        except ImportError:
            pass
    sys.path.insert(0, str(SOURCE))
    from core import paper_validation as pv
    state = pv.verify_archive(archive)
    identity = pv.frozen_identity()
    if state["protocol"]["protocol_hash"] != manifest["original_protocol_hash"]:
        raise ValueError("This continuation is pinned to the original old50 protocol only")
    if identity != state["protocol"]["identity"] or hashlib.sha256(bootstrap.canonical(identity)).hexdigest() != manifest["identity_sha256"]:
        raise pv.ProtocolChanged("Original old50 source/runtime identity differs; stop before network or data initialization")
    import config
    if len(config.UNIVERSE) != 50 or Path(config.DB_PATH).resolve() != cache / "market.db":
        raise AssertionError("Pinned 50-stock configuration must use only its independent market cache")
    before = launch.inventory(archive)
    before_count = len(state["events"])
    ready[0] = True
    with pv._lock(cache):
        if not (cache / "market.db").exists():
            with gzip.open(SOURCE / "data_seed/market.db.gz", "rb") as source, (cache / "market.db").open("xb") as target:
                shutil.copyfileobj(source, target)
        if not (cache / "sox.csv").exists():
            with (SOURCE / "data_seed/sox.csv").open("rb") as source, (cache / "sox.csv").open("xb") as target:
                shutil.copyfileobj(source, target)
        with closing(connect((cache / "market.db").as_uri() + "?mode=ro", uri=True)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables - {"ohlcv", "chip_weekly", "stock_info", "meta"}:
            raise ValueError("Independent public cache contains unexpected/private tables; stop without reading their records")
        from core import data_pipeline as dp, market_regime
        # Source identity was already proven; using only an independently
        # bootstrapped seed avoids rehydrating any later ordinary-app seed.
        dp._INITIALIZED_PATHS.add(config.DB_PATH)
        update, sox_pending = None, False
        with redirect_stdout(sys.stderr):
            if args.refresh:
                started = time.monotonic()
                deadline[0] = started + args.time_budget
                symbols = list(dict.fromkeys([*config.UNIVERSE, config.BENCHMARK_SYMBOL]))
                update = dp.update_symbols(symbols, time_budget_seconds=args.time_budget, max_attempts=1)
                if update.get("busy"):
                    return {"status": "market_cache_busy", "isolation": counters}
                sox = market_regime._cached_sox()
                if sox is None or sox.index[-1] + __import__("pandas").Timedelta(days=3) < dp._last_trading_day():
                    if time.monotonic() < deadline[0]:
                        # Both requests and curl_cffi are bounded before each
                        # request; response parsing/commits can finish later.
                        market_regime.refresh_sox()
                    else:
                        sox_pending = True
            captured = pv.capture(archive, cache / "market.db", cache / "sox.csv")
    after = launch.inventory(archive)
    from capture_latest import assert_prefix_unchanged
    assert_prefix_unchanged(before, after)
    event = captured["event"]
    result = {"status": captured["status"], "reference_commit": manifest["reference_commit"],
              "universe_size": len(config.UNIVERSE), "identity_sha256": manifest["identity_sha256"],
              "protocol_hash": state["protocol"]["protocol_hash"], "data_date": event["data_date"],
              "recorded_at": event["recorded_at"], "event_hash": event["event_hash"],
              "event_count_before": before_count, "event_count_after": len(pv.verify_archive(archive)["events"]),
              "reasons": event["reasons"], "current_block_reasons": captured.get("current_block_reasons", []),
              "current_snapshot_changed": captured.get("current_snapshot_changed", False),
              "unrecorded_dates": event["unrecorded_dates"], "execution_status": event["execution_status"],
              "fills": [], "performance": None, "update": update, "sox_refresh_pending": sox_pending,
              "refresh_explicitly_enabled": args.refresh, "isolation": counters,
              "worker_isolation": event.get("isolation"), "existing_archive_prefix_unchanged": True}
    return result


if __name__ == "__main__":
    result = main()
    sys.stdout.buffer.write(bootstrap.canonical(result))
