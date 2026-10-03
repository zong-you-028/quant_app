"""Isolated bootstrap for unchanged pinned model sources. No external I/O."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def database_path(database):
    text = str(database)
    candidate = unquote(urlsplit(text).path) if text.startswith("file:") else text
    if len(candidate) > 2 and candidate[0] == "/" and candidate[2] == ":":
        candidate = candidate[1:]
    return Path(candidate).resolve()


def source_verify(manifest):
    for name, item in manifest["source_files"].items():
        actual = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError("Pinned source was modified: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "replay", "capture-copy"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--host-root", type=Path, required=True)
    parser.add_argument("--market-db", type=Path)
    parser.add_argument("--sox-csv", type=Path)
    args = parser.parse_args()
    if not os.environ.get("APP_DATA_DIR") or os.environ.get("JOURNAL_DATABASE_URL"):
        raise AssertionError("Launcher must configure an isolated cache and clear journal credentials")
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    source_verify(manifest)
    counters = {"formal_database_attempts": 0, "external_network_attempts": 0,
                "journal_sql_attempts": 0}
    formal = (args.host_root / "data/market.db").resolve()
    original = sqlite3.connect

    def forbidden(*positional, **keyword):
        counters["external_network_attempts"] += 1
        raise AssertionError("External network is forbidden in the pinned replay launcher")

    def authorize(action, first, second, database, trigger):
        if action == sqlite3.SQLITE_READ and first and "journal_" in first.lower():
            counters["journal_sql_attempts"] += 1
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def connect(database, *positional, **keyword):
        if str(database) != ":memory:" and database_path(database) == formal:
            counters["formal_database_attempts"] += 1
            raise AssertionError("Formal host database access forbidden")
        conn = original(database, *positional, **keyword)
        conn.set_authorizer(authorize)
        return conn

    sqlite3.connect = sqlite3.dbapi2.connect = connect
    socket.create_connection = socket.socket.connect = forbidden
    import requests
    requests.sessions.Session.request = forbidden
    # Priority comes from an actual directory containing the historic bytes;
    # we do not replace config.UNIVERSE or monkeypatch the identity function.
    sys.path.insert(0, str(SOURCE))
    from core import paper_validation as pv
    state = pv.verify_archive(args.archive)
    if state["protocol"]["protocol_hash"] != manifest["original_protocol_hash"]:
        raise ValueError("Archive is not the pinned original protocol")
    identity = pv.frozen_identity()
    if identity != state["protocol"]["identity"]:
        raise pv.ProtocolChanged("Pinned source/runtime identity differs from original protocol; use the recorded dependency versions")
    if hashlib.sha256(canonical(identity)).hexdigest() != manifest["identity_sha256"]:
        raise ValueError("Pinned manifest identity hash differs from the actual imported sources/runtime")
    import config
    if len(config.UNIVERSE) != 50 or config.ROTATION_TOP_K != 8:
        raise AssertionError("Historic runtime must retain the original 50-stock pool")
    before_count = len(state["events"])
    if args.mode == "verify":
        result = {"status": "pinned_archive_verified", "head_hash": state["head_hash"]}
    elif args.mode == "replay":
        result = pv.replay_sealed(args.archive)
    else:
        if not os.environ.get("PINNED_CAPTURE_IS_DISPOSABLE_COPY"):
            raise AssertionError("Capture can run only on the launcher's disposable archive copy")
        if args.market_db:
            if not args.sox_csv:
                raise ValueError("Explicit independent market cache requires --sox-csv")
            market_db, sox_csv = args.market_db, args.sox_csv
        else:
            latest = state["events"][-1]
            snapshot = json.loads(gzip.decompress((args.archive / "snapshots" /
                (latest["snapshot_hash"] + ".json.gz")).read_bytes()))
            if pv._hash(snapshot) != latest["snapshot_hash"]:
                raise ValueError("Public snapshot differs from the sealed event")
            market_db = Path(os.environ["APP_DATA_DIR"]) / "offline_public_source.db"
            with connect(market_db) as conn:
                for table, columns in pv.TABLES.items():
                    item = snapshot["tables"][table]
                    if tuple(item["columns"]) != columns:
                        raise ValueError("Unexpected public snapshot table schema")
                    definition = ",".join('"' + col + '" ' +
                        ("TEXT" if col in ("symbol", "date", "name") else "REAL") for col in columns)
                    conn.execute(f'CREATE TABLE "{table}" ({definition})')
                    conn.executemany(f'INSERT INTO "{table}" VALUES (' + ",".join("?" for _ in columns) + ")", item["rows"])
            conn.close()
            import pandas as pd
            sox_csv = Path(os.environ["APP_DATA_DIR"]) / "offline_sox.csv"
            pd.Series({pd.Timestamp(d): v for d, v in snapshot["sox"]}, name="close").to_csv(sox_csv)
        result = pv.capture(args.archive, market_db, sox_csv)
    result["pinned_launcher"] = {"reference_commit": manifest["reference_commit"],
        "universe_size": len(config.UNIVERSE), "runtime": identity["runtime"],
        "identity_sha256": manifest["identity_sha256"], "isolation": counters,
        "loaded_source_root": str(SOURCE.resolve()), "original_archive_writes": False,
        "event_count_before": before_count, "event_count_after": len(pv.verify_archive(args.archive)["events"])}
    sys.stdout.buffer.write(canonical(result))


if __name__ == "__main__":
    main()
