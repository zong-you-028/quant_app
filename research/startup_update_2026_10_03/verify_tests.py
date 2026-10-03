"""Full regression suite with access to the formal SQLite database forbidden."""
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
import pytest

formal = Path(config.DB_PATH).resolve()
original = sqlite3.connect
paths = set()
attempts = 0
connections = 0


def guarded(database, *args, **kwargs):
    global attempts, connections
    path = str(database)
    if path != ":memory:":
        resolved = Path(path).resolve()
        if resolved == formal:
            attempts += 1
            raise AssertionError("Regression tests may not access the formal database")
        paths.add(str(resolved))
    connections += 1
    return original(database, *args, **kwargs)


sqlite3.connect = sqlite3.dbapi2.connect = guarded
started = time.monotonic()
try:
    status = pytest.main(["-q"])
finally:
    sqlite3.connect = sqlite3.dbapi2.connect = original
    report = {"pytest_exit_code": status, "elapsed_seconds": round(time.monotonic()-started, 2),
              "formal_database_attempts": attempts, "isolated_connections": connections,
              "isolated_database_count": len(paths)}
    Path(__file__).with_name("test_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
sys.exit(status)
