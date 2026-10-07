"""Full regression suite; reject plain and URI access to the formal DB."""
import json
from pathlib import Path
import sqlite3
import sys
import time
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
import pytest

formal = (ROOT / "data/market.db").resolve()
original = sqlite3.connect
paths = set()
attempts = 0
connections = 0


def resolved_sqlite_path(database):
    value = str(database)
    if value == ":memory:" or value.startswith("file::memory:"):
        return None
    if value.startswith("file:"):
        value = unquote(value[5:].split("?", 1)[0])
        if value.startswith("///"):
            value = value.lstrip("/")
    return Path(value).resolve()


def guarded(database, *args, **kwargs):
    global attempts, connections
    resolved = resolved_sqlite_path(database)
    if resolved == formal:
        attempts += 1
        raise AssertionError("Regression tests may not access the formal database")
    if resolved is not None:
        paths.add(str(resolved))
    connections += 1
    return original(database, *args, **kwargs)


sqlite3.connect = sqlite3.dbapi2.connect = guarded
started = time.monotonic()
status = 99
try:
    status = pytest.main(["-q"])
finally:
    sqlite3.connect = sqlite3.dbapi2.connect = original
    report = {"pytest_exit_code": int(status), "elapsed_seconds": round(time.monotonic()-started, 2),
              "formal_database_attempts": attempts, "isolated_connections": connections,
              "isolated_database_count": len(paths), "guard_includes_sqlite_file_uri": True}
    Path(__file__).with_name("test_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
sys.exit(status)
