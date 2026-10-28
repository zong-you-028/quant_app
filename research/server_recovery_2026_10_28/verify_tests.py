"""Full root regression suite with formal-database and live-HTTP guards.

The suite retains its normal synthetic fixtures and temporary databases. Real
requests cannot leave this process, while tests may replace request methods
with their own mocks. Results and terminal output stay beside this harness.
"""
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The suite opts in to fake cloud adapters in individual tests only.
os.environ["MARKET_DATABASE_URL"] = ""
os.environ["JOURNAL_DATABASE_URL"] = ""
os.environ["FINMIND_TOKEN"] = ""

import pytest
import requests

from core.robustness_audit import database_path


class Tee:
    def __init__(self, original, log):
        self.original = original
        self.log = log

    @property
    def encoding(self):
        return self.original.encoding

    def write(self, value):
        self.log.write(value)
        return self.original.write(value)

    def flush(self):
        self.log.flush()
        self.original.flush()

    def isatty(self):
        return False


class SuiteResults:
    def __init__(self):
        self.counts = Counter()
        self.collected = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if hasattr(report, "wasxfail"):
                self.counts["xpassed" if report.passed else "xfailed"] += 1
            else:
                self.counts[report.outcome] += 1
        elif report.failed:
            self.counts["errors"] += 1
        elif report.when == "setup" and report.skipped:
            self.counts["skipped"] += 1

    def pytest_collectreport(self, report):
        if report.failed:
            self.counts["collection_errors"] += 1

    def pytest_sessionfinish(self, session, exitstatus):
        self.collected = session.testscollected


def main():
    os.chdir(ROOT)
    formal = (ROOT / "data" / "market.db").resolve()
    # Verify URI normalization without attempting to open the protected file.
    uri_forms = [formal.as_uri() + "?mode=ro",
                 "file:" + quote(str(formal).replace("\\", "/")) + "?mode=ro"]
    if any(database_path(value) != formal for value in uri_forms):
        raise AssertionError("SQLite URI normalization must identify the formal database")
    original_connect = sqlite3.connect
    original_dbapi2_connect = sqlite3.dbapi2.connect
    original_request = requests.sessions.Session.request
    counters = {"formal_database_attempts": 0, "isolated_connections": 0,
                "network_attempts": 0}
    isolated_paths = set()

    def guarded_connect(database, *args, **kwargs):
        path = database_path(database)
        if path == formal:
            counters["formal_database_attempts"] += 1
            raise AssertionError("Regression tests may not access the formal database")
        if path is not None:
            isolated_paths.add(str(path))
        counters["isolated_connections"] += 1
        return original_connect(database, *args, **kwargs)

    def no_live_http(*args, **kwargs):
        counters["network_attempts"] += 1
        raise AssertionError("Regression tests may not perform live HTTP requests")

    suite_results = SuiteResults()
    # Test exactly the staged/tracked release scope, excluding other chats'
    # unfinished root research tests. Newly staged tests are included.
    names = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "--", "test_*.py"],
        text=True).splitlines()
    test_files = sorted(ROOT / name for name in names if "/" not in name)
    if not test_files:
        raise AssertionError("Expected root regression test files")
    status = 99
    started = time.monotonic()
    sqlite3.connect = sqlite3.dbapi2.connect = guarded_connect
    requests.sessions.Session.request = no_live_http
    try:
        with Path(__file__).with_name("test_verification.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                status = pytest.main(["-q", *[str(path) for path in test_files]],
                                     plugins=[suite_results])
    finally:
        sqlite3.connect = original_connect
        sqlite3.dbapi2.connect = original_dbapi2_connect
        requests.sessions.Session.request = original_request
        report = {
            "pytest_exit_code": int(status),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "root_test_files": len(test_files),
            "collected_tests": suite_results.collected,
            "test_counts": {name: suite_results.counts[name] for name in
                            ("passed", "failed", "skipped", "xfailed", "xpassed",
                             "errors", "collection_errors")},
            **counters,
            "isolated_database_count": len(isolated_paths),
            "guard_includes_sqlite_file_uri": True,
            "guard_blocks_requests_session_request": True,
        }
        Path(__file__).with_name("test_verification.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    return status


if __name__ == "__main__":
    sys.exit(main())
