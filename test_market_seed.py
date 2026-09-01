import gzip
import os
import shutil
import sqlite3
import tempfile

import config


def test_market_seed_contains_full_benchmark_history():
    seed = os.path.join(config.BASE_DIR, "data_seed", "market.db.gz")
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        with gzip.open(seed, "rb") as source, open(path, "wb") as target:
            shutil.copyfileobj(source, target)
        conn = sqlite3.connect(path)
        count, start, end = conn.execute(
            "SELECT count(*), min(date), max(date) FROM ohlcv WHERE symbol = ?",
            (config.BENCHMARK_SYMBOL,),
        ).fetchone()
        conn.close()
        assert count > 2000
        assert start <= "2015-01-10"
        assert end >= "2026-08-31"
    finally:
        os.remove(path)
