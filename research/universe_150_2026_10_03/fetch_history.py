"""Bounded, resumable PUBLIC data acquisition; never opens the app database.

Same price-cleaning convention as the existing 50-stock seed, so the universe
comparison changes membership only. This is NOT official total-return data.
Raw added-symbol responses are retained under the ignored research cache.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config
from core.data_pipeline import _adjust_corporate_actions
import pandas as pd
import requests

CACHE = ROOT / "data" / "universe_150_research"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(CACHE.resolve()):
        raise ValueError("History writes must stay in the isolated research cache")
    return path


def fetch(dataset, symbol, end, raw_dir):
    raw_path = check_path(raw_dir / f"{dataset}_{symbol}_{end}.json.gz")
    if raw_path.exists():
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        headers = {"Authorization": f"Bearer {config.FINMIND_TOKEN}"} if config.FINMIND_TOKEN else {}
        response = requests.get(config.FINMIND_URL, headers=headers, params={
            "dataset": dataset, "data_id": symbol,
            "start_date": "2015-01-01", "end_date": end,
        }, timeout=20)
        if response.status_code in (401, 402, 403, 429):
            raise PermissionError(f"Upstream denied/limited request (HTTP {response.status_code}); stop batch")
        payload = response.json()
        if response.status_code in (401, 402, 403, 429) or payload.get("status") in (401, 402, 403, 429):
            raise PermissionError(f"Upstream denied/limited request ({response.status_code}/{payload.get('status')}); stop batch")
        response.raise_for_status()
        if payload.get("status") != 200 or not payload.get("data"):
            raise ValueError(f"No valid {dataset} rows for {symbol}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        time.sleep(.1)
    return pd.DataFrame(payload["data"]), {"raw": str(raw_path.relative_to(ROOT)), "sha256": digest(raw_path)}


def price_frame(frame, symbol, end, listed_date=None):
    required = {"date", "stock_id", "open", "max", "min", "close", "Trading_Volume"}
    if not required <= set(frame):
        raise ValueError("Unexpected price schema")
    if set(frame.stock_id.astype(str)) != {symbol}:
        raise ValueError("Response symbol mismatch")
    result = frame.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})[
        ["date", "open", "high", "low", "close", "volume"]].copy()
    result["date"] = pd.to_datetime(result.date, errors="raise").dt.strftime("%Y-%m-%d")
    if (result.date > end).any() or (result.date < "2015-01-01").any() or result.date.duplicated().any():
        raise ValueError("Unexpected price dates")
    result = result.sort_values("date")
    if listed_date:
        # FinMind price history can include pre-listing emerging-board quotes.
        # Current-market ISIN start is a conservative floor: even a legitimate
        # earlier OTC period is omitted after a transfer, rather than guessing.
        result = result[result.date >= listed_date]
    for col in ("open", "high", "low", "close", "volume"):
        result[col] = pd.to_numeric(result[col], errors="raise")
    result = _adjust_corporate_actions(result)
    result.insert(0, "symbol", symbol)
    if result.empty:
        raise ValueError("No post-listing price rows")
    return result


def chip_frame(frame, symbol, end):
    required = {"date", "stock_id", "ForeignInvestmentShares", "NumberOfSharesIssued"}
    if not required <= set(frame) or set(frame.stock_id.astype(str)) != {symbol}:
        raise ValueError("Unexpected shareholding schema/symbol")
    result = pd.DataFrame({"symbol": symbol, "date": pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d"),
                           "big_shares": pd.to_numeric(frame.ForeignInvestmentShares, errors="raise"),
                           "total_shares": pd.to_numeric(frame.NumberOfSharesIssued, errors="raise")})
    if (result.date > end).any():
        raise ValueError("Future shareholding rows")
    result = result.drop_duplicates("date").sort_values("date")
    result = result[(result.total_shares > 0) & (result.big_shares >= 0) & (result.big_shares <= result.total_shares)]
    if result.empty:
        raise ValueError("No valid shareholding rows")
    return result


def run(ranking, budget=480):
    spec = json.loads(Path(ranking).read_text(encoding="utf-8"))
    symbols = spec["symbols"]
    if len(symbols) != 150 or len(set(symbols)) != 150:
        raise ValueError("Exactly 150 unique ranked symbols required")
    end = spec["asof"]
    CACHE.mkdir(parents=True, exist_ok=True)
    database = check_path(CACHE / "market.db")
    initial_seed_sha = None
    if not database.exists():
        initial_seed_sha = digest(ROOT / "data_seed" / "market.db.gz")
        with gzip.open(ROOT / "data_seed" / "market.db.gz", "rb") as source, database.open("wb") as target:
            shutil.copyfileobj(source, target)
    shutil.copyfile(ROOT / "data_seed" / "sox.csv", CACHE / "sox.csv")
    manifest_path = CACHE / "acquisition.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"symbols": {}}
    manifest.setdefault("base_seed_sha256", initial_seed_sha or "unknown: pre-existing research DB without seed provenance")
    manifest.update({"asof": end, "ranking_sha256": digest(ranking),
                     "price_convention": "existing app jump heuristic, NOT verified total return", "formal_database_opens": 0})
    started = time.monotonic()
    with closing(sqlite3.connect(database)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables <= {"ohlcv", "chip_weekly", "stock_info"}:
            raise ValueError("Research seed includes private/non-market tables")
        names = {row.get("symbol"): row.get("name", row.get("symbol")) for row in spec.get("top150", [])}
        listing = {row["symbol"]: row.get("listed_date") for row in spec.get("top150", [])}
        for index, symbol in enumerate(symbols, 1):
            if time.monotonic() - started > budget:
                manifest["stop_reason"] = "time_budget"
                break
            p = connection.execute("SELECT COUNT(*),MIN(date),MAX(date) FROM ohlcv WHERE symbol=?", (symbol,)).fetchone()
            c = connection.execute("SELECT COUNT(*),MIN(date),MAX(date) FROM chip_weekly WHERE symbol=?", (symbol,)).fetchone()
            if p[0] >= 252 and p[2] == end and c[0] and c[2] == end:
                manifest["symbols"].setdefault(symbol, {"status": "cached_seed", "prices": p, "chips": c})
                continue
            try:
                price, price_proof = fetch("TaiwanStockPrice", symbol, end, CACHE / "raw")
                chip, chip_proof = fetch("TaiwanStockShareholding", symbol, end, CACHE / "raw")
                bars = price_frame(price, symbol, end, listing.get(symbol))
                chips = chip_frame(chip, symbol, end)
                with connection:
                    connection.execute("DELETE FROM ohlcv WHERE symbol=?", (symbol,))
                    connection.execute("DELETE FROM chip_weekly WHERE symbol=?", (symbol,))
                    connection.executemany("INSERT INTO ohlcv VALUES(?,?,?,?,?,?,?)", bars.itertuples(index=False, name=None))
                    connection.executemany("INSERT INTO chip_weekly VALUES(?,?,?,?)", chips.itertuples(index=False, name=None))
                    connection.execute("INSERT OR REPLACE INTO stock_info VALUES (?,?)", (symbol, names.get(symbol, symbol)))
                manifest["symbols"][symbol] = {"status": "downloaded", "prices": [len(bars), bars.date.min(), bars.date.max()],
                                               "chips": [len(chips), chips.date.min(), chips.date.max()], "price_source": price_proof, "chip_source": chip_proof}
                print(f"{index}/150 {symbol}: {len(bars)} price / {len(chips)} chip", flush=True)
            except PermissionError as exc:
                manifest["stop_reason"] = str(exc)
                print(manifest["stop_reason"], flush=True)
                break
            except Exception as exc:
                manifest["symbols"][symbol] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                print(f"{index}/150 {symbol}: {type(exc).__name__}", flush=True)
            finally:
                manifest["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
        trimmed = {}
        # Reconcile already downloaded rows when acquisition is resumed; also
        # retain original raw responses as provenance for the explicit filter.
        with connection:
            for symbol, start in listing.items():
                if start:
                    removed = connection.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND date<?", (symbol, start)).fetchone()[0]
                    if removed:
                        connection.execute("DELETE FROM ohlcv WHERE symbol=? AND date<?", (symbol, start))
                        trimmed[symbol] = {"removed_price_rows": removed, "first_current_market_listing": start}
        manifest["listing_floor_filter"] = {
            "rule": "prices >= official current-market ISIN listing date; conservative exclusion of earlier OTC transfers too",
            "trimmed_this_run": trimmed,
        }
        manifest["coverage"] = {symbol: {
            "prices": connection.execute("SELECT COUNT(*),MIN(date),MAX(date) FROM ohlcv WHERE symbol=?", (symbol,)).fetchone(),
            "chips": connection.execute("SELECT COUNT(*),MIN(date),MAX(date) FROM chip_weekly WHERE symbol=?", (symbol,)).fetchone(),
        } for symbol in symbols}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"database": str(database), "elapsed": manifest["elapsed_seconds"], "covered": sum(bool(row["prices"][0]) and bool(row["chips"][0]) for row in manifest["coverage"].values()), "stop_reason": manifest.get("stop_reason")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True)
    parser.add_argument("--budget", type=float, default=480)
    args = parser.parse_args()
    run(args.universe, args.budget)
