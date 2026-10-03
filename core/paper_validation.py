"""Prospective signal archive. It records decisions, never orders or returns.

Run this CLI separately from the app. Source SQLite is read-only and only the
three public market tables are selected; genuine app calculation runs inside a
disposable subprocess/cache with all HTTP requests forbidden. Wall-clock UTC
timestamps cannot be supplied by the caller. Historical execution outputs are
deliberately discarded: a new experiment starts all three arms in cash.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, closing
from datetime import datetime, timezone, timedelta
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
TABLES = {
    "ohlcv": ("symbol", "date", "open", "high", "low", "close", "volume"),
    "chip_weekly": ("symbol", "date", "big_shares", "total_shares"),
    "stock_info": ("symbol", "name"),
}
SOURCE_FILES = ("config.py", "core/rotation.py", "core/strategy_models.py",
                "core/execution.py", "core/chip_data.py", "core/market_regime.py",
                "core/data_pipeline.py", "core/paper_validation.py",
                "core/tpex_prices.py", "data_seed/universe_150.json")
CONFIG_KEYS = ("ROTATION_MOM_DAYS", "ROTATION_SKIP_DAYS", "ROTATION_TOP_K",
               "ROTATION_EXECUTION_LAG", "ROTATION_REBAL_DAYS", "ROTATION_ABS_MOM",
               "ROTATION_ABS_THRESH", "ROTATION_DEFENSIVE", "ROTATION_DEFENSIVE_POOL_MULT",
               "ROTATION_FASTSELL_GATE", "ROTATION_FASTSELL_Z", "ROTATION_FASTSELL_WIN",
               "ROTATION_SOX_GATE", "ROTATION_SOX_MA", "ROTATION_SOX_LAG",
               "ROTATION_SOX_SYMBOL", "ROTATION_USE_UNIVERSE", "ROTATION_MIN_OBS",
               "COST_PER_TURNOVER", "BENCHMARK_SYMBOL", "UNIVERSE", "WATCHLIST",
               "CA_JUMP_THRESHOLD", "DATA_SOURCE", "FALLBACK_TO_SYNTHETIC",
               "UNIVERSE_KIND", "UNIVERSE_ASOF", "UNIVERSE_MARKETS",
               "UNIVERSE_LISTING_DATES", "UNIVERSE_PROVISIONAL")


class ProtocolChanged(ValueError):
    """A changed rule or code needs a new experiment directory."""


class ArchiveCorrupt(ValueError):
    """An archived protocol, event, or public snapshot was modified."""


def _utc_now():
    return datetime.now(timezone.utc)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _database_path(database):
    text = str(database)
    candidate = unquote(urlsplit(text).path) if text.startswith("file:") else text
    if len(candidate) > 2 and candidate[0] == "/" and candidate[2] == ":":
        candidate = candidate[1:]  # Windows file:///D:/... URI
    return Path(candidate).resolve()


def _exclusive(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".archive.lock"
    try:
        _exclusive(path, _json({"pid": os.getpid(), "created_at": _utc_now().isoformat()}))
    except FileExistsError as error:
        raise RuntimeError("Archive is locked; inspect the owner before recovering a stale lock") from error
    try:
        yield
    finally:
        path.unlink()


def frozen_identity():
    """Freeze relevant public configuration and the complete calculation sources."""
    import config
    import numpy as np
    import pandas as pd
    from core.strategy_models import ACTIVE_MODEL_ID
    values = {key: getattr(config, key) for key in CONFIG_KEYS}
    fixed = (values["ROTATION_TOP_K"], values["ROTATION_REBAL_DAYS"],
             values["ROTATION_EXECUTION_LAG"], values["COST_PER_TURNOVER"])
    if fixed != (8, 20, 2, .003) or ACTIVE_MODEL_ID != "buffered_momentum":
        raise ProtocolChanged("This experiment requires frozen buffered_momentum/8/20/2/.003")
    return {"schema": SCHEMA, "active_model": ACTIVE_MODEL_ID, "parameters": values,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in SOURCE_FILES},
            "arms": [ACTIVE_MODEL_ID, "production_common_calendar", "006208_buy_hold"],
            "baseline": "original 60-day ranking/gates, recomputed on active model's 006208 calendar; not legacy app return path",
            "runtime": {"python": platform.python_version(), "python_implementation": platform.python_implementation(),
                        "numpy": np.__version__, "pandas": pd.__version__, "sqlite": sqlite3.sqlite_version},
            "evaluation_protocol": {
                "operational_checkpoint": {"forward_trading_days": 126, "minimum_rebalances": 6},
                "primary_comparison": {"verified_forward_trading_days": 252,
                                       "arms": [ACTIVE_MODEL_ID, "production_common_calendar"],
                                       "method": "paired 20-trading-day block bootstrap mean net-return difference, 95% CI"},
                "secondary_comparison": "006208 buy-and-hold from the same initial cash and future executable quotes",
                "performance_conclusions_require": "verified future fills, raw executable quotes, costs and corporate-action handling",
                "unverified_fills_allow_performance_claims": False},
            "parallel_universe_comparison": {
                "reference_archive": "data/paper_validation/forward_2026_10_03_v1",
                "reference_protocol_hash": "5d2780bb00594ed1cfc65f79d6cd913725bf48d044958fe98041786d6113d66a",
                "arms": "150 buffered_momentum versus frozen old50 buffered_momentum",
                "verified_common_forward_trading_days": 252,
                "method": "paired 20-trading-day block bootstrap mean net-return difference, 95% CI",
                "start": "both start in cash; initial signals as of 2026-10-02; first future executable quotes only",
                "gaps": "missing timely seals or unverified fills block performance conclusions; no retrospective signal reconstruction",
            } if len(values["UNIVERSE"]) == 150 else None,
            "starting_state": {"cash": 1., "positions": {}, "currency": "TWD",
                               "notional_only": True},
            "recording": "latest available data date only; no retrospective signals or fills",
            "execution": "future t+2 opening quote validation required; no fills implemented",
            "corporate_actions": "cached adjusted prices are not verified broker execution prices"}


def read_public_snapshot(market_db, sox_csv):
    """SELECT public tables only; refuse the actual app DB even for read-only QA."""
    database = Path(market_db).resolve()
    if database == (ROOT / "data/market.db").resolve():
        raise ValueError("Use an explicit independent market-only copy, not the formal app database")
    if not database.is_file() or not Path(sox_csv).is_file():
        raise ValueError("Public market SQLite and SOX CSV are both required")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        payload = {"schema": SCHEMA, "tables": {}}
        for table, columns in TABLES.items():
            selected = ",".join('"' + column + '"' for column in columns)
            order = "symbol,date" if "date" in columns else "symbol"
            payload["tables"][table] = {"columns": list(columns), "rows": [list(row) for row in
                conn.execute(f'SELECT {selected} FROM "{table}" ORDER BY {order}')]}
    import pandas as pd
    frame = pd.read_csv(sox_csv, index_col=0)
    if frame.empty or len(frame.columns) != 1:
        raise ValueError("SOX snapshot requires one close column")
    series = pd.to_numeric(frame.iloc[:, 0], errors="raise")
    series.index = pd.to_datetime(frame.index)
    if series.index.has_duplicates or not series.gt(0).all():
        raise ValueError("Invalid SOX prices or duplicate dates")
    payload["sox"] = [[str(date.date()), float(value)] for date, value in series.sort_index().items()]
    # Strict JSON rejects NaN/infinity rather than hiding missing prices.
    _json(payload)
    return payload


def _worker(snapshot):
    """Genuine app model entry in a disposable market cache, never a journal."""
    import pandas as pd
    with tempfile.TemporaryDirectory(prefix="quant_paper_model_") as temporary:
        os.environ["APP_DATA_DIR"] = temporary
        os.environ["JOURNAL_DATABASE_URL"] = ""
        import config
        if Path(config.DB_PATH).resolve() != Path(temporary, "market.db").resolve():
            raise AssertionError("Worker must import config only after setting its isolated cache")
        original_connect = sqlite3.connect
        counters = {"formal_database_attempts": 0, "external_network_attempts": 0}
        def guarded_connect(database, *args, **kwargs):
            candidate = _database_path(database)
            if candidate == (ROOT / "data/market.db").resolve():
                counters["formal_database_attempts"] += 1
                raise AssertionError("Formal app database access is forbidden")
            if str(database) != ":memory:" and candidate != Path(config.DB_PATH).resolve():
                raise AssertionError("Worker can connect only to its isolated public market cache")
            return original_connect(database, *args, **kwargs)
        sqlite3.connect = sqlite3.dbapi2.connect = guarded_connect
        with closing(sqlite3.connect(config.DB_PATH)) as conn:
            for table, item in snapshot["tables"].items():
                if table not in TABLES or tuple(item["columns"]) != TABLES[table]:
                    raise ValueError("Snapshot table is not public/expected")
                columns = item["columns"]
                definitions = ",".join('"' + col + '" ' +
                    ("TEXT" if col in ("symbol", "date", "name") else "REAL") for col in columns)
                conn.execute(f'CREATE TABLE "{table}" ({definitions})')
                conn.executemany(f'INSERT INTO "{table}" VALUES (' + ",".join("?" for _ in columns) + ")", item["rows"])
            conn.commit()
        pd.Series({pd.Timestamp(d): v for d, v in snapshot["sox"]}, name="close").to_csv(Path(temporary, "sox.csv"))
        import requests
        def forbidden(*args, **kwargs):
            counters["external_network_attempts"] += 1
            raise AssertionError("Network, data hydration, and journal access are forbidden")
        requests.sessions.Session.request = forbidden
        from core import data_pipeline as dp
        dp._INITIALIZED_PATHS.add(config.DB_PATH)
        dp.ensure_data = lambda symbol: None
        dp._finmind_get = forbidden
        from core import market_regime, rotation
        market_regime.refresh_sox = forbidden
        identity = frozen_identity()
        names = dict(snapshot["tables"]["stock_info"]["rows"])
        rotation.get_stock_name = lambda symbol: names.get(symbol, "")
        benchmark = dp.load_ohlcv(config.BENCHMARK_SYMBOL)
        calendar = pd.DatetimeIndex(benchmark["close"].dropna().index).sort_values()
        original_panel = rotation._load_panel
        def common_calendar_panel(symbols, mom_days, min_obs=None, skip_days=None):
            ret, _, _, model_names = original_panel(symbols, mom_days, min_obs, skip_days)
            closes = pd.DataFrame({s: dp.load_ohlcv(s)["close"] for s in ret.columns}).reindex(calendar)
            from core.strategy_models import build_rank_scores
            return (closes.pct_change(fill_method=None),
                    build_rank_scores(closes, "production", mom_days=mom_days,
                                      skip_days=config.ROTATION_SKIP_DAYS),
                    closes.pct_change(mom_days, fill_method=None), model_names)
        outputs = {}
        for mode in ("buffered_momentum", "production"):
            rotation._load_panel = common_calendar_panel if mode == "production" else original_panel
            result = rotation.run_rotation(score_mode=mode)
            weights = result["target_weights"]
            last = weights.iloc[-1]
            label = "production_common_calendar" if mode == "production" else mode
            outputs[label] = {"target": {str(s): float(w) for s, w in last.items() if w > 0},
                             "data_date": result["last_date"], "selection_date": result["selection_date"],
                             "calendar_anchor": str(weights.index[0].date()),
                             "calendar": [str(d.date()) for d in weights.index],
                             "data_quality": result["data_quality"], "sox": result["sox"]}
        rotation._load_panel = original_panel
        if identity != frozen_identity():
            raise ProtocolChanged("Source/config changed while isolated calculation was running")
        calendars = [model["calendar"] for model in outputs.values()]
        if calendars[0] != calendars[1]:
            raise AssertionError("Comparison arms require exactly the same 006208 calendar")
        sqlite3.connect = sqlite3.dbapi2.connect = original_connect
        return {"models": outputs, "calendar": [str(d.date()) for d in calendar],
                "data_date": str(calendar[-1].date()), "worker_identity": identity,
                "isolation": counters}


def calculate_app_targets(snapshot):
    """Subprocess keeps isolation intact even when the caller imported the app."""
    environment = dict(os.environ, JOURNAL_DATABASE_URL="")
    completed = subprocess.run([sys.executable, "-m", "core.paper_validation", "--_worker"],
        input=_json(snapshot), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=ROOT, env=environment, check=False, timeout=180)
    if completed.returncode:
        raise RuntimeError("Isolated model calculation failed: " + completed.stderr.decode("utf-8", errors="replace")[-2000:])
    return json.loads(completed.stdout)


def _history_revision(previous, current):
    """Detect old row additions, deletions and changed adjusted prices/chips/SOX."""
    differences = {}
    for table in ("ohlcv", "chip_weekly"):
        before = {tuple(row[:2]): row[2:] for row in previous["tables"][table]["rows"]}
        after = {tuple(row[:2]): row[2:] for row in current["tables"][table]["rows"]}
        cutoffs = {}
        for symbol, date in before:
            cutoffs[symbol] = max(date, cutoffs.get(symbol, date))
        keys = set(before) | {key for key in after if key[0] in cutoffs and key[1] <= cutoffs[key[0]]}
        count = sum(before.get(key) != after.get(key) for key in keys)
        if count:
            differences[table] = count
    before, after = dict(previous["sox"]), dict(current["sox"])
    if before:
        cutoff = max(before)
        count = sum(before.get(key) != after.get(key) for key in set(before) | {key for key in after if key <= cutoff})
        if count:
            differences["sox"] = count
    return differences


def _target_issues(snapshot, calculation, identity):
    """Fail closed on corrupt weights instead of silently changing a decision."""
    reasons = []
    expected = set(identity["arms"]) - {"006208_buy_hold"}
    if set(calculation["models"]) != expected:
        reasons.append("unexpected_model_arm")
    public_symbols = {row[0] for row in snapshot["tables"]["ohlcv"]["rows"]}
    parameters = identity.get("parameters", {})
    pool_key = "UNIVERSE" if parameters.get("ROTATION_USE_UNIVERSE", True) else "WATCHLIST"
    known = public_symbols.intersection(parameters.get(pool_key, public_symbols))
    for mode, model in calculation["models"].items():
        target = model.get("target")
        if not isinstance(target, dict):
            reasons.append(mode + ":invalid_target_mapping")
            continue
        if len(target) > 8:
            reasons.append(mode + ":more_than_8_target_stocks")
        if set(target) - known:
            reasons.append(mode + ":unknown_target_symbol")
        weights = []
        for value in target.values():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                weight = float("nan")
            if isinstance(value, bool) or not math.isfinite(weight) or weight <= 0:
                reasons.append(mode + ":nonfinite_or_nonpositive_target_weight")
            else:
                weights.append(weight)
        if sum(weights) > 1 + 1e-12:
            reasons.append(mode + ":target_weight_exceeds_cash")
    return reasons


def verify_archive(directory):
    """Verify all immutable records and their public snapshot content hashes."""
    root = Path(directory)
    protocol = json.loads((root / "protocol.json").read_bytes())
    expected = protocol.pop("protocol_hash")
    if _hash(protocol) != expected:
        raise ArchiveCorrupt("Protocol hash mismatch")
    protocol["protocol_hash"] = expected
    records, previous = [], expected
    for number, path in enumerate(sorted((root / "events").glob("*.json")), 1):
        event = json.loads(path.read_bytes())
        value = dict(event)
        digest = value.pop("event_hash")
        if value["sequence"] != number or value["previous_hash"] != previous or _hash(value) != digest:
            raise ArchiveCorrupt("Event hash chain mismatch")
        snapshot = json.loads(gzip.decompress((root / "snapshots" / (event["snapshot_hash"] + ".json.gz")).read_bytes()))
        if _hash(snapshot) != event["snapshot_hash"]:
            raise ArchiveCorrupt("Public snapshot hash mismatch")
        if event["recorded_at"] < protocol["frozen_at"] or (records and event["recorded_at"] < records[-1]["recorded_at"]):
            raise ArchiveCorrupt("Non-monotonic archive clock")
        previous = digest
        records.append(event)
    return {"protocol": protocol, "events": records, "head_hash": previous}


def append_latest(directory, snapshot, calculation, identity=None):
    """Seal only the latest signal; pending/blocked are never fake executions.

    ``identity`` is a test seam; production ``capture`` always supplies the real
    current source/config identity, verified against the isolated calculation.
    """
    root = Path(directory)
    identity = frozen_identity() if identity is None else identity
    now = _utc_now()
    stamp = now.isoformat()
    digest = _hash(snapshot)
    date = calculation["data_date"]
    if calculation.get("worker_identity", identity) != identity:
        raise ProtocolChanged("Worker and caller identities differ")
    with _lock(root):
        path = root / "protocol.json"
        if not path.exists():
            protocol = {"identity": identity, "frozen_at": stamp,
                        "initial_data_date": date, "initial_snapshot_hash": digest,
                        "initial_calendars": {mode: {"anchor": model["calendar_anchor"],
                            "sha256": _hash(model["calendar"]), "observations": len(model["calendar"])}
                            for mode, model in calculation["models"].items()},
                        "performance_status": "not_started; signals only; no realized future returns"}
            protocol["protocol_hash"] = _hash(protocol)
            _exclusive(path, _json(protocol))
        state = verify_archive(root)
        protocol, records = state["protocol"], state["events"]
        if protocol["identity"] != identity:
            raise ProtocolChanged("Frozen source/config changed: create a new experiment; do not overwrite this one")
        if stamp < protocol["frozen_at"] or (records and stamp < records[-1]["recorded_at"]):
            raise ValueError("Clock moved backward; recording cannot be backdated")
        if records and date < max(event["data_date"] for event in records):
            raise ValueError("Historical signal backfill is forbidden")
        previous_signal = next((e for e in reversed(records) if e["kind"] == "signal"), None)
        previous = records[-1] if records else None
        revisions = {}
        if previous:
            earlier = json.loads(gzip.decompress((root / "snapshots" / (previous["snapshot_hash"] + ".json.gz")).read_bytes()))
            revisions = _history_revision(earlier, snapshot)
        reasons = _target_issues(snapshot, calculation, identity)
        local = now.astimezone(timezone(timedelta(hours=8)))
        if date > local.date().isoformat() or (date == local.date().isoformat() and local.hour < 18):
            reasons.append("future_or_incomplete_data_date")
        for mode, model in calculation["models"].items():
            if model["data_date"] != date:
                reasons.append(mode + ":calendar_not_current")
            if model["data_quality"].get("stale", True):
                reasons.append(mode + ":stale_or_missing_source")
            if not model.get("sox", {}).get("ok", False):
                reasons.append(mode + ":SOX_gate_unavailable")
        if revisions:
            reasons.append("historical_source_revision_requires_review_and_new_experiment")
        if records and any("historical_source_revision_requires_review_and_new_experiment" in e["reasons"] for e in records):
            reasons.append("experiment_already_blocked_by_history_revision")
        sealed = next((e for e in records if e["data_date"] == date and e["kind"] == "signal"), None)
        if sealed and not revisions:
            return {"status": "already_sealed", "event": sealed,
                    "current_block_reasons": list(dict.fromkeys(reasons)),
                    "current_snapshot_changed": digest != sealed["snapshot_hash"]}
        kind = "blocked" if reasons else "signal"
        if previous and previous["data_date"] == date and previous["snapshot_hash"] == digest and previous["reasons"] == reasons:
            return {"status": "unchanged_blocked", "event": previous}
        gaps = []
        if previous_signal:
            gaps = [day for day in calculation["calendar"] if previous_signal["data_date"] < day < date]
        snapshot_path = root / "snapshots" / (digest + ".json.gz")
        if not snapshot_path.exists():
            _exclusive(snapshot_path, gzip.compress(_json(snapshot), mtime=0))
        targets = {}
        if kind == "signal":
            targets = {mode: model["target"] for mode, model in calculation["models"].items()}
            targets["006208_buy_hold"] = {"006208": 1.}
        event = {"sequence": len(records) + 1, "kind": kind, "recorded_at": stamp,
                 "data_date": date, "snapshot_hash": digest, "previous_hash": state["head_hash"],
                 "targets": targets, "reasons": list(dict.fromkeys(reasons)), "history_revisions": revisions,
                 "unrecorded_dates": gaps, "backfilled_signals": [],
                 "data_quality": {mode: model["data_quality"] for mode, model in calculation["models"].items()},
                 "isolation": calculation.get("isolation"),
                 "calendar_details": {mode: {"anchor": model["calendar_anchor"], "observations": len(model["calendar"])}
                                      for mode, model in calculation["models"].items()},
                 "execution_status": "blocked" if reasons else "pending_future_open_quotes",
                 "execution_date": None, "execution_lag_trading_days": 2,
                 "execution_rule": "t+2 relative to this data_date, and always after recorded_at; past opens forbidden",
                 "fills": [], "performance": None,
                 "starting_state": protocol["identity"]["starting_state"] if not previous_signal else None,
                 "historical_model_state_is_warmup_only": True,
                 "missing_days_are_never_retrospectively_executed": True}
        event["event_hash"] = _hash(event)
        _exclusive(root / "events" / (f"{len(records) + 1:06d}.json"), _json(event))
        return {"status": "recorded_" + kind, "event": event}


def capture(directory, market_db=None, sox_csv=None):
    """Default public seed or explicit independent read-only market cache."""
    with tempfile.TemporaryDirectory(prefix="quant_paper_seed_") as temporary:
        if market_db is None:
            market_db = Path(temporary, "market.db")
            market_db.write_bytes(gzip.decompress((ROOT / "data_seed/market.db.gz").read_bytes()))
            sox_csv = ROOT / "data_seed/sox.csv"
        elif sox_csv is None:
            raise ValueError("Explicit market cache requires --sox-csv")
        snapshot = read_public_snapshot(market_db, sox_csv)
        calculation = calculate_app_targets(snapshot)
        return append_latest(directory, snapshot, calculation, identity=frozen_identity())


def replay_sealed(directory, sequence=None):
    """Recompute an immutable archived target, without recording a new signal.

    Freshness is evaluated at today's clock by the app and is not proof that
    archived data are current. Replay compares target decisions only, never
    pretends to retrospectively produce fills or prospective return evidence.
    """
    root = Path(directory)
    state = verify_archive(root)
    if state["protocol"]["identity"] != frozen_identity():
        raise ProtocolChanged("Replay requires the exact archived calculation sources/config")
    eligible = [event for event in state["events"] if event["kind"] == "signal"
                and (sequence is None or event["sequence"] == sequence)]
    if not eligible:
        raise ValueError("No sealed signal matches the requested sequence")
    event = eligible[-1]
    snapshot = json.loads(gzip.decompress((root / "snapshots" / (event["snapshot_hash"] + ".json.gz")).read_bytes()))
    calculation = calculate_app_targets(snapshot)
    if calculation["worker_identity"] != state["protocol"]["identity"]:
        raise ProtocolChanged("Replay worker does not match the archived sources/config")
    targets = {mode: model["target"] for mode, model in calculation["models"].items()}
    targets["006208_buy_hold"] = {"006208": 1.}
    if targets != event["targets"]:
        raise ArchiveCorrupt("Archived target does not reproduce from its sealed public snapshot")
    return {"status": "replay_targets_match", "sequence": event["sequence"],
            "recorded_at": event["recorded_at"], "data_date": event["data_date"],
            "snapshot_hash": event["snapshot_hash"], "targets": targets,
            "isolation": calculation.get("isolation"), "fills": [], "performance": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--market-db", type=Path)
    parser.add_argument("--sox-csv", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        sys.stdout.buffer.write(_json(_worker(json.loads(sys.stdin.buffer.read()))))
        return
    if args.output is None:
        parser.error("--output independent experiment directory is required")
    if args.replay:
        result = replay_sealed(args.output, args.sequence)
    elif args.verify:
        state = verify_archive(args.output)
        result = {"status": "verified", "events": len(state["events"]), "head_hash": state["head_hash"]}
    else:
        result = capture(args.output, args.market_db, args.sox_csv)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
