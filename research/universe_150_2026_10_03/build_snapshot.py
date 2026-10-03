"""Rebuild the frozen estimated pool from hash-verified official archives.

Run from the repository root:
    python -m research.universe_150_2026_10_03.build_snapshot

No network, config import, database or trading journal access is performed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from core.market_cap_universe import UniverseSnapshotError, build_snapshot


ORIGINAL_50 = [
    "2330", "2317", "2454", "2308", "2303", "2412", "2882", "2881", "1301", "2603",
    "2891", "3711", "2002", "2886", "2884", "1303", "2327", "2357", "3008", "2382",
    "2395", "5871", "2880", "2892", "2885", "1216", "2207", "2379", "3045", "2912",
    "1101", "2887", "4938", "5880", "2883", "2890", "2345", "3037", "2301", "3034",
    "2105", "9910", "2474", "1402", "2409", "2354", "2360", "6505", "3443", "3017",
]


def rebuild(folder=None):
    folder = Path(folder or Path(__file__).parent).resolve()
    metadata = json.loads((folder / "sources.json").read_text(encoding="utf-8"))
    required = {"twse_closes", "twse_shares", "tpex_values", "twse_isin",
                "tpex_isin", "twse_cap_check", "twse_shareholding_check"}
    if set(metadata) != required:
        raise UniverseSnapshotError("Missing or unexpected official archive sources")
    values = {}
    for name, info in metadata.items():
        path = (folder / info["raw_file"]).resolve()
        if not path.is_relative_to(folder):
            raise UniverseSnapshotError(f"Archive path outside research folder: {name}")
        raw = gzip.decompress(path.read_bytes())
        if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise UniverseSnapshotError(f"Official archive integrity mismatch: {name}")
        decoded = raw.decode(info["encoding"])
        values[name] = decoded if "isin" in name else json.loads(decoded)
    snapshot = build_snapshot(asof="2026-10-02", top_n=150,
        twse_closes=values["twse_closes"], twse_shares=values["twse_shares"],
        tpex_values=values["tpex_values"], twse_isin_html=values["twse_isin"],
        tpex_isin_html=values["tpex_isin"], twse_check=values["twse_cap_check"],
        scope="priced_ordinary", allow_cap_differences=True)
    candidate = folder / "candidate_150.json"
    if candidate.exists():
        old_symbols = json.loads(candidate.read_text(encoding="utf-8"))["symbols"]
        if snapshot["symbols"] != old_symbols:
            raise UniverseSnapshotError("Final pool differs from the candidate used for isolated downloads")
    selected, original = set(snapshot["symbols"]), set(ORIGINAL_50)
    snapshot["original_50_comparison"] = {
        "reference": "config.UNIVERSE literal before the 150-stock expansion experiment",
        "symbols": ORIGINAL_50,
        "overlap_count": len(selected & original),
        "added_count": len(selected - original),
        "removed_symbols": sorted(original - selected),
        "added_symbols": [s for s in snapshot["symbols"] if s not in original],
    }
    snapshot["sources"] = metadata
    snapshot["rebuild"] = {"raw_integrity_verified": True,
        "command": "python -m research.universe_150_2026_10_03.build_snapshot",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parser_sha256": hashlib.sha256((folder.parents[1] / "core" / "market_cap_universe.py").read_bytes()).hexdigest()}
    output = folder / "universe_150.json"
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: snapshot[k] for k in ("asof", "display_label", "scope", "eligible_count",
        "market_counts", "selected_market_counts", "classification_counts", "classification_dates",
        "source_dates", "estimated", "provisional", "source_incomplete", "quote_scoped_pool_complete",
        "excluded_unpriced", "boundary", "unit_checks", "twse_published_cap_checks", "limitations")}
    summary["original_50_comparison"] = {k: snapshot["original_50_comparison"][k]
        for k in ("overlap_count", "added_count", "removed_symbols")}
    summary["field_units"] = {
        "twse": {"price": "STOCK_DAY_ALL.ClosingPrice: NTD per share",
                 "shares": "t187ap03_L.已發行普通股數或TDR原股發行股數: common shares, including private shares",
                 "cap": "ClosingPrice x issued common shares: estimated NTD cap"},
        "tpex": {"price": "tpex_daily_market_value.ClosePrice: NTD per share",
                 "shares": "Capitals: issued shares according to official schema",
                 "cap": "ClosePrice x Capitals: estimated NTD cap",
                 "published_market_value": "MarketValue: million NTD scale matches all 888 eligible rows within one million NTD",
                 "unit_evidence_limit": "Official OpenAPI schema calls this field 市值 but does not explicitly label its unit; ranking does not rely on this unlabelled value."},
        "twse_crosscheck": {"source": "TWSE home values.json ranking.market.AMT",
                            "unit": "hundred million NTD; displayed by official web-home.js",
                            "scope": "Published home top20 only, includes an ETF; 19 common-share checks, not a complete listed-share ranking"},
    }
    summary["documentation"] = {
        "twse_schema": "https://openapi.twse.com.tw/v1/swagger.json",
        "tpex_schema": "https://www.tpex.org.tw/openapi/swagger.json",
        "twse_home_display": "https://www.twse.com.tw/res/js/web-home.js",
        "official_classification": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
    }
    (folder / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


if __name__ == "__main__":
    result = rebuild()
    summary = {k: result[k] for k in ("asof", "eligible_count", "market_counts", "selected_market_counts",
        "classification_counts", "classification_dates", "boundary", "source_incomplete", "quote_scoped_pool_complete")}
    summary["original_50_comparison"] = {k: result["original_50_comparison"][k]
        for k in ("overlap_count", "added_count", "removed_symbols")}
    print(json.dumps(summary, ensure_ascii=True, indent=2))
