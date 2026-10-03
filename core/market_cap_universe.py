"""Estimated current TWSE/TPEx common-share market-cap snapshot.

Independent of config, market.db and journals. This freezes current membership;
it does not provide historical point-in-time constituent or delisting data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import requests


class UniverseSnapshotError(ValueError):
    """Reject incomplete, malformed or inconsistent official sources."""


def normalize_date(value):
    raw = str(value).strip().replace("/", "").replace("-", "")
    if not raw.isdigit() or len(raw) not in (7, 8):
        raise UniverseSnapshotError(f"Invalid date: {value!r}")
    year = int(raw[:-4]) + (1911 if len(raw) == 7 else 0)
    try:
        return date(year, int(raw[-4:-2]), int(raw[-2:])).isoformat()
    except ValueError as exc:
        raise UniverseSnapshotError(f"Invalid date: {value!r}") from exc


def _number(value, label):
    try:
        number = Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise UniverseSnapshotError(f"Invalid {label}: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise UniverseSnapshotError(f"Invalid {label}: {value!r}")
    return number


class _Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = []


def parse_common_shares(html, market, asof):
    """Use official ordinary-share CFI ES*, not a code-length heuristic alone."""
    expected = {"twse": "上市", "tpex": "上櫃"}.get(market)
    if expected is None:
        raise UniverseSnapshotError(f"Unknown market: {market}")
    match = re.search(r"最近更新日期\s*:\s*(\d{4}/\d{2}/\d{2})", html)
    if not match:
        raise UniverseSnapshotError(f"Missing {market} ISIN update date")
    updated = normalize_date(match.group(1))
    if updated < asof:
        raise UniverseSnapshotError(f"{market} classification predates valuation")
    parser = _Rows()
    parser.feed(html)
    records = {}
    for row in parser.rows:
        if len(row) != 7:
            continue
        parts = row[0].split(maxsplit=1)
        if len(parts) != 2:
            continue
        code, name = parts
        if not re.fullmatch(r"[1-9]\d{3}", code) or not re.fullmatch(r"ES[A-Z]{4}", row[5]):
            continue
        if expected not in row[3]:
            raise UniverseSnapshotError(f"Wrong ISIN market for {code}")
        listed = normalize_date(row[2])
        if listed > asof:
            continue
        if code in records:
            raise UniverseSnapshotError(f"Duplicate common-share ISIN: {code}")
        records[code] = {"symbol": code, "name": name, "isin": row[1],
                         "cfi": row[5], "listed_date": listed, "market": market}
    if not records:
        raise UniverseSnapshotError(f"No classified {market} ordinary shares")
    return records, updated


def _indexed(rows, key, label):
    if not isinstance(rows, list) or not rows:
        raise UniverseSnapshotError(f"Empty/malformed {label}")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            raise UniverseSnapshotError(f"Missing {label} key {key}")
        code = str(row[key]).strip()
        if code in result:
            raise UniverseSnapshotError(f"Duplicate {label} symbol: {code}")
        result[code] = row
    return result


def build_snapshot(*, asof, twse_closes, twse_shares, tpex_values,
                   twse_isin_html, tpex_isin_html, top_n=150, twse_check=None,
                   scope="all_classified", allow_cap_differences=False):
    asof = normalize_date(asof)
    if scope not in ("all_classified", "priced_ordinary"):
        raise UniverseSnapshotError(f"Unknown ranking scope: {scope}")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise UniverseSnapshotError("top_n must be a positive integer")
    tc, td = parse_common_shares(twse_isin_html, "twse", asof)
    oc, od = parse_common_shares(tpex_isin_html, "tpex", asof)
    closes = _indexed(twse_closes, "Code", "TWSE closes")
    otc = _indexed(tpex_values, "SecuritiesCompanyCode", "TPEx values")
    if {normalize_date(r.get("Date", "")) for r in twse_closes} != {asof}:
        raise UniverseSnapshotError("TWSE close dates differ from requested date")
    if {normalize_date(r.get("Date", "")) for r in tpex_values} != {asof}:
        raise UniverseSnapshotError("TPEx dates differ from requested date")
    shares = _indexed(twse_shares, "公司代號", "TWSE issued common shares")
    if {normalize_date(r.get("出表日期", "")) for r in twse_shares} != {asof}:
        raise UniverseSnapshotError("TWSE issued-common-share date differs")
    # An explicitly priced pool may omit a security with no quote. It must never
    # omit a quoted common share simply because the share-count source failed.
    missing_shares = sorted((set(tc) & set(closes)) - set(shares))
    if missing_shares:
        raise UniverseSnapshotError(f"Missing TWSE issued shares for quoted ordinary stocks: {missing_shares}")
    missing_twse = sorted(set(tc) - set(closes))
    missing_tpex = sorted(set(oc) - set(otc))
    for market, classes, missing in (("twse", tc, missing_twse), ("tpex", oc, missing_tpex)):
        if len(missing) == len(classes):
            raise UniverseSnapshotError(f"No quoted ordinary shares for {market}; cannot degrade to one market")
    if (missing_twse or missing_tpex) and scope == "all_classified":
        raise UniverseSnapshotError(f"Missing common-share valuations TWSE={missing_twse}, TPEx={missing_tpex}")
    excluded = [{"symbol": c, "name": classes[c]["name"], "market": m,
                 "reason": "No same-day official closing-price/valuation row; trading status not confirmed",
                 "trading_status_confirmed": False}
                for m, codes, classes in (("twse", missing_twse, tc), ("tpex", missing_tpex, oc)) for c in codes]
    rows = []
    for market, classes in (("twse", tc), ("tpex", oc)):
        for code, classification in classes.items():
            if code in (missing_twse if market == "twse" else missing_tpex):
                continue
            if market == "twse":
                close = _number(closes[code].get("ClosingPrice"), f"{code} close")
                issued = _number(shares[code].get("已發行普通股數或TDR原股發行股數"), f"{code} issued shares")
                published = None
            else:
                source = otc[code]
                close = _number(source.get("ClosePrice"), f"{code} close")
                issued = _number(source.get("Capitals"), f"{code} issued shares")
                published = _number(source.get("MarketValue"), f"{code} published cap")
            if issued != issued.to_integral_value():
                raise UniverseSnapshotError(f"Fractional issued shares: {code}")
            cap = close * issued
            # The raw OpenAPI schema names MarketValue but omits its unit.
            # Check the million-NTD scale for every ordinary share, while ranking
            # by the exact shares-times-price product for comparable markets.
            if published is not None and abs(cap - published * 1_000_000) > 1_000_000:
                raise UniverseSnapshotError(f"TPEx cap unit/arithmetic mismatch: {code}")
            rows.append({**classification, "asof": asof, "close_ntd": float(close),
                         "issued_common_shares": int(issued), "market_cap_ntd": float(cap),
                         "published_market_value_million_ntd": float(published) if published else None})
    if len({r["symbol"] for r in rows}) != len(rows):
        raise UniverseSnapshotError("Duplicate security across markets")
    if len(rows) < top_n:
        raise UniverseSnapshotError(f"Only {len(rows)} common shares; need {top_n}")
    rows.sort(key=lambda r: (-r["market_cap_ntd"], r["symbol"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    checks = []
    if twse_check is not None:
        published_rows = twse_check.get("ranking", {}).get("market", [])
        if not published_rows:
            raise UniverseSnapshotError("Missing TWSE published market-cap check")
        by_code = {r["symbol"]: r for r in rows}
        for r in published_rows:
            code = str(r.get("NO", ""))
            if normalize_date(r.get("DATE", "")) != asof:
                raise UniverseSnapshotError(f"TWSE published ranking date/symbol mismatch: {code}")
            if code not in by_code:
                # Official home ranking also contains ETFs; only compare common shares.
                continue
            cap = _number(r.get("AMT"), "TWSE published cap") * 100_000_000
            computed = Decimal(str(by_code[code]["market_cap_ntd"]))
            difference = computed - cap
            # The home ranking uses listed shares; the MOPS basic table gives
            # issued ordinary shares. Small real issuance/listing differences are
            # recorded. Gross unit/stock-split mismatches fail closed.
            if abs(difference) / cap > Decimal("0.03") and not allow_cap_differences:
                raise UniverseSnapshotError(f"TWSE shares/close cap disagrees: {code}")
            checks.append({"symbol": code, "published_listed_cap_ntd": float(cap),
                           "issued_common_cap_ntd": float(computed),
                           "relative_difference": float(difference / cap),
                           "material_difference": abs(difference) / cap > Decimal("0.03")})
    top = rows[:top_n]
    boundary = None
    if len(rows) > top_n:
        last, next_row = top[-1], rows[top_n]
        gap = last["market_cap_ntd"] - next_row["market_cap_ntd"]
        boundary = {"last_selected": {k: last[k] for k in ("rank", "symbol", "name", "market_cap_ntd")},
                    "first_excluded": {k: next_row[k] for k in ("rank", "symbol", "name", "market_cap_ntd")},
                    "gap_ntd": gap, "gap_pct_of_last_selected": gap / last["market_cap_ntd"] * 100}
    return {"schema_version": 2, "asof": asof, "top_n": top_n,
            "ranking_basis": "ordinary-share closing price (NTD) x issued common shares (TWSE MOPS) / Capitals (TPEx)",
            "symbols": [r["symbol"] for r in top], "top150": top, "all_ranked": rows,
            "eligible_count": len(rows),
            "market_counts": {m: sum(r["market"] == m for r in rows) for m in ("twse", "tpex")},
            "selected_market_counts": {m: sum(r["market"] == m for r in top) for m in ("twse", "tpex")},
            "classification_counts": {"twse": len(tc), "tpex": len(oc)},
            "source_dates": {"twse_close": asof, "twse_issued_common_shares": asof, "tpex_daily_market_value": asof},
            "classification_dates": {"twse": td, "tpex": od}, "twse_published_cap_checks": checks,
            "source_incomplete": bool(excluded), "excluded_unpriced": excluded,
            "scope": scope, "quote_scoped_pool_complete": True, "estimated": True,
            "ranking_scope": "same-day priced TWSE/TPEx classified ordinary shares" if scope == "priced_ordinary" else "all classified TWSE/TPEx ordinary shares",
            "display_label": "當日有有效收盤報價的上市櫃普通股估算市值前" + str(top_n) if scope == "priced_ordinary" else "上市櫃普通股估算市值前" + str(top_n),
            "provisional": True, "boundary": boundary,
            "unit_checks": {"twse_price_ntd_times_common_shares": sum(r["market"] == "twse" for r in rows),
                            "tpex_market_value_million_ntd_arithmetic": sum(r["market"] == "tpex" for r in rows),
                            "tpex_tolerance_ntd": 1_000_000},
            "limitations": ["Current membership and current ISIN classification, not historical point-in-time constituents.",
                            "Includes KY and TWSE innovation-board common shares; CFI excludes ETF, preferred shares and TDR.",
                            "TPEx schema does not label MarketValue units; million-NTD scale checked for every ordinary share; ranking uses shares x close.",
                            "TWSE issued-common-share cap includes issued private shares and can differ from published listed-share cap. Home top20 differences are recorded; MI_QFIIS shares can lag stock splits and are not used for ranking."]}


def fetch_snapshot(asof, *, output_dir, top_n=150, timeout=40,
                   scope="all_classified", allow_cap_differences=False):
    """Archive exact official response bodies; neither market may fail silently."""
    asof = normalize_date(asof)
    folder = Path(output_dir)
    if any((folder / name).exists() for name in ("official_raw", "sources.json", "universe_150.json", "candidate_150.json")):
        raise UniverseSnapshotError("Research snapshot already exists; use a new output directory")
    compact = asof.replace("-", "")
    sources = {
        "twse_closes": ("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", "utf-8"),
        "twse_shares": ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "utf-8"),
        "twse_shareholding_check": (f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date={compact}&selectType=ALLBUT0999&response=json", "utf-8"),
        "tpex_values": ("https://www.tpex.org.tw/openapi/v1/tpex_daily_market_value", "utf-8"),
        "twse_isin": ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "cp950"),
        "tpex_isin": ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", "cp950"),
        "twse_cap_check": ("https://www.twse.com.tw/res/data/zh/home/values.json", "utf-8"),
    }
    def fetch(item):
        name, (url, encoding) = item
        last = None
        for _ in range(2):
            try:
                response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                raw = response.content
                value = raw.decode(encoding)
                if "isin" not in name:
                    value = json.loads(value)
                return name, raw, value, {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "encoding": encoding,
                    "last_modified": response.headers.get("Last-Modified")}
            except (requests.RequestException, UnicodeError, json.JSONDecodeError) as exc:
                last = exc
        raise UniverseSnapshotError(f"Official source failed: {name}: {last}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(fetch, sources.items()))
    raw_dir = folder / "official_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, raw, _, info in responses:
        path = raw_dir / f"{name}.gz"
        path.write_bytes(gzip.compress(raw, mtime=0))
        metadata[name] = {**info, "raw_file": f"official_raw/{path.name}"}
    (folder / "sources.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    values = {name: value for name, _, value, _ in responses}
    snapshot = build_snapshot(asof=asof, top_n=top_n,
        twse_closes=values["twse_closes"], twse_shares=values["twse_shares"],
        tpex_values=values["tpex_values"], twse_isin_html=values["twse_isin"],
        tpex_isin_html=values["tpex_isin"], twse_check=values["twse_cap_check"],
        scope=scope, allow_cap_differences=allow_cap_differences)
    snapshot["sources"] = metadata
    (folder / "universe_150.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=150)
    parser.add_argument("--scope", choices=("all_classified", "priced_ordinary"), default="all_classified")
    parser.add_argument("--allow-issued-listed-difference", action="store_true",
                        help="Record, rather than reject, issued-versus-listed share valuation differences")
    args = parser.parse_args()
    result = fetch_snapshot(args.asof, top_n=args.top_n, output_dir=args.output_dir,
                            scope=args.scope, allow_cap_differences=args.allow_issued_listed_difference)
    print(json.dumps({k: result[k] for k in ("asof", "eligible_count", "market_counts", "selected_market_counts", "symbols")}, ensure_ascii=True))
