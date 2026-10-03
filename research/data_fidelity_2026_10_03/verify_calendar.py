"""Read public seed only; retain a small official price-calendar cross-check.

Use --refresh-official to retrieve three TWSE monthly responses. Subsequent
runs use the archived responses. No formal market/journal database is opened.
"""
import argparse
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ENDPOINT = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
QUERIES = (("006208", "20150101"), ("006208", "20160101"), ("2330", "20160101"))


def roc_date(value):
    year, month, day = map(int, value.split("/"))
    return f"{year+1911:04d}-{month:02d}-{day:02d}"


def run(refresh=False):
    OUT.mkdir(parents=True, exist_ok=True)
    official = []
    for symbol, month in QUERIES:
        destination = OUT / f"twse_{symbol}_{month}.json"
        if refresh:
            response = requests.get(ENDPOINT, params={"response": "json", "stockNo": symbol,
                                                      "date": month}, timeout=15)
            response.raise_for_status()
            response.encoding = "utf-8"
            payload = response.json()
            if payload.get("stat") != "OK":
                raise RuntimeError(f"TWSE response unavailable: {symbol}/{month}")
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload = json.loads(destination.read_text(encoding="utf-8"))
        official.append((symbol, month, payload, destination))
    seed = ROOT / "data_seed/market.db.gz"
    with tempfile.TemporaryDirectory(prefix="quant_calendar_proof_") as directory:
        database = Path(directory) / "market.db"
        with gzip.open(seed, "rb") as source:
            database.write_bytes(source.read())
        assert database.resolve() != (ROOT / "data/market.db").resolve()
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            rows = pd.read_sql_query("SELECT symbol,date FROM ohlcv ORDER BY symbol,date", connection)
    rows["date"] = pd.to_datetime(rows["date"])
    benchmark = pd.DatetimeIndex(rows.loc[rows.symbol == "006208", "date"])
    stock = pd.DatetimeIndex(rows.loc[rows.symbol == "2330", "date"])
    extra = stock.difference(benchmark)
    checks = []
    for symbol, month, payload, destination in official:
        known = set(rows.loc[rows.symbol == symbol, "date"].dt.strftime("%Y-%m-%d"))
        missing = [r for r in payload["data"] if roc_date(r[0]) not in known]
        zero = [r for r in missing if str(r[1]).replace(",", "") == "0" and r[6] == "--"]
        checks.append({"symbol": symbol, "month": month, "official_sessions": len(payload["data"]),
                       "rows_without_cached_quote": len(missing), "no_trade_no_close_rows": len(zero),
                       "missing_dates": [roc_date(r[0]) for r in missing],
                       "missing_traded_dates": [roc_date(r[0]) for r in missing if r not in zero],
                       "response_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    report = {"seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
              "benchmark_rows": len(benchmark), "stock_2330_rows": len(stock),
              "stock_dates_without_benchmark_quote": len(extra),
              "stock_dates_without_benchmark_by_year": {str(int(y)): int(n) for y, n in
                 pd.Series(extra.year).value_counts().sort_index().items()},
              "latest_stock_date_without_benchmark_quote": str(extra[-1].date()),
              "official_source": ENDPOINT, "official_month_checks": checks,
              "formal_database_connections": 0,
              "conclusion": "Sample missing ETF dates are official no-trade/no-close sessions; an ETF quote calendar is not a complete exchange-session calendar.",
              "limitations": ["Only three monthly responses checked; not a full historical-calendar audit.",
                              "Some historical Saturdays were trading sessions; weekend dates alone do not establish bad data.",
                              "App buffered model uses ETF quote dates; legacy production uses stock-union dates.",
                              "The fair-comparison baseline must explicitly use the same ETF quote calendar.",
                              "Corporate-action, price-adjustment, universe-membership and source publication-time issues remain unverified."]}
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 公開行情日曆核對", "", "僅讀隔離公開 seed，沒有存取正式 DB／投資紀錄。", "",
             f"006208 有 {len(benchmark)} 個報價日，2330 有 {len(stock)} 日；股票有報價而 ETF 未收錄報價共 {len(extra)} 日，最後一日 {extra[-1].date()}。",
             "這些差異都在 2015～2018 年。抽查不是全歷史認證。", "",
             "| 官方抽查 | 官方日期數 | 快取未有報價日 | 官方無成交且無收盤 | 有成交但快取缺漏 |",
             "|---|---:|---:|---:|---:|"]
    for check in checks:
        lines.append(f"| {check['symbol']} {check['month'][:6]} | {check['official_sessions']} | {check['rows_without_cached_quote']} | {check['no_trade_no_close_rows']} | {len(check['missing_traded_dates'])} |")
    lines += ["", "抽查證明部分早期 ETF 缺報價日確實沒有成交，不能直接稱為資料錯誤，也不能當休市日。",
              "新模型採 ETF 報價日曆、舊模型採股票日期聯集；本次公平比較將原 60 日規則明確在同一 ETF 日曆重播。",
              "這個研究基準不宣稱等於舊 app 的全部歷史結果，app 模型日曆未修改。",
              "週六亦有歷史交易日，不能簡單刪除所有週末。未來成交日期仍需實際資料或官方交易日曆核對。", "",
              "原始官方回應與 SHA-256 保存在同目錄。", "",
              "重播：`python research/data_fidelity_2026_10_03/verify_calendar.py`。",
              "更新抽查來源：加 `--refresh-official`。", "",
              f"[證交所 006208 2016 年 1 月公開日成交資料]({ENDPOINT}?response=html&date=20160101&stockNo=006208)。"]
    (OUT / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-official", action="store_true")
    run(parser.parse_args().refresh_official)
