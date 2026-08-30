# -*- coding: utf-8 -*-
"""
twse_universe.py - 用 TWSE/TPEX 官方 OpenAPI 建「規則式動態選股池」
取代手列的 50 檔:抓全市場(上市+上櫃)當日成交金額,過濾成普通股,依「流動性
(成交金額)」由大到小取前 N 大 -> 規則決定誰進池,而非憑印象挑。
分工:OpenAPI 決定『誰進池』(全市場流動性排名);FinMind 給『歷史價』(動能/回測)。

★ 誠實但書:這是「當前流動性」快照選池,對歷史回測仍略帶前視(今日流動的股票不代表
  2015 年就流動)。真正的 point-in-time 成分股需 TEJ(付費)。但「規則篩全市場」已比
  「手列 50 檔」嚴謹一級,且把『誰進池』從主觀變客觀。
"""
import json
import os

import pandas as pd
import requests

import config

TWSE_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_DAY_ALL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
_UA = {"User-Agent": "Mozilla/5.0"}


def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return float("nan")


def fetch_market_snapshot() -> pd.DataFrame:
    """
    抓上市+上櫃當日全市場行情,過濾成「普通股」(代號 4 位數字),
    回傳 DataFrame[code, name, close, trade_value, market]。
    """
    rows = []
    # 上市(TWSE)
    try:
        d = requests.get(TWSE_DAY_ALL, timeout=40, headers=_UA).json()
        for r in d:
            rows.append({"code": str(r.get("Code", "")).strip(),
                         "name": r.get("Name", ""),
                         "close": _num(r.get("ClosingPrice")),
                         "trade_value": _num(r.get("TradeValue")),
                         "market": "twse"})
    except Exception:
        pass
    # 上櫃(TPEX)
    try:
        d = requests.get(TPEX_DAY_ALL, timeout=40, headers=_UA).json()
        for r in d:
            rows.append({"code": str(r.get("SecuritiesCompanyCode", "")).strip(),
                         "name": r.get("CompanyName", ""),
                         "close": _num(r.get("Close")),
                         "trade_value": _num(r.get("TransactionAmount")),
                         "market": "tpex"})
    except Exception:
        pass

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("TWSE/TPEX OpenAPI 無回應")
    # 只留「4 位數字」普通股(排除 ETF 00xx、權證、特別股、KY 以外的怪代號)
    df = df[df["code"].str.fullmatch(r"\d{4}")]
    df = df[df["close"] > 0].dropna(subset=["trade_value"])
    df = df.drop_duplicates("code").reset_index(drop=True)
    return df


def build_universe(top_n: int = 100, snapshot: pd.DataFrame = None,
                   save: bool = True) -> list:
    """依當日成交金額(流動性)取前 top_n 大普通股;回傳代號 list,並快取成 json。"""
    df = snapshot if snapshot is not None else fetch_market_snapshot()
    top = df.sort_values("trade_value", ascending=False).head(top_n)
    codes = top["code"].tolist()
    if save:
        path = os.path.join(config.DATA_DIR, "universe_auto.json")
        json.dump({"asof": "snapshot", "top_n": top_n, "codes": codes,
                   "names": dict(zip(top["code"], top["name"]))},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return codes


def load_cached_universe() -> list:
    """讀快取的自動選股池;沒有則回空 list。"""
    path = os.path.join(config.DATA_DIR, "universe_auto.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8")).get("codes", [])
    return []


if __name__ == "__main__":
    snap = fetch_market_snapshot()
    print(f"全市場普通股 {len(snap)} 檔(上市 {sum(snap.market=='twse')} / "
          f"上櫃 {sum(snap.market=='tpex')})")
    top = snap.sort_values("trade_value", ascending=False).head(100)
    print("\n流動性前 20 大(成交金額,億元):")
    for _, r in top.head(20).iterrows():
        print(f"  {r['code']} {r['name']:<8} {r['trade_value']/1e8:8.1f} 億")
    auto = set(top["code"])
    cur = set(config.UNIVERSE)
    print(f"\n自動前100 vs 手列50:重疊 {len(auto & cur)} 檔")
    print(f"手列50有、但不在自動前100:{sorted(cur - auto)}")
