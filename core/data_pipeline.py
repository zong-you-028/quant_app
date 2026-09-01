# -*- coding: utf-8 -*-
"""
data_pipeline.py - 資料層
職責:
  1. SQLite 連線與建表(ohlcv 日頻、chip_weekly 週頻籌碼)
  2. 特徵工程:MACD(DIF/DEA/OSC 及斜率/符號)、籌碼集中度特徵
  3. 日週對齊:週頻籌碼 reindex 到日頻後 ffill()
  4. seed_sample_data():無真實資料時產生合成資料,讓系統可立即執行
"""
import datetime as _dt
import gzip
import os
import sqlite3
import tempfile
import threading
import time
import numpy as np
import pandas as pd
import requests

import config


_INITIALIZED_PATHS = set()
_INIT_LOCK = threading.Lock()
_MARKET_SEED_VERSION = "2026-08-28-v1"
_FINMIND_BLOCKED_UNTIL = None


class FinMindBlockedError(RuntimeError):
    """FinMind rejected this Render egress IP; callers may use TWSE fallback."""


# ---------------------------------------------------------------------------
# DB 連線 / 建表
# ---------------------------------------------------------------------------
def get_conn():
    """回傳 SQLite 連線(connection)。每次使用後請自行 close。"""
    conn = sqlite3.connect(config.DB_PATH)
    return conn


def _hydrate_market_seed(conn) -> None:
    """Merge the bundled real-data seed into the actual APP_DATA_DIR database."""
    seed_gz = os.path.join(config.BASE_DIR, "data_seed", "market.db.gz")
    if not os.path.exists(seed_gz):
        return
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", ("market_seed_version",)
    ).fetchone()
    if row and row[0] == _MARKET_SEED_VERSION:
        return

    temp_path = None
    attached = False
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_path = temp_file.name
            with gzip.open(seed_gz, "rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    temp_file.write(chunk)
        conn.execute("ATTACH DATABASE ? AS market_seed", (temp_path,))
        attached = True
        for table in ("ohlcv", "chip_weekly", "stock_info"):
            conn.execute(
                f"INSERT OR REPLACE INTO {table} SELECT * FROM market_seed.{table}"
            )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("market_seed_version", _MARKET_SEED_VERSION),
        )
        conn.commit()
    finally:
        if attached:
            conn.execute("DETACH DATABASE market_seed")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def init_db():
    """建立 ohlcv(日頻)與 chip_weekly(週頻籌碼)兩張表(若不存在)。"""
    conn = get_conn()
    cur = conn.cursor()
    # 日頻 OHLCV;以 (symbol, date) 為主鍵避免重複寫入
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL,
            date   TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    # 週頻集保大戶籌碼;big_shares=大戶持股、total_shares=總股數
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chip_weekly (
            symbol       TEXT NOT NULL,
            date         TEXT NOT NULL,
            big_shares   REAL,
            total_shares REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    # 股票基本資料(代號 -> 中文名稱),快取 FinMind TaiwanStockInfo 查詢結果
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_info (
            symbol TEXT PRIMARY KEY,
            name   TEXT
        )
        """
    )
    conn.commit()
    _hydrate_market_seed(conn)
    conn.close()
    _INITIALIZED_PATHS.add(config.DB_PATH)


def ensure_db() -> None:
    """確保新建的 SQLite 檔已具備行情資料表。"""
    path = config.DB_PATH
    if path in _INITIALIZED_PATHS:
        return
    with _INIT_LOCK:
        if path not in _INITIALIZED_PATHS:
            init_db()


# ---------------------------------------------------------------------------
# 讀取原始資料
# ---------------------------------------------------------------------------
def load_ohlcv(symbol: str) -> pd.DataFrame:
    """讀取某標的日頻 OHLCV,以 DatetimeIndex 排序回傳。"""
    ensure_db()
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM ohlcv "
        "WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,),
    )
    conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def load_chip_weekly(symbol: str) -> pd.DataFrame:
    """讀取某標的週頻籌碼,以 DatetimeIndex 排序回傳。"""
    ensure_db()
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT date, big_shares, total_shares FROM chip_weekly "
        "WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,),
    )
    conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def has_symbol(symbol: str) -> bool:
    """檢查 DB 是否已有此標的的日頻資料。"""
    ensure_db()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM ohlcv WHERE symbol = ? LIMIT 1", (symbol,))
    row = cur.fetchone()
    conn.close()
    return row is not None


# ---------------------------------------------------------------------------
# 特徵工程
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    """指數移動平均(EMA);adjust=False 為交易常用遞迴定義。"""
    return series.ewm(span=span, adjust=False).mean()


def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """計算 MACD 系列特徵:DIF、DEA、OSC,以及斜率、符號、相對位置。"""
    close = df["close"]
    ema_fast = _ema(close, config.MACD_FAST)            # 快速 EMA
    ema_slow = _ema(close, config.MACD_SLOW)            # 慢速 EMA
    df["dif"] = ema_fast - ema_slow                      # DIF
    df["dea"] = _ema(df["dif"], config.MACD_SIGNAL)      # DEA = EMA9(DIF)
    df["osc"] = df["dif"] - df["dea"]                    # OSC(柱狀體)
    # 三者斜率(用一階差分 diff 近似當日變化量)
    df["dif_slope"] = df["dif"].diff()
    df["dea_slope"] = df["dea"].diff()
    df["osc_slope"] = df["osc"].diff()
    # OSC 正負號:>0 -> 1、<0 -> -1、=0 -> 0
    df["osc_sign"] = np.sign(df["osc"])
    # DIF 是否站上 DEA(多方訊號)
    df["dif_above_dea"] = (df["dif"] > df["dea"]).astype(int)
    return df


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI(相對強弱指標):用 Wilder EMA 平滑漲跌幅,回傳 0~100。"""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)             # 無波動時給中性 50


def _add_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    補強技術特徵(全部僅用當下/過去資訊,無未來洩漏):
      動能 mom_5/20/60、RSI、均線乖離 sma_gap、趨勢 trend/trend_up、
      波動 vol_20、布林位置 bb_pos、量能 vol_ratio。
    """
    close = df["close"]
    # 動能(多時間尺度報酬)
    df["mom_5"] = close.pct_change(5)
    df["mom_20"] = close.pct_change(20)
    df["mom_60"] = close.pct_change(60)
    # RSI(14)
    df["rsi_14"] = _rsi(close, 14)
    # 均線乖離與趨勢排列
    sma_fast = close.rolling(config.TREND_FAST).mean()
    sma_slow = close.rolling(config.TREND_SLOW).mean()
    df["sma_gap"] = close / sma_slow - 1.0
    df["trend"] = sma_fast / sma_slow - 1.0
    df["trend_up"] = (sma_fast > sma_slow).astype(int)
    # 波動度(20 日報酬標準差)
    df["vol_20"] = df["ret"].rolling(20).std()
    # 布林通道位置 %b
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    df["bb_pos"] = (close - ma20) / (2.0 * sd20.replace(0, np.nan))
    # 量能比(相對 20 日均量)
    vol_ma = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / vol_ma.replace(0, np.nan) - 1.0
    return df


def _add_chip_features(df: pd.DataFrame, chip: pd.DataFrame) -> pd.DataFrame:
    """
    計算籌碼特徵並做日週對齊。
      chip_ratio     = big_shares / total_shares(集中度)
      chip_ratio_chg = 週變化率 pct_change()
      chip_ratio_mom = 4 週動能(rolling 4 期變化)
    對齊方式:先在「週頻」上算好特徵,再 reindex 到日頻索引並 ffill()
    (新籌碼公布前沿用上週值,符合資訊可得時點,避免未來資訊洩漏)。
    """
    if chip is None or chip.empty:
        # 無籌碼資料時補 0,維持欄位完整
        df["chip_ratio"] = 0.0
        df["chip_ratio_chg"] = 0.0
        df["chip_ratio_mom"] = 0.0
        return df

    w = chip.copy()
    # 集中度:避免除以 0
    w["chip_ratio"] = w["big_shares"] / w["total_shares"].replace(0, np.nan)
    w["chip_ratio_chg"] = w["chip_ratio"].pct_change()                 # 週變化率
    w["chip_ratio_mom"] = w["chip_ratio"] - w["chip_ratio"].shift(4)   # 4 週動能

    chip_feats = w[["chip_ratio", "chip_ratio_chg", "chip_ratio_mom"]]
    # 日週對齊:reindex 到日頻索引 -> ffill(沿用最近一次公布值)
    aligned = chip_feats.reindex(df.index, method=None).ffill()
    df = df.join(aligned)
    # 對齊後仍可能在最前面留有 NaN(尚無任何籌碼公布),補 0
    df[["chip_ratio", "chip_ratio_chg", "chip_ratio_mom"]] = (
        df[["chip_ratio", "chip_ratio_chg", "chip_ratio_mom"]].fillna(0.0)
    )
    return df


def build_features(symbol: str) -> pd.DataFrame:
    """
    主特徵組裝函式:讀 OHLCV + 籌碼 -> 計算 MACD/籌碼特徵 -> 日週對齊。
    回傳含 FEATURE_COLS 全部欄位的日頻 DataFrame(已去除暖機期 NaN)。
    """
    df = load_ohlcv(symbol)
    if df.empty:
        raise ValueError(f"找不到標的 {symbol} 的 OHLCV 資料,請先 seed_sample_data 或匯入真實資料。")

    chip = load_chip_weekly(symbol)
    # 當日報酬要先算(技術特徵的波動度會用到);供 ML 貼標籤與回測,非特徵
    df["ret"] = df["close"].pct_change()
    df = _add_macd(df)                  # MACD 特徵
    df = _add_tech_features(df)         # 動能/RSI/趨勢/波動/量能 技術特徵
    df = _add_chip_features(df, chip)   # 籌碼特徵 + 日週對齊

    # 丟掉均線/動能等暖機期造成的 NaN(僅針對特徵欄位)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=config.FEATURE_COLS).copy()
    return df


# ---------------------------------------------------------------------------
# 合成資料(讓系統無真實資料也能立即運行)
# ---------------------------------------------------------------------------
def seed_sample_data(symbol: str, n_days: int = 500, seed: int = 7) -> None:
    """
    產生「帶趨勢的合成股價」+「與報酬弱相關的週籌碼」,寫入 SQLite。
    - 股價:幾何隨機漫步疊加緩慢正弦趨勢(geometric random walk + trend)
    - 籌碼:集中度受未來報酬輕微牽引(弱相關),模擬大戶提前佈局
    """
    init_db()
    rng = np.random.default_rng(seed)

    # --- 日頻交易日(以週一~週五為交易日)---
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # --- 合成股價:漂移 + 隨機波動 + 慢速趨勢 ---
    t = np.arange(n_days)
    trend = 0.0004 * np.sin(t / 60.0)                  # 緩慢週期趨勢
    noise = rng.normal(0, 0.012, n_days)               # 日波動
    daily_ret = 0.0002 + trend + noise                  # 合成日報酬
    close = 100.0 * np.exp(np.cumsum(daily_ret))       # 由報酬累積成價格

    # 由 close 反推 OHLC(加入日內隨機振幅)
    intraday = np.abs(rng.normal(0, 0.006, n_days))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = np.concatenate([[close[0]], close[:-1]])   # 開盤≈昨收
    volume = rng.integers(1000, 50000, n_days).astype(float)

    ohlcv = pd.DataFrame({
        "symbol": symbol,
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })

    # --- 週頻籌碼:每週五一筆;集中度與「未來一週報酬」弱相關 ---
    fwd_week_ret = pd.Series(daily_ret, index=dates).shift(-5).rolling(5).sum()
    fwd_week_ret = fwd_week_ret.fillna(0.0)
    # 週五取樣
    week_mask = dates.weekday == 4
    week_dates = dates[week_mask]
    # 集中度 base 0.45,加入弱相關訊號(係數 0.8)與雜訊
    base = 0.45
    signal = 0.8 * fwd_week_ret.values[week_mask]
    chip_noise = rng.normal(0, 0.01, week_mask.sum())
    chip_ratio = np.clip(base + signal + chip_noise, 0.05, 0.95)
    total_shares = rng.integers(800000, 1200000, week_mask.sum()).astype(float)
    big_shares = chip_ratio * total_shares

    chip = pd.DataFrame({
        "symbol": symbol,
        "date": week_dates.strftime("%Y-%m-%d"),
        "big_shares": big_shares,
        "total_shares": total_shares,
    })

    # --- 寫入 DB(先刪同標的舊資料,避免主鍵衝突)---
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
    cur.execute("DELETE FROM chip_weekly WHERE symbol = ?", (symbol,))
    conn.commit()
    ohlcv.to_sql("ohlcv", conn, if_exists="append", index=False)
    chip.to_sql("chip_weekly", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 真實資料(FinMind 台股 API)
# ---------------------------------------------------------------------------
def _finmind_get(dataset: str, data_id: str, start: str) -> pd.DataFrame:
    """
    呼叫 FinMind /api/v4/data,回傳該 dataset 的 DataFrame。
    免 token 可少量使用;config.FINMIND_TOKEN 有值時帶上以提高額度。
    status != 200 視為失敗(例如代號錯誤、超過免費額度),拋出例外。
    """
    global _FINMIND_BLOCKED_UNTIL
    if (_FINMIND_BLOCKED_UNTIL is not None
            and _dt.datetime.now() < _FINMIND_BLOCKED_UNTIL):
        raise FinMindBlockedError("FinMind ip banned (circuit open)")

    params = {"dataset": dataset, "data_id": data_id, "start_date": start}
    headers = {}
    if config.FINMIND_TOKEN:
        headers["Authorization"] = f"Bearer {config.FINMIND_TOKEN}"
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.get(
                config.FINMIND_URL, params=params, headers=headers, timeout=30
            )
        except requests.RequestException as ex:
            last_error = ex
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

        try:
            js = resp.json()
        except ValueError:
            js = {}
        if 400 <= resp.status_code < 500:
            # Invalid credentials, quota or IP ban will not recover by retrying.
            msg = js.get("msg") or resp.reason or "client error"
            if resp.status_code == 403 and "ip banned" in str(msg).lower():
                _FINMIND_BLOCKED_UNTIL = _dt.datetime.now() + _dt.timedelta(minutes=30)
                raise FinMindBlockedError(
                    f"FinMind {dataset}/{data_id} HTTP 403: {msg}"
                )
            raise RuntimeError(f"FinMind {dataset}/{data_id} HTTP {resp.status_code}: {msg}")
        try:
            resp.raise_for_status()
        except requests.RequestException as ex:
            last_error = ex
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        if js.get("status") != 200:
            raise RuntimeError(f"FinMind {dataset} 失敗:{js.get('msg')}")
        return pd.DataFrame(js.get("data", []))
    raise RuntimeError(f"FinMind {dataset}/{data_id} 更新失敗: {last_error}")


def _twse_recent_prices(symbol: str, asof=None) -> pd.DataFrame:
    """Fetch current/previous month OHLCV from the official TWSE endpoint."""
    end = pd.Timestamp(asof or _dt.datetime.now()).normalize()
    months = [end, end - pd.DateOffset(months=1)]
    rows = []
    for month in months:
        resp = requests.get(
            "https://www.twse.com.tw/exchangeReport/STOCK_DAY",
            params={"response": "json", "date": month.strftime("%Y%m01"),
                    "stockNo": symbol},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        resp.raise_for_status()
        js = resp.json()
        if js.get("stat") != "OK":
            continue
        for values in js.get("data", []):
            if len(values) < 9:
                continue
            y, m, d = (int(part) for part in values[0].split("/"))

            def number(value):
                return pd.to_numeric(str(value).replace(",", ""), errors="coerce")

            rows.append({
                "symbol": symbol,
                "date": f"{y + 1911:04d}-{m:02d}-{d:02d}",
                "open": number(values[3]), "high": number(values[4]),
                "low": number(values[5]), "close": number(values[6]),
                "volume": number(values[1]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"TWSE 查無 {symbol} 最近行情")
    return out.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")


def fetch_twse_recent_data(symbol: str) -> None:
    """Merge official TWSE recent prices into the existing local history."""
    init_db()
    fresh = _twse_recent_prices(symbol)
    conn = get_conn()
    old = pd.read_sql_query(
        "SELECT symbol,date,open,high,low,close,volume FROM ohlcv WHERE symbol = ?",
        conn, params=(symbol,),
    )
    combined = pd.concat([old, fresh], ignore_index=True)
    combined = combined.drop_duplicates("date", keep="last").sort_values("date")
    combined = _adjust_corporate_actions(combined)
    cur = conn.cursor()
    cur.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
    conn.commit()
    combined.to_sql("ohlcv", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def _adjust_corporate_actions(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    清理資料錯誤 + 回溯還原公司行動(除權息/股票分割/減資)造成的價格跳空。
      1) 丟棄收盤<=0 或 NaN 的壞資料列(例:停牌/資料源補 0,否則會讓比例變 0/inf)。
      2) 台股單日漲跌幅 ±10%,故「收盤對收盤」比例超過 ±CA_JUMP_THRESHOLD 者
         必為公司行動(非交易)。以「由後往前」累乘還原因子,將該跳空日「之前」的
         OHLC 全部乘上跳空比例,使報酬序列連續;最新一段價格維持市場原值。
    回傳清理 + 還原後的同結構 DataFrame(欄位、順序不變)。
    """
    df = ohlcv.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    # 1) 丟壞列(收盤非正或 NaN)
    df = df[df["close"] > 0].reset_index(drop=True)
    if len(df) < 2:
        return df

    for col in ("open", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    ratio = (close / close.shift(1)).values             # 收盤對收盤比例
    thr = config.CA_JUMP_THRESHOLD
    n = len(df)

    # 2) 由後往前累乘還原因子:跳空日之前的價格乘上比例 -> 連續化
    factors = np.ones(n, dtype=float)
    f = 1.0
    for i in range(n - 1, -1, -1):
        factors[i] = f
        r = ratio[i]
        if i > 0 and np.isfinite(r) and (r < 1.0 - thr or r > 1.0 + thr):
            f *= r                                       # 此跳空之前的歷史價一律縮放
    if not np.isclose(f, 1.0) or (factors != 1.0).any():
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factors                  # 還原 OHLC(volume 非特徵,不調整)
    return df


def fetch_real_data(symbol: str, start: str = None) -> None:
    """
    從 FinMind 抓真實台股資料,寫入與合成資料完全相同的 ohlcv / chip_weekly 表,
    因此上層(build_features -> train -> backtest)完全不需更動。
      - 股價:TaiwanStockPrice(open/max/min/close/Trading_Volume)
      - 籌碼:TaiwanStockShareholding 的「外資持股」作為大戶集中度代理
              big_shares   = ForeignInvestmentShares(外資持股張數)
              total_shares = NumberOfSharesIssued(發行股數)
              -> chip_ratio = 外資持股比例(台股大型股最主要的法人籌碼指標)
    註:集保股權分散表(精準大戶級距)需 FinMind 付費等級,
        故免費版改用外資持股比例,語意同為「大戶集中度」。
    """
    start = start or config.FINMIND_START
    init_db()

    # 1) 日頻股價
    try:
        px = _finmind_get("TaiwanStockPrice", symbol, start)
    except FinMindBlockedError:
        fetch_twse_recent_data(symbol)
        return
    if px.empty:
        raise RuntimeError(f"FinMind 查無 {symbol} 股價(代號是否正確?例:2330)")
    ohlcv = pd.DataFrame({
        "symbol": symbol,
        "date": px["date"],
        "open": px["open"], "high": px["max"], "low": px["min"],
        "close": px["close"], "volume": px["Trading_Volume"],
    })
    # 還原股價:清理壞列 + 回溯還原除權息/分割/減資跳空(免費版無還原股價,故自行處理)
    ohlcv = _adjust_corporate_actions(ohlcv)

    # 2) 籌碼:外資持股 -> 大戶集中度代理(抓不到就留空,_add_chip_features 會補 0)
    try:
        sh = _finmind_get("TaiwanStockShareholding", symbol, start)
    except Exception:
        sh = pd.DataFrame()
    if not sh.empty and {"ForeignInvestmentShares", "NumberOfSharesIssued"} <= set(sh.columns):
        chip = pd.DataFrame({
            "symbol": symbol,
            "date": sh["date"],
            "big_shares": pd.to_numeric(sh["ForeignInvestmentShares"], errors="coerce"),
            "total_shares": pd.to_numeric(sh["NumberOfSharesIssued"], errors="coerce"),
        }).dropna()
        chip = chip[chip["total_shares"] > 0]
    else:
        chip = pd.DataFrame(columns=["symbol", "date", "big_shares", "total_shares"])

    # 3) 寫入 DB(先刪同標的舊資料,避免主鍵衝突)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
    cur.execute("DELETE FROM chip_weekly WHERE symbol = ?", (symbol,))
    conn.commit()
    ohlcv.to_sql("ohlcv", conn, if_exists="append", index=False)
    if not chip.empty:
        chip.to_sql("chip_weekly", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 股票名稱查詢(FinMind TaiwanStockInfo,免費;查到後寫入 stock_info 快取)
# ---------------------------------------------------------------------------
_NAME_CACHE = {}   # 程序內記憶體快取(symbol -> name),避免重複查 DB / API


def get_stock_name(symbol: str) -> str:
    """
    回傳股票中文名稱(例:2330 -> 台積電)。三層快取:
      1) 記憶體 _NAME_CACHE
      2) SQLite stock_info 表(跨程序持久)
      3) FinMind TaiwanStockInfo API(免費),查到後寫回 DB + 記憶體
    查無(合成/離線/代號錯誤)時回傳空字串,呼叫端自行退回顯示代號。
    """
    symbol = symbol.strip().upper()
    if symbol in _NAME_CACHE:
        return _NAME_CACHE[symbol]

    init_db()
    # 2) DB 快取
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM stock_info WHERE symbol = ?", (symbol,))
    row = cur.fetchone()
    conn.close()
    if row and row[0]:
        _NAME_CACHE[symbol] = row[0]
        return row[0]

    # 3) FinMind API(僅在資料來源為 finmind 時嘗試)
    name = ""
    if config.DATA_SOURCE == "finmind":
        try:
            info = _finmind_get("TaiwanStockInfo", symbol, config.FINMIND_START)
            if not info.empty and "stock_name" in info.columns:
                match = info[info["stock_id"].astype(str) == symbol]
                if not match.empty:
                    name = str(match["stock_name"].iloc[-1])
        except Exception:
            name = ""

    if name:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO stock_info (symbol, name) VALUES (?, ?)",
                    (symbol, name))
        conn.commit()
        conn.close()
        _NAME_CACHE[symbol] = name
    return name


# ---------------------------------------------------------------------------
# 每日資料更新(增量:只刷新「落後到最新交易日」的標的)
#   做法:偵測 DB 內某檔最後一筆日期是否落後最近交易日;落後就重抓整段並覆寫
#   (重抓才能正確重算除權息回溯還原 factor,純 append 會破壞還原連續性)。
#   以「整體節流」(預設 6 小時)避免每次開 app 都重抓 50 檔。
# ---------------------------------------------------------------------------
def _ensure_meta():
    conn = get_conn()
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()


def _meta_get(key: str):
    _ensure_meta()
    conn = get_conn()
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def _meta_set(key: str, value: str):
    _ensure_meta()
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                 (key, str(value)))
    conn.commit()
    conn.close()


def last_ohlcv_date(symbol: str):
    """DB 內某檔最後一筆日期(pd.Timestamp);查無回 None。"""
    ensure_db()
    conn = get_conn()
    row = conn.execute("SELECT MAX(date) FROM ohlcv WHERE symbol = ?",
                       (symbol,)).fetchone()
    conn.close()
    if row and row[0]:
        try:
            return pd.to_datetime(row[0])
        except Exception:
            return None
    return None


def _last_trading_day(asof: _dt.datetime = None) -> pd.Timestamp:
    """
    估「最近一個應該已有資料的交易日」:
      FinMind 約傍晚才更新當日 K 棒,故 18:00 前先看「昨天」;再往前跳過週末。
    (無內建台股假日表;若遇平日休市,頂多多刷一次抓不到新資料,由節流吸收。)
    """
    now = asof or _dt.datetime.now()
    d = now.date()
    if now.hour < 18:
        d = d - _dt.timedelta(days=1)
    while d.weekday() >= 5:                    # 5=六、6=日 -> 往前找平日
        d = d - _dt.timedelta(days=1)
    return pd.Timestamp(d)


def needs_update(symbol: str, asof: _dt.datetime = None) -> bool:
    """此檔是否落後最近交易日(合成資料來源時一律不更新)。"""
    if config.DATA_SOURCE != "finmind":
        return False
    last = last_ohlcv_date(symbol)
    if last is None:
        return True
    return last.normalize() < _last_trading_day(asof).normalize()


def update_data(symbol: str, force: bool = False) -> str:
    """單檔:落後才重抓整段覆寫。回傳 'updated' / 'current' / 'failed'。"""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return "failed"
    if not force and not needs_update(symbol):
        return "current"
    try:
        fetch_real_data(symbol)
        return "updated"
    except Exception:
        return "failed"


def update_symbols(symbols, force: bool = False, ignore_throttle: bool = False,
                   throttle_hours: float = 6.0, progress=None) -> dict:
    """
    批次更新每日資料(只刷新落後者)。
      force=True:每檔都強制重抓;ignore_throttle=True:略過整體時間節流(手動按鈕用)。
    回傳 {updated, current, failed, throttled(bool), asof}。
    """
    asof = _last_trading_day().strftime("%Y-%m-%d")
    if not force and not ignore_throttle:
        ts = _meta_get("last_refresh")
        if ts:
            try:
                age = (_dt.datetime.now() - _dt.datetime.fromisoformat(ts)).total_seconds()
                if age < throttle_hours * 3600:
                    return {"updated": 0, "current": 0, "failed": 0,
                            "throttled": True, "asof": asof}
            except Exception:
                pass

    res = {"updated": 0, "current": 0, "failed": 0, "throttled": False,
           "asof": asof, "failed_symbols": [], "errors": {}}
    seen = set()
    syms = [s for s in symbols if s]
    for i, s in enumerate(syms):
        s = s.strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        if not force and not needs_update(s):
            st = "current"
        else:
            try:
                fetch_real_data(s)
                st = "updated"
            except Exception as ex:
                st = "failed"
                res["failed_symbols"].append(s)
                res["errors"][s] = str(ex)
        res[st] = res.get(st, 0) + 1
        if progress:
            progress(i + 1, len(syms), s, st)
    # 全部失敗時不要啟動六小時節流，讓修好 Token/API 後可立即重試。
    if res["failed"] < len(seen):
        _meta_set("last_refresh", _dt.datetime.now().isoformat())
    return res


def ensure_data(symbol: str, force: bool = False) -> str:
    """
    確保 DB 內有此標的資料,回傳實際使用的來源字串("cached"/"finmind"/"synthetic")。
      - DB 已有且未強制更新 -> 直接沿用("cached")
      - DATA_SOURCE=="finmind" -> 嘗試抓真實資料;失敗且允許 fallback 才退回合成
      - 其餘 -> 合成資料
    """
    if has_symbol(symbol) and not force:
        return "cached"
    if config.DATA_SOURCE == "finmind":
        try:
            fetch_real_data(symbol)
            if has_symbol(symbol):
                return "finmind"
        except Exception as ex:
            if not config.FALLBACK_TO_SYNTHETIC:
                raise
            print(f"[warn] 真實資料抓取失敗({ex}),改用合成資料。")
    seed_sample_data(symbol)
    return "synthetic"


if __name__ == "__main__":
    # 簡易自測:抓真實 2330 並印出特徵尾段(失敗自動退回合成)
    src = ensure_data("2330")
    print("data source:", src)
    out = build_features("2330")
    print(out[config.FEATURE_COLS + ["ret"]].tail())
