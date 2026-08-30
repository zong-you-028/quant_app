# -*- coding: utf-8 -*-
"""
journal.py - 投資紀錄(交易日誌 / 持倉帳本 + 總資產 + 定期定額)
從 0 開始記錄每一筆投資:投入金額、買入點位/時間、賣出點位/時間、報酬,
並彙總「總資產」(持倉現值 + 已實現所得),支援「定期定額(DCA)」自動回補買入。
資料存在與行情同一個 SQLite(config.DB_PATH),純函式、易測試。

資料模型:
  trades       一筆 = 一個來回:id, symbol, name, amount(投入), shares(=amount/buy_price),
               buy_price, buy_time, sell_price, sell_time, status('open'/'closed'),
               source('manual'/'dca')
  dca_plans    定期定額計畫:id, symbol, name, amount(每期), freq, next_date, active
  asset_history 總資產快照:ts, invested, market_value, realized, total_assets, total_pnl

報酬計算:
  股數 shares = 投入金額 / 買入價
  已平倉:proceeds = shares × 賣出價;損益 = proceeds − 投入
  持有中:現值 = shares × 最新收盤;浮動損益 = 現值 − 投入
總資產:total_assets = 持倉現值(market_value) + 已實現賣出所得(realized_proceeds)
        = 累計投入 + 總損益(已實現 + 未實現)
"""
import calendar
import math
import datetime as _dt

import pandas as pd

from core.data_pipeline import load_ohlcv, get_stock_name, ensure_data
from core.journal_storage import dialect, get_journal_conn, insert_id

get_conn = get_journal_conn


# DCA 頻率 -> 中文標籤(供 UI;也是合法值白名單)
FREQ_LABELS = {"weekly": "每週", "biweekly": "每兩週", "monthly": "每月"}


def _now() -> str:
    """目前時間字串(到分鐘),作為買/賣的預設時間戳。"""
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _parse_date(s):
    """字串 -> date(支援 'YYYY-MM-DD' 與 'YYYY-MM-DD HH:MM');失敗回 None。"""
    if not s:
        return None
    if isinstance(s, _dt.date) and not isinstance(s, _dt.datetime):
        return s
    if isinstance(s, _dt.datetime):
        return s.date()
    for f in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(str(s), f).date()
        except Exception:
            continue
    return None


def _advance_date(d: _dt.date, freq: str) -> _dt.date:
    """把日期往後推一個週期(weekly/biweekly/monthly;月底自動夾日)。"""
    if freq == "weekly":
        return d + _dt.timedelta(days=7)
    if freq == "biweekly":
        return d + _dt.timedelta(days=14)
    # monthly:加一個日曆月,日數超過當月天數時夾到月底
    y, m = d.year, d.month + 1
    if m > 12:
        y, m = y + 1, 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return _dt.date(y, m, day)


def init_journal() -> None:
    """建立/升級 trades、dca_plans、asset_history 三張表(若不存在)。"""
    conn = get_conn()
    pg = dialect(conn) == "postgres"
    id_type = "BIGSERIAL PRIMARY KEY" if pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS trades (
            id         {id_type},
            symbol     TEXT NOT NULL,
            name       TEXT,
            amount     REAL NOT NULL,        -- 投入金額(新台幣)
            shares     REAL NOT NULL,        -- 股數 = amount / buy_price
            buy_price  REAL NOT NULL,
            buy_time   TEXT NOT NULL,
            sell_price REAL,
            sell_time  TEXT,
            status     TEXT NOT NULL DEFAULT 'open',  -- open / closed
            source     TEXT NOT NULL DEFAULT 'manual', -- manual / dca / rotation
            stop_loss  REAL,                  -- 停損價(由本月推薦加入時記錄)
            take_profit REAL                  -- 停利價(由本月推薦加入時記錄)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS dca_plans (
            id         {id_type},
            symbol     TEXT NOT NULL,
            name       TEXT,
            amount     REAL NOT NULL,        -- 每期投入金額
            freq       TEXT NOT NULL DEFAULT 'monthly',
            next_date  TEXT NOT NULL,        -- 下次扣款日(YYYY-MM-DD)
            active     INTEGER NOT NULL DEFAULT 1,
            created_at TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS asset_history (
            id           {id_type},
            ts           TEXT NOT NULL,
            invested     REAL,
            market_value REAL,
            realized     REAL,
            total_assets REAL,
            total_pnl    REAL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS cash_movements (
            id         {id_type},
            amount     REAL NOT NULL,
            kind       TEXT NOT NULL,
            trade_id   INTEGER,
            ts         TEXT NOT NULL,
            note       TEXT
        )
        """
    )
    # 舊版 trades 表升級:補上新欄(若不存在)
    if pg:
        cols = [r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'trades'"
        ).fetchall()]
    else:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()]
    if "source" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN source TEXT DEFAULT 'manual'")
    if "stop_loss" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN stop_loss REAL")
    if "take_profit" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN take_profit REAL")
    if "note" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN note TEXT")
    if "last_rotation" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN last_rotation TEXT")
    conn.commit()
    conn.close()


def get_last_close(symbol: str):
    """從本地 DB 取最新收盤價(不連網);查無回傳 None。"""
    try:
        df = load_ohlcv(symbol)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return None


def get_close_on(symbol: str, date_str: str):
    """取某日(或其之前最近一個交易日)的收盤價;查無回傳 None。"""
    try:
        df = load_ohlcv(symbol)
        if df is None or df.empty:
            return None
        target = pd.to_datetime(date_str)
        sub = df[df.index <= target]
        if sub.empty:
            return None
        return float(sub["close"].iloc[-1])
    except Exception:
        return None


def _add_cash_movement(amount: float, kind: str, trade_id: int = None,
                       ts: str = None, note: str = "") -> int:
    amount = float(amount)
    if not math.isfinite(amount):
        raise ValueError("現金金額格式錯誤")
    init_journal()
    conn = get_conn()
    movement_id = insert_id(
        conn,
        "INSERT INTO cash_movements (amount, kind, trade_id, ts, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (amount, kind, trade_id, ts or _now(), (note or "").strip() or None),
    )
    conn.commit()
    conn.close()
    return int(movement_id)


def cash_balance() -> float:
    """目前帳戶現金；由期初現金、買賣與手動調整累加。"""
    init_journal()
    conn = get_conn()
    row = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM cash_movements").fetchone()
    conn.close()
    return float(row[0] or 0.0)


def set_cash_balance(balance: float, note: str = "匯入目前現金") -> float:
    """把帳戶現金調整至指定餘額，保留差額異動以便追溯。"""
    balance = float(balance)
    if not math.isfinite(balance) or balance < 0:
        raise ValueError("現金餘額不可小於 0")
    delta = balance - cash_balance()
    if abs(delta) > 1e-9:
        _add_cash_movement(delta, "cash_adjustment", note=note)
    return balance


def list_cash_movements(limit: int = 100) -> list:
    init_journal()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, amount, kind, trade_id, ts, note "
        "FROM cash_movements ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "amount": float(r[1]), "kind": r[2],
         "trade_id": r[3], "ts": r[4], "note": r[5] or ""}
        for r in rows
    ]


def add_buy(symbol: str, amount: float, buy_price: float,
            name: str = "", buy_time: str = None, source: str = "manual",
            stop_loss: float = None, take_profit: float = None,
            note: str = "") -> int:
    """
    記錄一筆買入(開倉)。金額/價格須 > 0。回傳新紀錄 id。
    name 留空時自動查中文名;buy_time 留空時用現在時間。
    source 標記來源(manual/dca/rotation);由本月推薦加入時可帶 stop_loss/take_profit。
    """
    symbol = (symbol or "").strip().upper()
    amount = float(amount)
    buy_price = float(buy_price)
    if not symbol:
        raise ValueError("代號不可空白")
    if amount <= 0:
        raise ValueError("投入金額需大於 0")
    if buy_price <= 0:
        raise ValueError("買入價需大於 0")
    stop_loss = float(stop_loss) if stop_loss not in (None, "") else None
    take_profit = float(take_profit) if take_profit not in (None, "") else None

    name = name or get_stock_name(symbol) or ""
    shares = amount / buy_price
    buy_time = buy_time or _now()

    init_journal()
    conn = get_conn()
    # 來源=輪動推薦者,最近輪替日 = 買入日(之後每月續抱時更新)
    last_rotation = buy_time.split(" ")[0] if source == "rotation" else None
    new_id = insert_id(conn,
        "INSERT INTO trades (symbol, name, amount, shares, buy_price, buy_time, "
        "status, source, stop_loss, take_profit, note, last_rotation) "
        "VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)",
        (symbol, name, amount, shares, buy_price, buy_time, source,
         stop_loss, take_profit, (note or "").strip() or None, last_rotation),
    )
    conn.commit()
    conn.close()
    return int(new_id)


def close_trade(trade_id: int, sell_price: float, sell_time: str = None) -> None:
    """記錄某筆紀錄的賣出(平倉)。sell_price 須 > 0;sell_time 預設現在。"""
    sell_price = float(sell_price)
    if sell_price <= 0:
        raise ValueError("賣出價需大於 0")
    sell_time = sell_time or _now()
    init_journal()
    conn = get_conn()
    conn.execute(
        "UPDATE trades SET sell_price = ?, sell_time = ?, status = 'closed' "
        "WHERE id = ?",
        (sell_price, sell_time, int(trade_id)),
    )
    conn.commit()
    conn.close()


def delete_trade(trade_id: int) -> None:
    """刪除一筆紀錄。"""
    init_journal()
    conn = get_conn()
    conn.execute("DELETE FROM cash_movements WHERE trade_id = ?", (int(trade_id),))
    conn.execute("DELETE FROM trades WHERE id = ?", (int(trade_id),))
    conn.commit()
    conn.close()


# 手動編輯白名單(數字欄 / 文字欄)
_NUM_FIELDS = {"amount", "shares", "buy_price", "sell_price",
               "stop_loss", "take_profit"}
_TEXT_FIELDS = {"name", "buy_time", "sell_time", "note"}


def update_trade(trade_id: int, **fields) -> None:
    """
    手動修改一筆紀錄的任意欄位(金額/買入價/買賣時間/停損停利/備註/名稱/股數)。
    規則:
      - amount / buy_price / sell_price / shares 須 > 0;空字串 = 不變。
      - stop_loss / take_profit 傳空字串或 None = 清除(不設停損/停利)。
      - 改了 amount 或 buy_price 且未明給 shares 時,股數自動重算 = 金額/買入價。
      - buy_time / sell_time 空字串 = 不變(時間不可清空);note/name 空字串 = 清除。
    """
    init_journal()
    conn = get_conn()
    row = conn.execute(
        "SELECT amount, buy_price FROM trades WHERE id = ?",
        (int(trade_id),)).fetchone()
    if not row:
        conn.close()
        raise ValueError(f"找不到紀錄 #{trade_id}")
    cur_amount, cur_bp = float(row[0]), float(row[1])

    clean = {}
    for k, v in fields.items():
        if k in _NUM_FIELDS:
            if v is None or str(v).strip() == "":
                if k in ("stop_loss", "take_profit"):
                    clean[k] = None              # 清除停損/停利
                continue                          # 其餘數字欄空白 = 不變
            x = float(v)
            if k != "stop_loss" and k != "take_profit" and x <= 0:
                conn.close()
                raise ValueError(f"{k} 需大於 0")
            clean[k] = x
        elif k in _TEXT_FIELDS:
            s = (str(v).strip() if v is not None else "")
            if k in ("buy_time", "sell_time"):
                if s:
                    clean[k] = s                  # 時間:空=不變
            else:
                clean[k] = s or None              # note/name:空=清除

    # 股數自動重算(除非使用者明改 shares)
    if "shares" not in clean and ("amount" in clean or "buy_price" in clean):
        amt = clean.get("amount", cur_amount)
        bp = clean.get("buy_price", cur_bp)
        if amt > 0 and bp > 0:
            clean["shares"] = amt / bp

    if clean:
        sets = ", ".join(f"{k} = ?" for k in clean)
        conn.execute(f"UPDATE trades SET {sets} WHERE id = ?",
                     (*clean.values(), int(trade_id)))
        conn.commit()
    conn.close()


def mark_rotation(symbol: str, when: str = None,
                  stop_loss: float = None, take_profit: float = None) -> int:
    """
    月度換股「續抱確認」:把該代號所有「持有中、來源=輪動」的紀錄更新最近輪替日
    (預設今天),可一併更新停損/停利(依現價重算 -> 等於月度移動停損,只升不降
    由呼叫端決定傳入值)。回傳更新筆數。
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("代號不可空白")
    when = when or _dt.date.today().strftime("%Y-%m-%d")
    init_journal()
    conn = get_conn()
    sets, vals = ["last_rotation = ?"], [when]
    if stop_loss is not None:
        sets.append("stop_loss = ?")
        vals.append(float(stop_loss))
    if take_profit is not None:
        sets.append("take_profit = ?")
        vals.append(float(take_profit))
    vals.append(symbol)
    cur = conn.execute(
        f"UPDATE trades SET {', '.join(sets)} "
        "WHERE symbol = ? AND status = 'open' AND source = 'rotation'", vals)
    conn.commit()
    n = cur.rowcount
    conn.close()
    return int(n)


def reopen_trade(trade_id: int) -> None:
    """把誤平倉的紀錄復原成持有中(清掉賣出價/賣出時間)。"""
    init_journal()
    conn = get_conn()
    conn.execute(
        "UPDATE trades SET status = 'open', sell_price = NULL, sell_time = NULL "
        "WHERE id = ?", (int(trade_id),))
    conn.commit()
    conn.close()


def sell_partial(trade_id: int, sell_shares: float, sell_price: float,
                 sell_time: str = None) -> int:
    """
    部分賣出:把一筆持倉拆成「已平倉(賣掉的股數)」+「持有中(剩餘股數)」。
    成本(amount)按股數比例分拆;賣出股數 >= 全部時等同整筆平倉。
    回傳已平倉那筆的 id。
    """
    sell_shares = float(sell_shares)
    sell_price = float(sell_price)
    if sell_shares <= 0:
        raise ValueError("賣出股數需大於 0")
    if sell_price <= 0:
        raise ValueError("賣出價需大於 0")
    init_journal()
    conn = get_conn()
    row = conn.execute(
        "SELECT symbol, name, amount, shares, buy_price, buy_time, status, "
        "source, stop_loss, take_profit, note FROM trades WHERE id = ?",
        (int(trade_id),)).fetchone()
    conn.close()
    if not row:
        raise ValueError(f"找不到紀錄 #{trade_id}")
    (sym, name, amount, shares, bp, bt, status, source, sl, tp, note) = row
    if status != "open":
        raise ValueError("已平倉的紀錄不能再賣出")
    shares = float(shares)
    if sell_shares >= shares - 1e-9:             # 賣光 = 整筆平倉
        close_trade(trade_id, sell_price, sell_time)
        return int(trade_id)

    frac = sell_shares / shares
    sold_amount = float(amount) * frac
    sell_time = sell_time or _now()
    conn = get_conn()
    new_id = insert_id(conn,
        "INSERT INTO trades (symbol, name, amount, shares, buy_price, buy_time, "
        "sell_price, sell_time, status, source, stop_loss, take_profit, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'closed', ?, ?, ?, ?)",
        (sym, name, sold_amount, sell_shares, bp, bt, sell_price, sell_time,
         source, sl, tp, note))
    conn.execute(
        "UPDATE trades SET amount = ?, shares = ? WHERE id = ?",
        (float(amount) - sold_amount, shares - sell_shares, int(trade_id)))
    conn.commit()
    conn.close()
    return int(new_id)


def _hold_days(buy_time: str, end_time: str) -> int:
    """兩個時間字串相差天數(解析失敗回 0)。"""
    a, b = _parse_date(buy_time), _parse_date(end_time)
    if a and b:
        return max(0, (b - a).days)
    return 0


def list_trades() -> list:
    """
    回傳所有紀錄(新到舊),每筆附帶計算欄位:
      pnl(損益)、ret(報酬率)、cur_price(持有中=最新收盤/已平倉=賣出價)、
      value(現值/賣出所得)、hold_days、status、source。
    """
    init_journal()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, symbol, name, amount, shares, buy_price, buy_time, "
        "sell_price, sell_time, status, source, stop_loss, take_profit, note, "
        "last_rotation FROM trades ORDER BY id DESC"
    ).fetchall()
    conn.close()

    out = []
    today = _now()
    for (tid, sym, name, amount, shares, bp, bt, sp, st, status, source,
         sl, tp, note, last_rot) in rows:
        rec = {
            "id": tid, "symbol": sym, "name": name or "",
            "amount": float(amount), "shares": float(shares),
            "buy_price": float(bp), "buy_time": bt,
            "sell_price": float(sp) if sp is not None else None,
            "sell_time": st, "status": status, "source": source or "manual",
            "stop_loss": float(sl) if sl is not None else None,
            "take_profit": float(tp) if tp is not None else None,
            "note": note or "",
            "last_rotation": last_rot or "",
        }
        if status == "closed" and sp is not None:
            proceeds = shares * float(sp)
            rec["cur_price"] = float(sp)
            rec["value"] = proceeds
            rec["pnl"] = proceeds - float(amount)
            rec["ret"] = (proceeds / float(amount) - 1.0) if amount else 0.0
            rec["hold_days"] = _hold_days(bt, st)
        else:  # open
            cur = get_last_close(sym)
            rec["cur_price"] = cur
            if cur is not None:
                value = shares * cur
                rec["value"] = value
                rec["pnl"] = value - float(amount)
                rec["ret"] = (value / float(amount) - 1.0) if amount else 0.0
            else:
                rec["value"] = None
                rec["pnl"] = None
                rec["ret"] = None
            rec["hold_days"] = _hold_days(bt, today)
        out.append(rec)
    return out


def add_current_asset(symbol: str, shares: float, average_cost: float,
                      asof: str = None, note: str = "") -> int:
    """匯入一筆目前持倉；建立期初 lot，不變動現金。"""
    shares, average_cost = float(shares), float(average_cost)
    if not math.isfinite(shares) or shares <= 0:
        raise ValueError("持有數量需大於 0")
    if not math.isfinite(average_cost) or average_cost <= 0:
        raise ValueError("平均成本需大於 0")
    return add_buy(symbol, shares * average_cost, average_cost,
                   buy_time=asof, source="initial", note=note)


def record_buy(symbol: str, shares: float, price: float,
               when: str = None, note: str = "") -> int:
    """記錄新買進交易並同步扣除現金。"""
    shares, price = float(shares), float(price)
    if not math.isfinite(shares) or shares <= 0:
        raise ValueError("買進數量需大於 0")
    if not math.isfinite(price) or price <= 0:
        raise ValueError("成交價需大於 0")
    trade_id = add_buy(symbol, shares * price, price, buy_time=when,
                       source="manual", note=note)
    _add_cash_movement(-(shares * price), "buy", trade_id, when,
                       f"買進 {symbol.upper()}")
    return trade_id


def record_sell(symbol: str, shares: float, price: float,
                when: str = None, note: str = "") -> list:
    """依 FIFO 賣出既有持倉並把賣出所得加入現金；回傳平倉 lot id。"""
    symbol = (symbol or "").strip().upper()
    shares, price = float(shares), float(price)
    if not symbol:
        raise ValueError("代號不可空白")
    if not math.isfinite(shares) or shares <= 0:
        raise ValueError("賣出數量需大於 0")
    if not math.isfinite(price) or price <= 0:
        raise ValueError("成交價需大於 0")
    lots = sorted(
        [t for t in list_trades()
         if t["status"] == "open" and t["symbol"] == symbol],
        key=lambda t: t["id"],
    )
    available = sum(t["shares"] for t in lots)
    if shares > available + 1e-9:
        raise ValueError(f"持有數量不足（目前 {available:g}）")
    remaining, closed_ids = shares, []
    for lot in lots:
        if remaining <= 1e-9:
            break
        sold = min(remaining, lot["shares"])
        closed_id = sell_partial(lot["id"], sold, price, when)
        _add_cash_movement(sold * price, "sell", closed_id, when,
                           note or f"賣出 {symbol}")
        closed_ids.append(closed_id)
        remaining -= sold
    return closed_ids


def positions() -> list:
    """按標的彙總目前持倉，與逐筆交易紀錄分開呈現。"""
    grouped = {}
    for trade in list_trades():
        if trade["status"] != "open":
            continue
        position = grouped.setdefault(trade["symbol"], {
            "symbol": trade["symbol"], "name": trade.get("name") or "",
            "shares": 0.0, "cost": 0.0, "value": 0.0, "priced": True,
            "lots": 0,
        })
        position["shares"] += trade["shares"]
        position["cost"] += trade["amount"]
        position["lots"] += 1
        if trade.get("value") is None:
            position["priced"] = False
        else:
            position["value"] += trade["value"]
    result = []
    for position in grouped.values():
        position["average_cost"] = position["cost"] / position["shares"]
        position["market_value"] = position["value"] if position["priced"] else None
        position["current_price"] = (
            position["market_value"] / position["shares"]
            if position["market_value"] is not None else None
        )
        position["pnl"] = (
            position["market_value"] - position["cost"]
            if position["market_value"] is not None else None
        )
        position["return"] = (
            position["pnl"] / position["cost"]
            if position["pnl"] is not None and position["cost"] else None
        )
        result.append(position)
    return sorted(result, key=lambda p: p["cost"], reverse=True)


def summary() -> dict:
    """
    整體績效彙總(從 0 開始;「目前持倉 + 已實現」雙軌會計):
      invested          : 總投入 = 「目前持倉」的成本(賣出後該筆本金移除,只算 open)
      cost_all          : 累計投入(含已平倉;僅作報酬率基底)
      realized_pnl      : 已實現損益 = Σ(賣出所得 − 投入)[已平倉]
      unrealized_pnl    : 未實現損益 = Σ(現值 − 投入)[持有中]
      total_pnl         : 已實現 + 未實現
      total_return      : total_pnl / cost_all(對「曾投入的總本金」算報酬)
      market_value      : 持倉現值(持有中;無現價則以成本計)
      realized_proceeds : 賣出回收現金 = Σ 賣出所得[已平倉](僅供參考,不計入總資產)
      total_assets      : 總資產 = 持倉現值(= 總投入 + 未實現;賣出回收的現金視為已領出)
      n_open/n_closed/n_total
    ★ 賣出時:該筆本金與損益都離開「持倉/總資產」,獲利另記在「已實現損益」(歷史)。
    """
    trades = list_trades()
    open_t = [t for t in trades if t["status"] == "open"]
    closed_t = [t for t in trades if t["status"] == "closed"]

    invested = sum(t["amount"] for t in open_t)             # 總投入 = 只算持倉(賣出移除)
    cost_all = sum(t["amount"] for t in trades)             # 累計投入(報酬率基底)
    realized = sum(t["pnl"] for t in closed_t if t["pnl"] is not None)
    unrealized = sum(t["pnl"] for t in open_t if t["pnl"] is not None)
    total = realized + unrealized

    def _val(t):  # 持倉現值;無現價(value None)時退回以成本計,確保總資產不漏算
        v = t.get("value")
        return v if v is not None else t["amount"]

    market_value = sum(_val(t) for t in open_t)
    realized_proceeds = sum(t["value"] for t in closed_t if t.get("value") is not None)
    cash = cash_balance()
    total_assets = market_value + cash
    return {
        "invested": invested,
        "cost_all": cost_all,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": total,
        "total_return": (total / cost_all) if cost_all else 0.0,
        "market_value": market_value,
        "cash": cash,
        "realized_proceeds": realized_proceeds,
        "total_assets": total_assets,
        "n_open": len(open_t),
        "n_closed": len(closed_t),
        "n_total": len(trades),
    }


# ---------------------------------------------------------------------------
# 總資產快照(記錄總資產隨時間的變化,讓「從 0 成長」看得見)
# ---------------------------------------------------------------------------
def snapshot_assets() -> dict:
    """把目前 summary() 的總資產等數字寫入 asset_history,回傳該筆 summary。"""
    s = summary()
    init_journal()
    conn = get_conn()
    conn.execute(
        "INSERT INTO asset_history (ts, invested, market_value, realized, total_assets, total_pnl) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (_now(), s["invested"], s["market_value"], s["realized_proceeds"],
         s["total_assets"], s["total_pnl"]),
    )
    conn.commit()
    conn.close()
    return s


def list_asset_history(limit: int = 30) -> list:
    """回傳最近 limit 筆總資產快照(新到舊,含 id 供刪除/編輯)。"""
    init_journal()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, ts, invested, market_value, realized, total_assets, total_pnl "
        "FROM asset_history ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    conn.close()
    return [{"id": r[0], "ts": r[1], "invested": r[2], "market_value": r[3],
             "realized": r[4], "total_assets": r[5], "total_pnl": r[6]}
            for r in rows]


def delete_asset_snapshot(snap_id: int) -> None:
    """刪除一筆總資產快照(誤按/重複記錄用)。"""
    init_journal()
    conn = get_conn()
    conn.execute("DELETE FROM asset_history WHERE id = ?", (int(snap_id),))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 定期定額(DCA):建立計畫 + 依排程自動回補買入
# ---------------------------------------------------------------------------
def add_dca_plan(symbol: str, amount: float, freq: str = "monthly",
                 start_date: str = None) -> int:
    """新增一個定期定額計畫(每期固定金額)。回傳計畫 id。"""
    symbol = (symbol or "").strip().upper()
    amount = float(amount)
    if not symbol:
        raise ValueError("代號不可空白")
    if amount <= 0:
        raise ValueError("每期金額需大於 0")
    if freq not in FREQ_LABELS:
        freq = "monthly"
    sd = _parse_date(start_date) or _dt.date.today()
    name = get_stock_name(symbol) or ""
    init_journal()
    conn = get_conn()
    pid = insert_id(conn,
        "INSERT INTO dca_plans (symbol, name, amount, freq, next_date, active, created_at) "
        "VALUES (?, ?, ?, ?, ?, 1, ?)",
        (symbol, name, amount, freq, sd.strftime("%Y-%m-%d"), _now()),
    )
    conn.commit()
    conn.close()
    return int(pid)


def list_dca_plans() -> list:
    """回傳所有定期定額計畫(新到舊)。"""
    init_journal()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, symbol, name, amount, freq, next_date, active "
        "FROM dca_plans ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "symbol": r[1], "name": r[2] or "", "amount": float(r[3]),
             "freq": r[4], "freq_label": FREQ_LABELS.get(r[4], r[4]),
             "next_date": r[5], "active": bool(r[6])}
            for r in rows]


def set_dca_active(plan_id: int, active: bool) -> None:
    """啟用 / 停用 一個定期定額計畫。"""
    init_journal()
    conn = get_conn()
    conn.execute("UPDATE dca_plans SET active = ? WHERE id = ?",
                 (1 if active else 0, int(plan_id)))
    conn.commit()
    conn.close()


def delete_dca_plan(plan_id: int) -> None:
    """刪除一個定期定額計畫(不影響已產生的買入紀錄)。"""
    init_journal()
    conn = get_conn()
    conn.execute("DELETE FROM dca_plans WHERE id = ?", (int(plan_id),))
    conn.commit()
    conn.close()


def run_dca_update(today: str = None) -> dict:
    """
    自動更新所有「啟用中」的定期定額計畫:把 next_date 起到今天為止、每個到期日
    依當日(或之前最近交易日)收盤價各補一筆買入(source='dca'),並推進 next_date。
    需要歷史價,故每檔會先 ensure_data 確保 DB 有資料。
    回傳 {"created": 產生筆數, "plans": 啟用計畫數}。
    """
    init_journal()
    today_d = _parse_date(today) or _dt.date.today()
    conn = get_conn()
    plans = conn.execute(
        "SELECT id, symbol, name, amount, freq, next_date FROM dca_plans WHERE active = 1"
    ).fetchall()
    conn.close()

    created = 0
    for (pid, sym, name, amount, freq, next_date) in plans:
        try:
            ensure_data(sym)          # 確保有歷史價可回補(此處允許連網)
        except Exception:
            pass
        nd = _parse_date(next_date)
        if nd is None:
            continue
        guard = 0
        while nd <= today_d and guard < 600:   # guard 防呆(約 50 年的月扣款)
            guard += 1
            price = get_close_on(sym, nd.strftime("%Y-%m-%d"))
            if price and price > 0:
                add_buy(sym, float(amount), price, name=name or "",
                        buy_time=nd.strftime("%Y-%m-%d") + " 09:00", source="dca")
                created += 1
            nd = _advance_date(nd, freq)        # 無價(早於資料起點/休市)亦推進,避免卡住
        conn = get_conn()
        conn.execute("UPDATE dca_plans SET next_date = ? WHERE id = ?",
                     (nd.strftime("%Y-%m-%d"), pid))
        conn.commit()
        conn.close()
    return {"created": created, "plans": len(plans)}


if __name__ == "__main__":
    init_journal()
    print("trades:", len(list_trades()), "| summary:", summary())
    print("dca plans:", len(list_dca_plans()), "| snapshots:", len(list_asset_history()))
