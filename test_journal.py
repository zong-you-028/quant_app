# -*- coding: utf-8 -*-
"""
test_journal.py - 驗證投資紀錄(交易日誌)CRUD + 報酬彙總邏輯
流程:add_buy 開倉 -> list_trades 含計算欄位 -> close_trade 平倉 ->
summary 已實現損益 -> delete_trade 還原,最後確認 DB 乾淨。
不連網(報酬以本地 DB 收盤或賣出價計算),純函式易測。

★ 測試隔離:把 DB 指向臨時檔,絕不碰使用者真實的 data/market.db
  (get_conn() 每次動態讀 config.DB_PATH,故在 import journal 前改即可)。
"""
import os
import tempfile

import config

# 行情(2330 等)仍需從真實 DB 複製過來給 analyze 用?不需要:journal 測試只用
# 自建的買賣紀錄與合成 TEST。直接整個 journal/行情都導到臨時 DB,完全隔離。
config.DB_PATH = os.path.join(tempfile.gettempdir(), "quant_test_journal.db")
if os.path.exists(config.DB_PATH):
    os.remove(config.DB_PATH)

from core import journal


def test_journal_roundtrip():
    journal.init_journal()
    before = len(journal.list_trades())

    # 1) 開倉:投入 30000、買入價 600 -> 50 股
    tid = journal.add_buy("2330", 30000, 600, "台積電")
    trades = journal.list_trades()
    assert len(trades) == before + 1, "新增後筆數應 +1"
    rec = next(t for t in trades if t["id"] == tid)
    assert rec["status"] == "open"
    assert abs(rec["shares"] - 50.0) < 1e-9, "股數 = 投入 / 買入價"
    assert rec["buy_time"], "需有買入時間戳"

    # 2) 平倉:賣出價 660 -> 已實現損益 = 50*(660-600) = 3000、報酬 +10%
    journal.close_trade(tid, 660)
    rec = next(t for t in journal.list_trades() if t["id"] == tid)
    assert rec["status"] == "closed"
    assert rec["sell_price"] == 660
    assert rec["sell_time"], "需有賣出時間戳"
    assert abs(rec["pnl"] - 3000.0) < 1e-6, f"已實現損益應 3000,得 {rec['pnl']}"
    assert abs(rec["ret"] - 0.10) < 1e-9, f"報酬率應 +10%,得 {rec['ret']}"

    # 3) 彙總:平倉後該筆本金離開「總投入」,獲利進「已實現/累計投入」
    s = journal.summary()
    assert s["cost_all"] >= 30000              # 累計投入(含已平倉)仍含這 30000
    assert s["realized_pnl"] >= 3000 - 1e-6
    assert s["n_closed"] >= 1

    # 4) 刪除:還原成測試前狀態
    journal.delete_trade(tid)
    assert len(journal.list_trades()) == before, "刪除後應回到測試前筆數"


def test_add_buy_validation():
    for bad in (0, -1):
        try:
            journal.add_buy("2330", bad, 600)
            raise AssertionError("投入金額 <= 0 應拋錯")
        except ValueError:
            pass
        try:
            journal.add_buy("2330", 30000, bad)
            raise AssertionError("買入價 <= 0 應拋錯")
        except ValueError:
            pass
    try:
        journal.add_buy("", 30000, 600)
        raise AssertionError("空代號應拋錯")
    except ValueError:
        pass


def test_total_assets_and_snapshot():
    journal.init_journal()
    base = journal.summary()["total_assets"]
    tid = journal.add_buy("2330", 30000, 600, "台積電")
    s = journal.summary()
    # 恆等式:總資產 = 持倉現值 = 總投入(持倉) + 未實現
    assert abs(s["total_assets"] - s["market_value"]) < 1e-6
    assert abs(s["total_assets"] - (s["invested"] + s["unrealized_pnl"])) < 1e-6
    expected_current_return = s["unrealized_pnl"] / s["invested"]
    assert abs(s["current_return"] - expected_current_return) < 1e-9
    # 快照:寫入一筆 asset_history,且記下的總資產與 summary 一致
    before = len(journal.list_asset_history())
    snap = journal.snapshot_assets()
    hist = journal.list_asset_history()
    assert len(hist) == before + 1
    assert abs(hist[0]["total_assets"] - snap["total_assets"]) < 1e-6
    journal.update_asset_snapshot(
        hist[0]["id"], ts="2026-08-31 15:30", invested="10000", total_assets="12500")
    edited = journal.list_asset_history()[0]
    assert edited["ts"] == "2026-08-31 15:30"
    assert edited["invested"] == 10000
    assert edited["total_assets"] == 12500
    assert edited["total_pnl"] == 2500
    journal.delete_asset_snapshot(hist[0]["id"])   # 清掉測試快照,不留孤兒
    assert len(journal.list_asset_history()) == before
    journal.delete_trade(tid)


def test_dca_plan_autofill():
    journal.init_journal()
    # 自我隔離:清掉其他既有計畫,避免它們在 run_dca_update 時一起補單干擾斷言
    for p in journal.list_dca_plans():
        journal.delete_dca_plan(p["id"])
    n_before = len(journal.list_trades())
    # 用合成資料的 TEST 標的,每月扣款,從過去日期回補到指定「今天」
    pid = journal.add_dca_plan("TEST", 5000, "monthly", "2026-01-01")
    plan = next(p for p in journal.list_dca_plans() if p["id"] == pid)
    assert plan["active"] and plan["freq_label"] == "每月"
    r = journal.run_dca_update("2026-06-03")
    # 1/1,2/1,3/1,4/1,5/1,6/1 共 6 期
    assert r["created"] == 6, f"應補 6 筆,得 {r['created']}"
    # 下次扣款日推進到 7/1
    plan2 = next(p for p in journal.list_dca_plans() if p["id"] == pid)
    assert plan2["next_date"] == "2026-07-01"
    # 補出來的買入都標記為 dca
    dca_trades = [t for t in journal.list_trades() if t.get("source") == "dca"]
    assert len(dca_trades) >= 6
    # 停用後再更新不應再補
    journal.set_dca_active(pid, False)
    r2 = journal.run_dca_update("2026-12-31")
    assert r2["created"] == 0
    # 清理
    for t in journal.list_trades():
        if t.get("source") == "dca":
            journal.delete_trade(t["id"])
    journal.delete_dca_plan(pid)
    assert len(journal.list_trades()) == n_before


def test_summary_on_sell():
    """賣出後:總投入扣掉該筆本金、已實現=賣出所得−投入、總資產=持倉現值+回收現金。"""
    journal.init_journal()
    base = journal.summary()
    inv0, real0, cash0 = base["invested"], base["realized_pnl"], base["realized_proceeds"]
    mv0 = base["market_value"]                          # 賣出後總資產應回到此
    tid = journal.add_buy("2330", 30000, 600)          # 50 股
    s1 = journal.summary()
    assert abs(s1["invested"] - (inv0 + 30000)) < 1e-6, "買入後總投入應 +本金"
    journal.close_trade(tid, 660)                       # 賣出所得 = 50×660 = 33000
    s2 = journal.summary()
    # 總投入扣回 30000(該筆本金離開持倉)
    assert abs(s2["invested"] - inv0) < 1e-6, "賣出後總投入應扣掉該筆本金"
    # 已實現 = 賣出所得 − 投入 = 33000 - 30000 = 3000
    assert abs(s2["realized_pnl"] - (real0 + 3000)) < 1e-6, "已實現=賣出所得−投入"
    # 回收現金 += 賣出所得 33000(僅供參考,不計入總資產)
    assert abs(s2["realized_proceeds"] - (cash0 + 33000)) < 1e-6, "回收現金=賣出所得"
    # 總資產 = 持倉現值(賣出後該檔已離開,不含回收現金 -> 回到賣出前的持倉現值)
    assert abs(s2["total_assets"] - s2["market_value"]) < 1e-6, "總資產=持倉現值"
    assert abs(s2["total_assets"] - mv0) < 1e-6, "賣掉的檔不該再算進總資產"
    journal.delete_trade(tid)


def test_update_partial_reopen():
    """手動編輯 / 部分賣出 / 復原平倉(投資紀錄 v2)。"""
    journal.init_journal()
    tid = journal.add_buy("2330", 60000, 1000, "台積電",
                          buy_time="2026-05-01", note="原始備註")
    # 1) 手動編輯:改金額 -> 股數自動重算;設停損停利;改備註與時間
    journal.update_trade(tid, amount=50000, stop_loss=900, take_profit=1300,
                         buy_time="2026-05-02", note="改過了")
    t = next(x for x in journal.list_trades() if x["id"] == tid)
    assert abs(t["shares"] - 50.0) < 1e-9, "改金額後股數應重算 = 50000/1000"
    assert t["stop_loss"] == 900 and t["take_profit"] == 1300
    assert t["buy_time"] == "2026-05-02" and t["note"] == "改過了"
    # 2) 空字串清除停損;空金額 = 不變
    journal.update_trade(tid, stop_loss="", amount="")
    t = next(x for x in journal.list_trades() if x["id"] == tid)
    assert t["stop_loss"] is None and abs(t["amount"] - 50000) < 1e-6
    # 3) 部分賣出 20 股 @1100:拆成已平倉(成本2萬) + 持有中(3萬/30股)
    nid = journal.sell_partial(tid, 20, 1100)
    ts = {x["id"]: x for x in journal.list_trades()}
    assert ts[tid]["status"] == "open" and abs(ts[tid]["shares"] - 30) < 1e-9
    assert abs(ts[tid]["amount"] - 30000) < 1e-6
    assert ts[nid]["status"] == "closed"
    assert abs(ts[nid]["pnl"] - (20 * 1100 - 20000)) < 1e-6, "部分賣出損益錯誤"
    # 4) 整筆平倉 -> 復原成持有中
    journal.close_trade(tid, 1200)
    journal.reopen_trade(tid)
    t = next(x for x in journal.list_trades() if x["id"] == tid)
    assert t["status"] == "open" and t["sell_price"] is None
    # 5) 非法值要擋
    try:
        journal.update_trade(tid, amount=-5)
        raise AssertionError("負金額應拋錯")
    except ValueError:
        pass
    journal.delete_trade(tid)
    journal.delete_trade(nid)


def test_current_assets_cash_and_transactions():
    """期初持倉不動現金；新買賣同步現金，持倉按標的彙總。"""
    journal.init_journal()
    base_cash = journal.cash_balance()
    before_ids = {t["id"] for t in journal.list_trades()}
    journal.set_cash_balance(base_cash + 100000)

    initial_id = journal.add_current_asset("TEST", 10, 100, "2026-08-01")
    initial = next(t for t in journal.list_trades() if t["id"] == initial_id)
    assert initial["source"] == "initial"
    assert abs(journal.cash_balance() - (base_cash + 100000)) < 1e-6

    journal.record_buy("TEST", 5, 120, "2026-08-02")
    assert abs(journal.cash_balance() - (base_cash + 99400)) < 1e-6
    closed_ids = journal.record_sell("TEST", 8, 150, "2026-08-03")
    assert closed_ids
    assert abs(journal.cash_balance() - (base_cash + 100600)) < 1e-6

    position = next(p for p in journal.positions() if p["symbol"] == "TEST")
    assert abs(position["shares"] - 7) < 1e-9
    assert abs(position["cost"] - 800) < 1e-6
    assert abs(position["average_cost"] - (800 / 7)) < 1e-6
    s = journal.summary()
    assert abs(s["cash"] - journal.cash_balance()) < 1e-6
    assert abs(s["total_assets"] - (s["market_value"] + s["cash"])) < 1e-6

    for trade in journal.list_trades():
        if trade["id"] not in before_ids:
            journal.delete_trade(trade["id"])
    journal.set_cash_balance(base_cash, "測試清理")


def test_unpriced_holding_is_not_reported_as_cost_value():
    """行情不存在時，市值與收益必須保持待行情，不能退回成本。"""
    from main import make_summary_text

    journal.init_journal()
    tid = journal.add_buy("NOQUOTE", 10000, 100, "無行情測試")
    s = journal.summary()
    assert s["market_value"] is None
    assert s["total_assets"] is None
    assert s["unrealized_pnl"] is None
    assert s["current_return"] is None
    assert "待行情" in make_summary_text(s)
    journal.delete_trade(tid)


if __name__ == "__main__":
    test_journal_roundtrip()
    test_add_buy_validation()
    test_total_assets_and_snapshot()
    test_summary_on_sell()
    test_dca_plan_autofill()
    test_update_partial_reopen()
    test_current_assets_cash_and_transactions()
    print("投資紀錄(journal)CRUD + 總資產 + 定期定額 + 手動編輯/部分賣出/復原 PASSED。")
    test_unpriced_holding_is_not_reported_as_cost_value()
