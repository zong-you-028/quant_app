"""Stock kinds, journal coverage and pending t+2 execution are distinct."""
from copy import deepcopy

import pandas as pd
import pytest

import main as app
from core.portfolio_reconcile import reconcile_portfolio
from test_model_ui import build_app, button, texts
import asyncio


def result(current=("A", "B"), upcoming=None, stale=False):
    upcoming = current if upcoming is None else upcoming
    return {"score_mode": app.ACTIVE_MODEL_ID, "top_k": 8,
            "model_current_weights": {s: .125 for s in current},
            "model_current_asof": "2026-10-02", "model_current_execution_date": "2026-09-29",
            "holdings": list(upcoming), "names": {s: f"公司{s}" for s in set(current) | set(upcoming)},
            "target_execution_date": None,
            "data_quality": {"stale": stale, "target_date": "2026-10-02"}}


def symbols(rows):
    return [row["symbol"] for row in rows]


def test_compares_all_lots_and_sources_without_changing_input():
    model = result()
    positions = [{"symbol": " a ", "shares": 100., "source": "manual"},
                 {"symbol": "A", "shares": 30., "source": "dca"},
                 {"symbol": "C", "name": "模型外公司", "shares": 50., "source": "manual"},
                 {"symbol": "Z", "shares": 0.}]
    saved = deepcopy((model, positions))
    view = reconcile_portfolio(model, positions)
    assert view["verified"]
    assert view["actual_count"] == 2
    assert symbols(view["keep"]) == ["A"] and view["keep"][0]["shares"] == 130.
    assert view["actionable_buys"] == ["B"]
    assert view["actionable_sells"] == ["C"]
    assert view["model_cash_weight"] == .75
    assert (model, positions) == saved


def test_pending_change_does_not_sell_outgoing_stock_early_or_churn_early_purchase():
    positions = [{"symbol": s, "shares": 10.} for s in ("A", "B", "C", "D")]
    view = reconcile_portfolio(result(upcoming=("B", "C")), positions)
    assert view["pending_change"]
    assert symbols(view["keep"]) == ["A", "B"]  # A remains until delayed execution
    assert symbols(view["early"]) == ["C"]
    assert view["actionable_sells"] == ["D"]
    assert symbols(view["planned_exits"]) == ["A", "D"]
    assert len(view["current"]) == len(view["next_target"]) == 2


def test_missing_outgoing_stock_is_not_a_buy_then_sell_instruction():
    view = reconcile_portfolio(result(upcoming=("B", "C")), [])
    assert symbols(view["current"]) == ["A", "B"]
    assert symbols(view["unheld_outgoing"]) == ["A"]
    assert view["actionable_buys"] == ["B"]
    assert symbols(view["planned_buys"]) == ["B", "C"]
    text = texts(app.make_reconciliation_rows(result(upcoming=("B", "C")), []))
    assert "目前未持有、下一次將移出，先核對時序" in text


def test_eight_targets_never_become_sixteen_when_rotated():
    old, new = tuple(f"OLD{i}" for i in range(8)), tuple(f"NEW{i}" for i in range(8))
    view = reconcile_portfolio(result(old, new), [{"symbol": s, "shares": 10.} for s in old])
    assert len(view["current"]) == len(view["next_target"]) == view["max_stocks"] == 8
    assert view["model_cash_weight"] == 0
    assert len(view["keep"]) == 8 and view["actionable_sells"] == []
    assert len(view["planned_buys"]) == len(view["planned_exits"]) == 8
    with pytest.raises(ValueError, match="超過 8 檔"):
        reconcile_portfolio(result(old + new), [])


def test_cash_transition_waits_for_execution():
    positions = [{"symbol": "A", "shares": 10.}]
    pending = reconcile_portfolio(result(("A",), ()), positions)
    assert symbols(pending["keep"]) == ["A"]
    assert pending["actionable_sells"] == []
    executed = reconcile_portfolio(result((), ()), positions)
    assert executed["model_cash_weight"] == 1.
    assert executed["actionable_sells"] == ["A"]


@pytest.mark.parametrize("change", ["stale", "older_date", "missing_date", "other_model", "missing_state"])
def test_unverified_results_never_emit_actionable_stock_lists(change):
    model = result()
    if change == "stale":
        model["data_quality"]["stale"] = True
    elif change == "older_date":
        model["model_current_asof"] = "2026-09-30"
    elif change == "missing_date":
        model["model_current_asof"] = None
    elif change == "other_model":
        model["score_mode"] = "production"
    else:
        model.pop("model_current_weights")
    view = reconcile_portfolio(model, [{"symbol": "C", "shares": 10.}])
    assert not view["verified"] and view["issues"]
    assert view["actionable_buys"] == view["actionable_sells"] == []
    if change == "missing_state":
        assert view["exit"] == []


@pytest.mark.parametrize("shares", [-1., float("nan"), float("inf")])
def test_invalid_book_never_silently_drops_a_position(shares):
    with pytest.raises(ValueError, match="帳本"):
        reconcile_portfolio(result(), [{"symbol": "A", "shares": shares}])


def test_renderer_explains_current_upcoming_and_historical_views():
    text = texts(app.make_reconciliation_rows(result(upcoming=("B", "C")), [{"symbol": "A", "shares": 10.}]))
    assert "目前應持有 2/8" in text and "下一次目標（尚待執行）" in text
    assert "應保留（帳本已有） · 1" in text and "目前名單尚未切換" in text
    assert "不是持有 16 檔" in text
    history = texts(app.make_reconciliation_rows(result(stale=True), [{"symbol": "C", "shares": 10.}]))
    assert "歷史模型名單" in history and "應退出" not in history and "待補入" not in history


def test_recheck_refreshes_book_without_rerunning_model_and_update_invalidates(monkeypatch):
    from test_model_ui import rotation_result
    from core import market_regime
    page = build_app(monkeypatch)
    model = rotation_result(stale=False)
    model.update(result())
    monkeypatch.setattr(app, "monthly_holdings", lambda: model)
    monkeypatch.setattr(app, "model_data_quality", lambda: model["data_quality"])
    monkeypatch.setattr(app.journal, "positions", lambda: [{"symbol": "A", "shares": 100.}])
    assert button(page.controls, "核對我的持倉").disabled
    asyncio.run(button(page.controls, "計算輪動名單").on_click(None))
    assert "應保留（帳本已有） · 1" in texts(page.controls)
    monkeypatch.setattr(app, "monthly_holdings", lambda: pytest.fail("recheck must not rerun the model"))
    monkeypatch.setattr(app.journal, "positions", lambda: [{"symbol": "C", "shares": 5.}])
    asyncio.run(button(page.controls, "核對我的持倉").on_click(None))
    assert "依模型應退出（目前名單外） · 1" in texts(page.controls)
    assert "公司C C · 帳本 5 股" in texts(page.controls) or "C · 帳本 5 股" in texts(page.controls)
    monkeypatch.setattr(app, "update_symbols", lambda *a, **kw: {"failed": 0, "stale": 0})
    monkeypatch.setattr(app, "format_update_status", lambda r: "updated")
    monkeypatch.setattr(market_regime, "refresh_sox", lambda **kw: pd.Series([100.]))
    monkeypatch.setattr(app.journal, "positions", lambda: [])
    asyncio.run(button(page.controls, "更新每日資料").on_click(None))
    assert button(page.controls, "核對我的持倉").disabled
    assert "依模型應退出（目前名單外） ·" not in texts(page.controls)


def test_book_read_failure_removes_old_comparison_and_restores_button(monkeypatch):
    from test_model_ui import rotation_result
    page = build_app(monkeypatch)
    model = rotation_result(stale=False)
    model.update(result())
    monkeypatch.setattr(app, "monthly_holdings", lambda: model)
    asyncio.run(button(page.controls, "計算輪動名單").on_click(None))
    monkeypatch.setattr(app.journal, "positions", lambda: (_ for _ in ()).throw(RuntimeError("book offline")))
    asyncio.run(button(page.controls, "核對我的持倉").on_click(None))
    assert "book offline" in texts(page.controls)
    assert "應保留（帳本已有） ·" not in texts(page.controls)
    assert not button(page.controls, "核對我的持倉").disabled


def test_refresh_quality_catches_new_date_without_turning_old_result_into_live_advice(monkeypatch):
    from test_model_ui import rotation_result
    page = build_app(monkeypatch)
    model = rotation_result(stale=False)
    model.update(result())
    monkeypatch.setattr(app, "monthly_holdings", lambda: model)
    asyncio.run(button(page.controls, "計算輪動名單").on_click(None))
    monkeypatch.setattr(app, "model_data_quality", lambda: {"stale": False, "target_date": "2026-10-05"})
    asyncio.run(button(page.controls, "核對我的持倉").on_click(None))
    assert "歷史模型名單" in texts(page.controls)
    assert "待補入（目前名單尚缺） ·" not in texts(page.controls)
    # A later journal refresh must retain the newly discovered model staleness.
    button(page.controls, "展開投資紀錄（0 筆）").on_click(None)
    assert "歷史模型名單" in texts(page.controls)
    assert "待補入（目前名單尚缺） ·" not in texts(page.controls)
