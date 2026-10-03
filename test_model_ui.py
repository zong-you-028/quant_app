"""Single active model, update lifecycle and stale-data trading guards."""
import asyncio
from types import SimpleNamespace

import flet as ft
import pandas as pd
import pytest

import main as app


def walk(control):
    if isinstance(control, (list, tuple)):
        for child in control:
            yield from walk(child)
        return
    if not isinstance(control, ft.Control):
        return
    yield control
    for child in getattr(control, "controls", []) or []:
        yield from walk(child)
    yield from walk(getattr(control, "content", None))


def texts(controls):
    return "\n".join(str(item.value) for item in walk(controls) if isinstance(item, ft.Text))


def button(controls, label):
    return next(item for item in walk(controls)
                if str(getattr(item, "content", "")) == label or getattr(item, "text", None) == label)


def rotation_result(stale=True):
    return {
        "score_mode": app.ACTIVE_MODEL_ID, "score_kind": "percentile", "model_label": app.active_model_spec()["label"],
        "validation_status": "歷史診斷・待前瞻驗證", "data_quality": {
            "asof": "2026-09-01", "target_date": "2026-10-02", "stale": stale,
            "stale_symbols": ["A"] if stale else [], "coverage": 0. if stale else 1., "total_symbols": 1,
        },
        "cagr": .18, "market_cagr": .12, "mdd": -.15, "market_mdd": -.2,
        "benchmark": "006208", "eval_start": "2024-08-06", "eval_end": "2026-09-01",
        "mom_days": 60, "top_k": 8, "rebal_days": 20, "last_date": "2026-09-01",
        "ranking": [("A", "測試公司", .85)], "held": ["A"], "holdings": ["A"],
        "names": {"A": "測試公司"}, "buys": ["A"], "abs_mom": True,
        "selection_date": "2026-08-20", "selection_execution_date": "2026-08-24",
        "target_execution_date": "2026-09-03",
    }


def stock_result(mode=app.ACTIVE_MODEL_ID):
    prices = pd.Series([100., 101., 102., 103., 104.], index=pd.bdate_range("2026-08-26", periods=5))
    return {
        "symbol": "A", "name": "測試公司", "verdict_short": "模型入選", "verdict_color": "#D32F2F",
        "score_mode": mode, "model_label": app.active_model_spec()["label"],
        "score_kind": "percentile", "score": .85, "mom": .11, "absolute_momentum": .16,
        "rank": 2, "n": 40, "in_top_k": True, "asof": "2026-09-01", "note": "通過閘門",
        "day_change": 1., "day_change_pct": .01, "last_close": 104., "price5": prices,
        "price6": prices, "vol_annual": .2, "mom5": .04, "mom20": .08, "mdd": -.15,
        "data_quality": rotation_result()["data_quality"],
    }


def test_scores_and_real_returns_keep_different_units(monkeypatch):
    controls = {key: ft.Text("--") for key in (
        "signal_name", "signal_sub", "note_hint", "price_val", "price_chg", "price_date",
        "rank_val", "abs_val", "vol_val", "mom_line")}
    controls.update({key: ft.Container() for key in ("signal_card", "price_holder", "chart_holder")})
    monkeypatch.setattr(app, "price_image", lambda *args: ft.Image(src="x"))
    monkeypatch.setattr(app, "stock_price_image", lambda *args: ft.Image(src="x"))
    app.apply_fit(controls, stock_result())
    assert "排名分 85.0 / 100" in controls["signal_sub"].value
    assert "85.0%" not in controls["signal_sub"].value
    assert controls["abs_val"].value == "+16%"
    assert "資料未達更新目標" in controls["note_hint"].value


def test_stale_cards_disable_add_and_renew(monkeypatch):
    monkeypatch.setattr(app.journal, "get_last_close", lambda symbol: 100.)
    monkeypatch.setattr(app.rotation, "stop_take_levels", lambda *args: (90., 120.))
    rows = app.make_holdings_rows(rotation_result(), on_add=lambda *args: None)
    assert button(rows, "加入庫存").disabled
    assert "排名分 85.0 / 100" in texts(rows)
    assert "扣成本後年化" in texts(rows) and "006208 同期年化" in texts(rows)
    assert "目標成交日 2026-09-03" in texts(rows)
    assert "最多 8 檔" in texts(rows) and "上期模型名單可保留至前 16 名" in texts(rows)
    assert "滿倉進場" not in texts(rows)
    label = app.active_model_spec()["label"]
    assert sum(isinstance(item, ft.Text) and item.value == label for item in walk(rows)) == 1
    compact_rows = app.make_holdings_rows(rotation_result(), on_add=lambda *args: None,
                                         include_model_header=False)
    assert sum(isinstance(item, ft.Text) and item.value == label for item in walk(compact_rows)) == 0
    assert button(compact_rows, "加入庫存").disabled
    rows = app.make_holdings_rows(rotation_result(), held_trades={
        "A": {"shares": 10., "buy_price": 100., "buy_time": "2026-08-20"}},
        on_renew=lambda symbol: None)
    assert button(rows, "續抱·更新輪替日").disabled


def test_risk_off_empty_held_never_revives_candidates():
    result = rotation_result()
    result["held"] = []
    result["cash_symbols"] = ["A"]
    result["sells"] = ["B"]
    result["names"]["B"] = "上期持股"
    rows = app.make_holdings_rows(result)
    assert "本期沒有通過全部閘門" in texts(rows)
    assert "測試公司" not in texts(rows)
    assert "未通過已啟用閘門" in texts(rows) and "名額保留現金" in texts(rows)
    assert "動能翻負" not in texts(rows)
    assert "移出本期目標名單" in texts(rows) and "移出前" not in texts(rows)


def test_pending_new_target_does_not_show_old_execution_date():
    result = rotation_result()
    result["held"] = []
    result["target_execution_date"] = None
    result["target_execution_pending"] = True
    text = texts(app.make_holdings_rows(result))
    assert "目標成交日 待後續交易日確認" in text
    assert "目標成交日 2026-08-24" not in text


class FakePage:
    def __init__(self):
        self.controls = []
        self.window = SimpleNamespace()
        self.updates = 0

    def add(self, control):
        self.controls.append(control)

    def update(self):
        self.updates += 1


def build_app(monkeypatch):
    for method in ("list_trades", "positions", "list_asset_history", "list_dca_plans"):
        monkeypatch.setattr(app.journal, method, lambda: [])
    monkeypatch.setattr(app.journal, "cash_balance", lambda: 0.)
    monkeypatch.setattr(app.journal, "summary", lambda: {
        "invested": 0., "realized_pnl": 0., "n_open": 0, "n_closed": 0,
    })
    monkeypatch.setattr(app, "model_data_quality", lambda: rotation_result()["data_quality"])
    monkeypatch.setattr(app.journal, "get_last_close", lambda symbol: 100.)
    monkeypatch.setattr(app.rotation, "stop_take_levels", lambda *args: (90., 120.))
    page = FakePage()
    app._build_app(page)
    return page


def test_single_active_model_update_invalidation_and_stale_callback_guard(monkeypatch):
    page = build_app(monkeypatch)
    seen = []
    assert not any(isinstance(item, ft.Dropdown) and item.label == "選股模型" for item in walk(page.controls))
    assert "低換手多視窗模型" in texts(page.controls)

    def scan_model(**kwargs):
        seen.append(("scan", kwargs["score_mode"]))
        result = rotation_result()
        result["score_mode"] = kwargs["score_mode"]
        result["data_quality"]["signal_asof"] = "2026-08-20"
        result["data_quality"]["chip_asof"] = "2026-08-28"
        return result

    monkeypatch.setattr(app.rotation, "run_rotation", scan_model)
    monkeypatch.setattr(app.journal, "add_buy", lambda *args, **kwargs: seen.append(("buy", args)))
    scan = button(page.controls, "計算輪動名單")
    asyncio.run(scan.on_click(None))
    label = app.active_model_spec()["label"]
    assert sum(isinstance(item, ft.Text) and item.value == label for item in walk(page.controls)) == 1
    assert "選股依據 2026-08-20" in texts(page.controls)
    assert "外資持股最舊日期 2026-08-28" in texts(page.controls)
    add = button(page.controls, "加入庫存")
    assert add.disabled
    # Disabled controls must still be guarded when a stale callback is invoked directly.
    add.on_click(None)
    assert all(kind != "buy" for kind, _ in seen)
    from core import market_regime
    monkeypatch.setattr(market_regime, "refresh_sox", lambda **kwargs: pd.Series([100.]))
    monkeypatch.setattr(app, "update_symbols", lambda *args, **kwargs: {"stale": 1, "failed": 0})
    monkeypatch.setattr(app, "format_update_status", lambda result: "尚未達更新目標")
    asyncio.run(button(page.controls, "更新每日資料").on_click(None))
    assert "本批更新結束，請依資料時效重新計算目前模型的輪動名單。" in texts(page.controls)
    assert not any(getattr(item, "content", None) == "加入庫存" for item in walk(page.controls))
    add.on_click(None)
    assert all(kind != "buy" for kind, _ in seen)

    def analyze(symbol, score_mode=None):
        seen.append(("stock", score_mode))
        return stock_result(score_mode)

    monkeypatch.setattr(app.rotation, "analyze_stock", analyze)
    monkeypatch.setattr(app, "load_ohlcv", lambda symbol: pd.DataFrame())
    monkeypatch.setattr(app, "analyze_exit_radar", lambda *args: {"available": False, "message": "test"})
    monkeypatch.setattr(app, "price_image", lambda *args: ft.Image(src="x"))
    monkeypatch.setattr(app, "stock_price_image", lambda *args: ft.Image(src="x"))
    asyncio.run(button(page.controls, "分析這檔").on_click(None))
    assert ("scan", app.ACTIVE_MODEL_ID) in seen and ("stock", app.ACTIVE_MODEL_ID) in seen
    assert not button(page.controls, "新增目前資產").disabled


def test_failed_calculation_restores_controls_and_retry(monkeypatch):
    page = build_app(monkeypatch)

    def fail(**kwargs):
        assert button(page.controls, "分析這檔").disabled
        assert button(page.controls, "更新每日資料").disabled
        raise RuntimeError("test failure")

    monkeypatch.setattr(app, "monthly_holdings", fail)
    asyncio.run(button(page.controls, "計算輪動名單").on_click(None))
    assert not button(page.controls, "重新計算名單").disabled
    assert "test failure" in texts(page.controls)
    assert not any(item.visible for item in walk(page.controls) if isinstance(item, ft.ProgressBar))


def test_research_model_payload_is_rejected_in_single_model_app(monkeypatch):
    page = build_app(monkeypatch)
    result = rotation_result(stale=False)
    result["score_mode"] = "production"
    monkeypatch.setattr(app, "monthly_holdings", lambda: result)
    asyncio.run(button(page.controls, "計算輪動名單").on_click(None))
    assert "運算結果模型不一致" in texts(page.controls)
    assert not any(getattr(item, "content", None) == "加入庫存" for item in walk(page.controls))
    assert not button(page.controls, "重新計算名單").disabled


def test_usage_guide_explains_actual_trade_and_inventory_bookkeeping(monkeypatch):
    page = build_app(monkeypatch)
    guide = next(item for item in walk(page.controls) if isinstance(item, ft.ExpansionTile))
    assert not getattr(guide, "expanded", False)
    content = texts(guide)
    assert "更新每日資料" in content and "計算輪動名單" in content
    assert "目標成交日" in content and "不會自動下單" in content
    assert "不扣現金" in content and "重複記帳" in content
    assert "不代表市場已成交" in content


def test_startup_update_shared_progress_no_duplicate_holdings_fetch(monkeypatch):
    import time
    from core import market_regime
    scheduled, calls, frames = [], [], []
    monkeypatch.delenv("QUANT_APP_OFFLINE", raising=False)
    monkeypatch.setattr(FakePage, "run_task", lambda page, task: scheduled.append(task), raising=False)
    monkeypatch.setattr(FakePage, "update", lambda page: frames.append(texts(page.controls)))
    page = build_app(monkeypatch)
    monkeypatch.setattr(app.journal, "refresh_open_market_data", lambda: pytest.fail("must not download holdings twice"))
    monkeypatch.setattr(market_regime, "refresh_sox", lambda **kwargs: pd.Series([100.]))

    def update(symbols, **kwargs):
        calls.append((symbols, kwargs))
        assert button(page.controls, "更新中…").disabled
        assert button(page.controls, "分析這檔").disabled
        kwargs["on_start"](0, 2, "2330", "更新行情")
        time.sleep(.1)
        return {"pending": 1, "time_budget_reached": True, "failed": 0, "stale": 0}

    monkeypatch.setattr(app, "update_symbols", update)
    monkeypatch.setattr(app, "format_update_status", lambda r: "待續抓 1")

    async def run():
        auto = asyncio.create_task(scheduled[0]())
        await asyncio.sleep(.04)
        await button(page.controls, "更新中…").on_click(None)  # racing manual request is ignored
        await auto

    asyncio.run(run())
    assert len(calls) == 1 and calls[0][1]["max_attempts"] == 1
    assert calls[0][1]["time_budget_seconds"] == 90
    assert any("2330 更新行情" in frame for frame in frames)
    assert not button(page.controls, "繼續更新未完成資料").disabled
    assert "待續抓 1" in texts(page.controls) and "仍有資料未達更新目標" in texts(page.controls)
    assert not page.controls[0].controls[2].visible  # shared progress above all tabs


def test_sox_failure_is_visible_and_controls_recover(monkeypatch):
    from core import market_regime
    page = build_app(monkeypatch)
    monkeypatch.setattr(app, "update_symbols", lambda *a, **kw: {"failed": 0, "stale": 0})
    monkeypatch.setattr(app, "format_update_status", lambda r: "行情已更新")
    monkeypatch.setattr(market_regime, "refresh_sox", lambda **kw: (_ for _ in ()).throw(RuntimeError("source timeout")))
    asyncio.run(button(page.controls, "更新每日資料").on_click(None))
    assert "費半更新失敗：source timeout" in texts(page.controls)
    assert not button(page.controls, "重試未完成資料").disabled
    assert not button(page.controls, "分析這檔").disabled


def test_startup_failure_is_not_silenced(monkeypatch):
    scheduled = []
    monkeypatch.delenv("QUANT_APP_OFFLINE", raising=False)
    monkeypatch.setattr(FakePage, "run_task", lambda p, task: scheduled.append(task), raising=False)
    page = build_app(monkeypatch)
    monkeypatch.setattr(app, "update_symbols", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("cannot reach source")))
    asyncio.run(scheduled[0]())
    assert "cannot reach source" in texts(page.controls)
    assert not button(page.controls, "重試未完成資料").disabled
    assert not any(item.visible for item in walk(page.controls) if isinstance(item, ft.ProgressBar))


@pytest.mark.parametrize("viewport_width", [390, 420])
def test_mobile_fixed_width_input_rows_wrap_when_space_is_insufficient(monkeypatch, viewport_width):
    page = build_app(monkeypatch)
    # Cards have page padding 16 and inner padding 12 on either side.
    content_width = viewport_width - 2 * (16 + 12)
    input_rows = [row for row in walk(page.controls) if isinstance(row, ft.Row)
                  and any(isinstance(child, (ft.TextField, ft.Dropdown)) for child in row.controls)]
    assert input_rows
    for row in input_rows:
        if row.wrap or any(child.expand for child in row.controls):
            continue
        if all(child.width is not None for child in row.controls):
            required = sum(child.width for child in row.controls) + row.spacing * (len(row.controls) - 1)
            assert required <= content_width, f"Input row requires {required}px at {viewport_width}px viewport"


def test_monthly_wrapper_always_uses_active_model_even_if_config_changes(monkeypatch):
    seen = {}
    monkeypatch.setattr(app.rotation, "run_rotation", lambda **kwargs: seen.update(kwargs) or {})
    monkeypatch.setattr(app.config, "ROTATION_SCORE_MODE", "production", raising=False)
    app.monthly_holdings()
    assert seen == {"defensive": False, "score_mode": app.ACTIVE_MODEL_ID}
