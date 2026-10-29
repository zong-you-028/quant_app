"""First screen and repeat visits use restored quotes before any daily update."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import flet as ft
import pandas as pd
import pytest

import main as app
from core import market_startup as startup, strategy_models as models
from test_market_cache_storage import state, put_stock, bundle, maximum


def install_cloud(state, monkeypatch):
    monkeypatch.setattr(startup.daily.dp.config, "DB_PATH", str(state.path))
    put_stock(state, day="2026-10-02", chip="2026-10-02")
    state.cloud.save_bundle("stock:2330", bundle(day="2026-10-15"))
    monkeypatch.setattr(models, "data_quality", lambda: {
        "asof": maximum(state), "oldest_asof": maximum(state)})
    calls = []
    monkeypatch.setattr(startup.daily, "refresh_sox", lambda **kw:
                        calls.append("SOX") or pd.Series([5000.], index=pd.to_datetime(["2026-10-15"])))
    return calls


def test_cold_cache_restored_once_and_concurrent_repeat_visits_do_not_download_stocks(state, monkeypatch):
    calls = install_cloud(state, monkeypatch)
    monkeypatch.setattr(startup.daily, "update_symbols", lambda *a, **k: pytest.fail("restoration must not download stocks"))
    with ThreadPoolExecutor(max_workers=4) as workers:
        outcomes = list(workers.map(lambda _: startup.prepare_saved_market_data(), range(4)))
    assert all(result["durable"] and result["asof"] == "2026-10-15" for result in outcomes)
    assert maximum(state) == "2026-10-15" and calls == ["SOX"]


def test_sox_failure_cannot_hide_saved_stock_cache(state, monkeypatch):
    install_cloud(state, monkeypatch)
    def unavailable(**kwargs):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(startup.daily, "refresh_sox", unavailable)
    outcome = startup.prepare_saved_market_data()
    assert outcome["durable"] and outcome["asof"] == "2026-10-15"
    assert outcome["sox_error"] and maximum(state) == "2026-10-15"


class Page:
    def __init__(self):
        self.controls = []
        self.frames = []
    def add(self, *controls):
        self.controls.extend(controls)
    def update(self):
        self.frames.append(list(self.controls))
    def clean(self):
        self.controls.clear()


def test_first_authenticated_screen_reads_cloud_date_without_manual_update(state, monkeypatch):
    install_cloud(state, monkeypatch)
    dates = []
    monkeypatch.setattr(app, "_build_app", lambda page, **kwargs: dates.append(maximum(state)))
    page = Page()
    asyncio.run(app._open_saved_app(page))
    assert dates == ["2026-10-15"]
    assert any(isinstance(item, ft.Column) for frame in page.frames for item in frame)
    put_stock(state, day="2026-10-16")
    asyncio.run(app._open_saved_app(Page()))
    assert dates == ["2026-10-15", "2026-10-16"]


def test_restore_failure_does_not_present_oct2_seed_as_loaded_data(monkeypatch):
    monkeypatch.setattr(startup, "prepare_saved_market_data", lambda: {
        "configured": True, "durable": False, "error": "cloud unavailable"})
    monkeypatch.setattr(app, "_build_app", lambda *a, **k: pytest.fail("must not show seed app"))
    page = Page()
    asyncio.run(app._open_saved_app(page))
    column = page.controls[0]
    assert "cloud unavailable" in column.controls[0].value
    assert column.controls[1].content == "重試載入已保存行情"


def test_universe_snapshot_label_is_separate_from_market_date(monkeypatch):
    monkeypatch.setattr(app.config, "UNIVERSE_KIND", "market_cap")
    monkeypatch.setattr(app.config, "UNIVERSE_ASOF", "2026-10-02")
    assert "成分名單快照 2026-10-02（非行情日期）" in app._universe_text()


def test_yahoo_rate_limit_remains_explicit_and_does_not_modify_cache(state, monkeypatch):
    from core import daily_market_update as daily, market_regime as mr
    monkeypatch.setattr(mr, "_cached_sox", lambda *args: None)
    class YFRateLimitError(Exception):
        pass
    def unavailable(**kwargs):
        raise YFRateLimitError("source rate limit")
    monkeypatch.setitem(__import__("sys").modules, "yfinance", SimpleNamespace(
        Ticker=lambda symbol: SimpleNamespace(history=unavailable)))
    with pytest.raises(RuntimeError, match="費半來源暫時限流"):
        daily._download_sox()
