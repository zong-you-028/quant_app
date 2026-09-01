from core import data_pipeline as dp


def test_update_symbols_reports_errors_and_does_not_throttle_total_failure(monkeypatch):
    saved = []
    monkeypatch.setattr(dp, "needs_update", lambda symbol: True)
    monkeypatch.setattr(dp, "fetch_real_data",
                        lambda symbol: (_ for _ in ()).throw(RuntimeError("API limit")))
    monkeypatch.setattr(dp, "_meta_set", lambda key, value: saved.append((key, value)))

    result = dp.update_symbols(["2330", "2317"], ignore_throttle=True)

    assert result["failed"] == 2
    assert result["failed_symbols"] == ["2330", "2317"]
    assert result["errors"]["2330"] == "API limit"
    assert saved == []


def test_update_symbols_records_successful_refresh(monkeypatch):
    saved = []
    monkeypatch.setattr(dp, "needs_update", lambda symbol: True)
    monkeypatch.setattr(dp, "fetch_real_data", lambda symbol: None)
    monkeypatch.setattr(dp, "_meta_set", lambda key, value: saved.append((key, value)))

    result = dp.update_symbols(["2330"], ignore_throttle=True)

    assert result["updated"] == 1
    assert result["failed"] == 0
    assert saved and saved[0][0] == "last_refresh"


def test_finmind_token_uses_authorization_header(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        reason = "OK"

        def json(self):
            return {"status": 200, "data": [{"date": "2026-08-31"}]}

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr(dp.config, "FINMIND_TOKEN", "secret-token")
    monkeypatch.setattr(dp.requests, "get", fake_get)

    result = dp._finmind_get("TaiwanStockPrice", "2330", "2026-08-01")

    assert not result.empty
    assert "token" not in captured["params"]
    assert captured["headers"]["Authorization"] == "Bearer secret-token"


def test_finmind_does_not_retry_403(monkeypatch):
    calls = []
    monkeypatch.setattr(dp, "_FINMIND_BLOCKED_UNTIL", None)

    class Response:
        status_code = 403
        reason = "Forbidden"

        def json(self):
            return {"status": 403, "msg": "ip banned"}

    monkeypatch.setattr(dp.requests, "get",
                        lambda *args, **kwargs: calls.append(1) or Response())

    try:
        dp._finmind_get("TaiwanStockPrice", "2330", "2026-08-01")
    except RuntimeError as ex:
        assert "HTTP 403" in str(ex)
    else:
        raise AssertionError("403 should raise")
    assert len(calls) == 1


def test_fetch_real_data_uses_twse_when_finmind_ip_is_blocked(monkeypatch):
    used = []
    monkeypatch.setattr(dp, "init_db", lambda: None)
    monkeypatch.setattr(
        dp, "_finmind_get",
        lambda *args, **kwargs: (_ for _ in ()).throw(dp.FinMindBlockedError("ip banned")),
    )
    monkeypatch.setattr(dp, "fetch_twse_recent_data", lambda symbol: used.append(symbol))

    dp.fetch_real_data("2330")

    assert used == ["2330"]
