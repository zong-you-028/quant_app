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
