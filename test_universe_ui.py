import config
from main import _universe_text


def test_pool_label_distinguishes_candidates_from_holdings(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", [f"{1000+i}" for i in range(150)])
    monkeypatch.setattr(config, "UNIVERSE_KIND", "market_cap", raising=False)
    monkeypatch.setattr(config, "UNIVERSE_ASOF", "2026-10-02", raising=False)
    monkeypatch.setattr(config, "UNIVERSE_MARKETS", {f"{1000+i}": "twse" if i < 140 else "tpex" for i in range(150)}, raising=False)
    label = _universe_text()
    assert "150" in label and "上市 140／上櫃 10" in label
    assert "2026-10-02" in label and "252" in label
    assert "最多持有 8 檔" in label and "固定快照" in label
