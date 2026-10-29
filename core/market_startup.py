"""Restore durable public data before any authenticated screen reads SQLite."""
import json
import threading

from core import daily_market_update as daily
from core import market_cache_storage as storage


_LOCK = threading.Lock()
_READY = {}


def prepare_saved_market_data():
    """One cold-start restore; later sessions reuse the updated local cache.

    The optional SOX catch-up touches public index data only. A failed source
    remains visible and retryable; it cannot prevent saved stocks from loading.
    """
    if not storage.configured():
        return {"configured": False, "durable": False}
    key = daily._cache_key()
    with _LOCK:
        if key in _READY:
            return dict(_READY[key])
        restored = daily.restore_cache()
        outcome = {"configured": True, "durable": bool(restored.get("durable")),
                   "restored": restored.get("restored", 0), "sox_error": None}
        if not outcome["durable"]:
            outcome["error"] = restored.get("error") or "雲端行情尚未還原完成"
            print(json.dumps({"event": "market_cache_startup", "ready": False}), flush=True)
            return outcome
        try:
            series = daily.refresh_sox(raise_errors=True)
            outcome["sox_asof"] = str(series.index.max().date())
        except Exception:
            outcome["sox_error"] = "費半來源暫時無法更新；已保存台股仍可直接載入"
        # The dates come from restored storage, not the bundled universe date.
        from core.strategy_models import data_quality
        quality = data_quality()
        outcome["asof"] = quality.get("asof")
        outcome["oldest_asof"] = quality.get("oldest_asof")
        _READY[key] = dict(outcome)
        print(json.dumps({"event": "market_cache_startup", "ready": True,
                          "restored": outcome["restored"], "asof": outcome["asof"],
                          "oldest_asof": outcome["oldest_asof"],
                          "sox_asof": outcome.get("sox_asof"),
                          "sox_pending": bool(outcome["sox_error"])}), flush=True)
        return outcome
