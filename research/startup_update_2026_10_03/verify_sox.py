"""Live SOX incremental download on an isolated, deliberately old public cache."""
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
with tempfile.TemporaryDirectory(prefix="quant_sox_probe_") as directory:
    os.environ["APP_DATA_DIR"] = directory
    from core import market_regime as mr
    seed = pd.read_csv(ROOT / "data_seed/sox.csv", index_col=0)
    seed.loc[seed.index <= "2026-09-28"].to_csv(mr.SOX_CSV)
    before = Path(mr.SOX_CSV).read_bytes()
    try:
        result = mr.refresh_sox(raise_errors=True)
        report = {"success": True, "start": str(result.index.min().date()),
                  "latest": str(result.index.max().date()), "rows": len(result)}
    except Exception as exc:
        report = {"success": False, "error": str(exc),
                  "old_cache_preserved": Path(mr.SOX_CSV).read_bytes() == before}
    Path(__file__).with_name("live_sox_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
