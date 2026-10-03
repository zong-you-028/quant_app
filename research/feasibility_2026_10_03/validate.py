"""Verify phase coverage, paired summaries, and original phase-zero parity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.strategy_feasibility import ORIGINAL_NAMES, summarize_phases

HERE = Path(__file__).resolve().parent
result = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
original = json.loads((HERE.parent / "strategy_models_2026_10_03" / "results.json").read_text(encoding="utf-8"))
frame = pd.DataFrame(result["phase_rows"])
assert result["data"]["snapshot_sha256"] == original["data"]["snapshot_sha256"]
assert result["settings"]["cost"] == .003 and result["settings"]["lag"] == 2
assert set(frame.candidate) == set(ORIGINAL_NAMES)
assert len(frame) == len(ORIGINAL_NAMES) * 20 * 3
for (_, _), part in frame.groupby(["candidate", "segment"]):
    assert set(part.phase) == set(range(20))
    assert np.isfinite(part[["cagr", "sharpe", "mdd", "annual_turnover"]].to_numpy()).all()
for name in ORIGINAL_NAMES:
    for segment, key in (("development", "development"), ("final_diagnostic", "holdout")):
        row = frame[(frame.candidate == name) & (frame.phase == 0) & (frame.segment == segment)].iloc[0]
        for metric in ("cagr", "sharpe", "mdd", "annual_turnover"):
            assert abs(row[metric] - original[key][name][metric]) < 1e-12, (name, segment, metric)
assert summarize_phases(result["phase_rows"]) == result["summary"]
selected = max((name for name, item in result["summary"]["development"].items()
                if item["development_screen_pass"]),
               key=lambda name: result["summary"]["development"][name]["sharpe_median"])
assert selected == result["selected_on_development"]
assert result["replace_default"] is False and result["production_changed"] is False
paths = pd.read_csv(HERE / "daily_returns.csv", index_col="date")
assert paths.shape[1] == len(ORIGINAL_NAMES) * 20
assert not paths.isna().any().any()
assert paths.index[0] == result["development_start"] and paths.index[-1] == result["final_end"]
print("PASS: 100 replay paths, all 20 phases, phase-zero parity, paired summaries, development-only selection, same cached snapshot.")
