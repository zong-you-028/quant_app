# Comparison version and reproduction

The nine cost × execution-delay comparisons in `results.json` were completed before the app's 150-stock integration. The configuration at comparison time is the original fixed-50 configuration from commit `f49b4cae3c0f5c55acfd609aae5ce7abf6e672a0`. The report records that comparison's input and source hashes, not the subsequent app integration. No result, statistic, input hash or source hash has been rewritten after the run; no additional nine-scenario replay was performed for these notes.

The statements `app_config_changed: false` and `strategy_parameters_changed: false` describe the comparison run. They do not assert that the surrounding workspace remained unchanged after that run.

## Exact archived source

`config_at_comparison.py` was copied byte for byte from `research/paper_forward_2026_10_03/pinned_runtime/source/config.py`. Its SHA-256 matches the report:

`08d906895ee2616d1fdb5ead332c4929a4afd4b62b18844736269db34dd21e7c`

`strategy_models_at_comparison.py` has the exact SHA-256 recorded in the report:

`11d1beae60bf0dc97d669c92beb5135d8c48035c5ac329efd080d3220dcc6811`

The original pinned model predates a metadata-only conditional status label for a market-cap universe. The comparison already included that label, `市值150池試行・過擬合未排除・待前瞻驗證`. Subsequent integration changed it to `估算市值150池試行・過擬合未排除・待前瞻驗證`. The archived comparison model was reconstructed by reversing only those two added label characters in the current model and requiring its full file SHA-256 to match the recorded hash before saving it. No ranking formula, warm-up rule, portfolio rule or cost parameter changed in this reconstruction.

At this version-note check, only `config.py` and `core/strategy_models.py` differed from the report's nine recorded code fingerprints. The comparison script, rotation, robustness replay, execution, chip reader, SOX reader and bootstrap audit still matched their recorded hashes. All four recorded input fingerprints (research DB, SOX CSV, ranking JSON and acquisition manifest) also matched.

| File | Comparison / archived SHA-256 | Later integration SHA-256 observed while writing these notes |
| --- | --- | --- |
| config.py | 08d906895ee2616d1fdb5ead332c4929a4afd4b62b18844736269db34dd21e7c | ff266739077fbb89701f18b2e5e7d247538b8e26453cb5a2b5e2354fc1f3296c |
| core/strategy_models.py | 11d1beae60bf0dc97d669c92beb5135d8c48035c5ac329efd080d3220dcc6811 | 149f62af74597e37cb63c8b8264cd9ce07ffb80fd2baf003782b19a6387a43d3 |

These later integration hashes are descriptive observations, not replacement report fingerprints or a frozen final app version. The original pinned model has SHA-256 `afe918f1e558cc3290d2cfc0eed9aea91d6aa790577822e1e0eb4d4ebfbd14c5`, which differs because it lacks the conditional status metadata.

## Data pipeline distinction

`core/data_pipeline.py` was not included in this comparison's `code_sha256`. The actual price and chip data used by the comparison came from the explicitly supplied independent public research snapshot and the replay's patched loaders. The comparison did not exercise the current app's HTTP acquisition or automatic updates and is not evidence that those operations reproduce this research input.

The pinned pipeline has SHA-256 `5d609eb95486944d20947c512a32ed058ea621d9e09cd833a67e1abfae2c52ad`; the integration pipeline observed while writing these notes has SHA-256 `3d9f8aa2933e2a1c88e16991c45a5010275131a0530700fc0d69ebcdd27443d3`. The integration differences include the v4 150-stock public seed, filtering reads and full-history acquisition at the current-market listing date, TPEx market routing and latest-price support, and clearing TPEx cache before retry. These acquisition changes require separate integration verification. This note does not retroactively claim the pipeline fingerprint was recorded in `results.json`.

## Reproduce without changing the app

Do not copy the archived configuration over the production workspace. To reproduce this saved research version, use an isolated temporary code directory, the archived configuration and model, and the original pinned data pipeline. Compare its source hashes to `results.json` before execution. Pass the independent research inputs by absolute path, and write replay results to a new directory instead of overwriting the saved comparison. The current app's later public seed is a separate integration input.

The following Python example is a reproduction recipe, not a command already rerun for this note. It copies code only, opens the supplied research DB through the comparison's existing read-only/public-snapshot isolation, blocks research HTTP, and leaves the original result and formal DB untouched. Use the same Python environment/dependencies as the completed comparison.

```python
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(r"D:\quant_app")
evidence = repo / "research/universe_150_2026_10_03/comparison"
record = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
for source in record["sources"].values():
    assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]

with tempfile.TemporaryDirectory(prefix="universe-comparison-code-") as folder:
    isolated = Path(folder)
    shutil.copytree(repo / "core", isolated / "core")
    shutil.copyfile(evidence / "config_at_comparison.py", isolated / "config.py")
    shutil.copyfile(evidence / "strategy_models_at_comparison.py", isolated / "core/strategy_models.py")
    shutil.copyfile(evidence / "rotation_at_comparison.py", isolated / "core/rotation.py")
    pinned_pipeline = repo / "research/paper_forward_2026_10_03/pinned_runtime/source/core/data_pipeline.py"
    assert hashlib.sha256(pinned_pipeline.read_bytes()).hexdigest() == "5d609eb95486944d20947c512a32ed058ea621d9e09cd833a67e1abfae2c52ad"
    shutil.copyfile(pinned_pipeline, isolated / "core/data_pipeline.py")
    script = isolated / "research/universe_150_2026_10_03/compare_universes.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(repo / "research/universe_150_2026_10_03/compare_universes.py", script)
    for name, expected in record["code_sha256"].items():
        assert hashlib.sha256((isolated / name).read_bytes()).hexdigest() == expected
    subprocess.run([
        sys.executable, str(script),
        "--db", record["sources"]["research_database"]["path"],
        "--sox", record["sources"]["sox"]["path"],
        "--universe", record["sources"]["ranking"]["path"],
        "--output", str(evidence / "replay_local"),
    ], cwd=isolated, check=True)
```

The output's code fingerprint paths refer to the isolated code directory. Numerical equality should be checked against the saved paired daily-return and turnover series, not inferred from a similar CAGR alone. The listing-floor data filter was performed in the acquisition manifest before comparison; it is not reimplemented by silently filling or substituting missing history during replay.

## Interpretation remains unchanged

After the version-note check above, app integration separated quote freshness from ranking warm-up in `core/rotation.py`. A newly listed stock can have current quotes while remaining ineligible for momentum ranking; it must not be falsely reported as stale. `rotation_at_comparison.py` preserves the exact pre-fix rotation fingerprint recorded in `results.json`. Use this archived file in the isolated recipe above. The historical nine-scenario results were not rerun or rewritten. A separate actual-app integration run verified the same normal-cost daily period and CAGR after this data-quality fix.

The declaration contains 150 current, same-day priced ordinary shares selected by estimated capitalization; it is not an official full-market historical constituent list. Seven classified ordinary shares lacked an effective same-day close and were explicitly excluded from the ranking scope. Current issued capital × current close, present-day membership and survivor selection can exaggerate historical results. The nine tests and paired bootstrap intervals are retrospective and pointwise; they do not correct this pool selection or replace the earlier 11-model selection diagnostic.

The added-stock acquisition used a conservative price floor at each stock's current-market ISIN listing date. It excluded emerging-board history but can also discard legitimate earlier OTC history for transferred listings. It is not complete point-in-time eligibility evidence. The original 50-stock research seed was preserved for the fair fixed-50 baseline. As-of coverage and daily rank eligibility are retained in the saved outputs; short genuine histories remain in the declared 150 pool and are not imputed into 252-day eligibility.

The unchanged model targets at most eight stocks; rank 16 is the retention buffer. The primary 150-stock historical CAGR is higher, but all nine scenarios have worse maximum drawdown and higher turnover than the fixed-50 comparison. The 2022 and partial-2026 paired return differences are negative. These results support an explicitly labelled research/paper trial, not a claim of guaranteed future improvement or completed overfitting clearance.
