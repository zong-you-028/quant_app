"""Continue the original 50-stock experiment with its pinned f49b4ca code.

The app may now use another universe. This compatibility command still records
only old50, in its independent cache and archive. The existing authorized
default refreshes public market data; --offline disables all network access.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.paper_forward_2026_10_03.pinned_runtime.capture_latest import run as pinned_run


def run(offline=False):
    return pinned_run(offline=offline, refresh=not offline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    print(json.dumps(run(parser.parse_args().offline), ensure_ascii=False, indent=2))
