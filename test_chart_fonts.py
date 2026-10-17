"""Chinese chart labels work without the developer's Windows fonts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def test_shipped_font_matches_pinned_upstream_and_includes_license():
    folder = ROOT / "resources/fonts"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8-sig"))
    assert hashlib.sha256((folder / "NotoSansCJKtc-Regular.otf").read_bytes()).hexdigest() == source["sha256"]
    assert "SIL OPEN FONT LICENSE Version 1.1" in (folder / "LICENSE").read_text(encoding="utf-8")


def test_asset_png_and_flet_image_render_without_any_system_chinese_font(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "research/chart_text_2026_10_17/verify_chart.py"),
                             "--output", str(tmp_path)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["passed"] and report["operating_system_fonts_disabled"]
    assert report["baseline_missing_glyph_warnings"] > 0
    assert report["database_attempts"] == report["http_attempts"] == 0
    assert [case["points"] for case in report["cases"]] == [1, 7]
