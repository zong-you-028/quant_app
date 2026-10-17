"""Render synthetic asset charts without any operating-system fonts or DB."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    folder = parser.parse_args().output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.update(APP_DATA_DIR=str(folder / "unused_data"),
                      JOURNAL_DATABASE_URL="", MARKET_DATABASE_URL="", FINMIND_TOKEN="")
    counters = {"database_attempts": 0, "http_attempts": 0}

    def no_database(*args, **kwargs):
        counters["database_attempts"] += 1
        raise AssertionError("Chart QA must not access any database")

    def no_http(*args, **kwargs):
        counters["http_attempts"] += 1
        raise AssertionError("Chart QA must not access HTTP")

    sqlite3.connect = sqlite3.dbapi2.connect = no_database
    import requests
    requests.sessions.Session.request = no_http
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    # Rebuild with only Matplotlib's bundled fonts, like a minimal Linux image.
    font_manager.findSystemFonts = lambda *args, **kwargs: []
    # Keep the singleton: Agg and text modules retain references to it.
    bundled_fonts = Path(matplotlib.get_data_path()).resolve()
    font_manager.fontManager.ttflist = [item for item in font_manager.fontManager.ttflist
                                       if Path(item.fname).resolve().is_relative_to(bundled_fonts)]
    font_manager.fontManager.afmlist = []
    assert any(item.name == "DejaVu Sans" for item in font_manager.fontManager.ttflist)
    assert not any("CJK" in item.name or "JhengHei" in item.name
                   for item in font_manager.fontManager.ttflist)
    import main as app
    from core.chart_fonts import FONT_PATH

    history = [{"ts": f"2026-10-{day:02d} 18:00:00", "invested": 1200000,
                "total_assets": 1200000 + day * 20000 - (30000 if day == 4 else 0)}
               for day in range(1, 8)]
    with warnings.catch_warnings(record=True) as before:
        warnings.simplefilter("always")
        with matplotlib.rc_context({"font.sans-serif": ["DejaVu Sans"]}):
            figure = app.make_asset_figure(history)
            FigureCanvasAgg(figure).print_png(str(folder / "before.png"))
    missing_before = sum("Glyph" in str(item.message) and "missing" in str(item.message)
                         for item in before)
    assert missing_before > 0, "Must reproduce the missing-Chinese-font failure"
    cases = []
    for name, rows in (("single", history[:1]), ("multiple", history)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            figure = app.make_asset_figure(rows)
            canvas = FigureCanvasAgg(figure)
            canvas.print_png(str(folder / f"{name}.png"))
            image = app.asset_history_image(rows)
        missing = [str(item.message) for item in caught
                   if "Glyph" in str(item.message) and "missing" in str(item.message)]
        assert not missing, {"missing": len(missing), "family": matplotlib.rcParams["font.sans-serif"],
                             "resolved": font_manager.findfont(figure.axes[0].title.get_fontproperties())}
        axis = figure.axes[0]
        if len(rows) == 1:
            assert [label.get_text() for label in axis.get_xticklabels()] == ["2026-10-01"]
        resolved = Path(font_manager.findfont(axis.title.get_fontproperties(), fallback_to_default=False))
        assert resolved.resolve() == FONT_PATH.resolve()
        assert image.src and len(image.src) > 1000
        bounds = figure.bbox
        for artist in (axis.title, axis.get_legend()):
            box = artist.get_window_extent(canvas.get_renderer())
            assert bounds.x0 <= box.x0 <= box.x1 <= bounds.x1
            assert bounds.y0 <= box.y0 <= box.y1 <= bounds.y1
        cases.append({"name": name, "points": len(rows), "missing_glyph_warnings": 0,
                      "title_and_legend_inside_image": True, "flet_image_created": True,
                      "date_labels": [label.get_text() for label in axis.get_xticklabels()]})
    assert not any(counters.values()), counters
    report = {"passed": True, "operating_system_fonts_disabled": True,
              "font_family": "Noto Sans CJK TC", "baseline_missing_glyph_warnings": missing_before,
              "cases": cases, **counters}
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
