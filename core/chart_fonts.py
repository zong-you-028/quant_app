"""Use the shipped Traditional Chinese font on both Render and Windows."""
from pathlib import Path

import matplotlib
from matplotlib import font_manager


FONT_PATH = Path(__file__).resolve().parents[1] / "resources/fonts/NotoSansCJKtc-Regular.otf"


def configure_chart_fonts():
    # addfont is required in every process; Matplotlib does not persist it in
    # the font cache. The font ships in the image, so startup needs no download.
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    matplotlib.rcParams["font.family"] = ["sans-serif"]
    matplotlib.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return family
