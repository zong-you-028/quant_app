# Chart font

`NotoSansCJKtc-Regular.otf` is the unmodified Traditional Chinese regular face
from [Noto CJK](https://github.com/notofonts/noto-cjk). Its exact source revision
and SHA-256 are recorded in `source.json`; redistribution is covered by the
included SIL Open Font License in `LICENSE`.

The app registers this bundled font with Matplotlib at process startup. PNG
charts on Render therefore display Chinese titles, legends and stock names
without relying on Windows fonts or downloading fonts at runtime. The Docker
build also verifies the font and the asset-chart Chinese glyphs on Linux.
