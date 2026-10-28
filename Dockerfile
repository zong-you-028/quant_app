FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt && pip check

COPY . ./

# Verify the bundled Chinese font loads in the actual Linux build as well.
RUN python -c "from core.chart_fonts import configure_chart_fonts, FONT_PATH; from matplotlib.ft2font import FT2Font; configure_chart_fonts(); chars=FT2Font(str(FONT_PATH)).get_charmap(); assert all(ord(c) in chars for c in '總資產成長曲線累計投入0123456789')"

# Bootstrap Render with the same real market/name cache used locally.
# The seed contains no journal tables or personal trading records.
RUN mkdir -p data && python -c "import gzip, shutil; src=gzip.open('data_seed/market.db.gz','rb'); dst=open('data/market.db','wb'); shutil.copyfileobj(src,dst); src.close(); dst.close(); shutil.copyfile('data_seed/sox.csv','data/sox.csv')"

# Exercise the real HTTP/WebSocket startup, including Flet's FastAPI wrapper.
# An incompatible web dependency must fail the build, not a production restart.
RUN python scripts/check_web_startup.py

# Render supplies port 10000 by default. Flet reads these on startup and
# exposes this as a browser-based WebSocket application.
ENV FLET_FORCE_WEB_SERVER=true \
    FLET_SERVER_IP=0.0.0.0 \
    FLET_SERVER_PORT=10000

EXPOSE 10000

CMD ["python", "main.py"]
