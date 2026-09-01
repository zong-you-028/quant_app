FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Bootstrap Render with the same real market/name cache used locally.
# The seed contains no journal tables or personal trading records.
RUN mkdir -p data && python -c "import gzip, shutil; src=gzip.open('data_seed/market.db.gz','rb'); dst=open('data/market.db','wb'); shutil.copyfileobj(src,dst); src.close(); dst.close()"

# Render supplies port 10000 by default. Flet reads these on startup and
# exposes this as a browser-based WebSocket application.
ENV FLET_FORCE_WEB_SERVER=true \
    FLET_SERVER_IP=0.0.0.0 \
    FLET_SERVER_PORT=10000

EXPOSE 10000

CMD ["python", "main.py"]
