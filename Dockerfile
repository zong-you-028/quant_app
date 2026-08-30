FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Render supplies port 10000 by default. Flet reads these on startup and
# exposes this as a browser-based WebSocket application.
ENV FLET_FORCE_WEB_SERVER=true \
    FLET_SERVER_IP=0.0.0.0 \
    FLET_SERVER_PORT=10000

EXPOSE 10000

CMD ["python", "main.py"]
