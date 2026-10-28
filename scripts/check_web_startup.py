"""Smoke-test the real Render entry point with disposable data and no login."""
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="quant-web-startup-") as folder:
        # Select an unused local port. Never bind this build check publicly.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = dict(os.environ)
        env.update(
            FLET_FORCE_WEB_SERVER="true", FLET_SERVER_IP="127.0.0.1",
            FLET_SERVER_PORT=str(port), APP_DATA_DIR=folder,
            APP_PASSWORD="", FINMIND_TOKEN="", JOURNAL_DATABASE_URL="",
            MARKET_DATABASE_URL="", PYTHONUNBUFFERED="1",
            MPLCONFIGDIR=str(Path(folder) / "matplotlib"),
        )
        log_path = Path(folder) / "startup.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "main.py")], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Web server exited with status {process.returncode}")
                    try:
                        with opener.open(f"http://127.0.0.1:{port}/", timeout=2) as response:
                            assert response.status == 200
                            assert b"flutter" in response.read().lower()
                        break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Web server did not become ready within 60 seconds")
                        time.sleep(0.2)
                with connect(f"ws://127.0.0.1:{port}/ws", open_timeout=5, proxy=None):
                    pass
                assert process.poll() is None, "Server died after WebSocket connection"
            except Exception:
                log.flush()
                print(log_path.read_text(encoding="utf-8"), file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        output = log_path.read_text(encoding="utf-8")
        assert "Installing flet" not in output, "Web packages must already be installed"
        assert "Traceback" not in output, output
        print(json.dumps({
            "http_status": 200, "websocket_connected": True,
            "runtime_package_install": False, "login_submitted": False,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("flet", "flet-web", "flet-charts", "fastapi",
                          "starlette", "uvicorn", "certifi")},
        }))


if __name__ == "__main__":
    main()
