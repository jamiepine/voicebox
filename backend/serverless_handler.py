"""
RunPod Serverless handler for voicebox TTS server.

Starts the FastAPI/uvicorn server in a background thread and proxies
RunPod job requests as HTTP calls to the local server.

Usage (RunPod serverless):
    CMD ["python3", "-u", "-m", "backend.serverless_handler"]

Local testing:
    python3 -m backend.serverless_handler --rp_serve_api
"""

import os
import sys
import time
import logging
import threading
import base64

import httpx
import runpod
import uvicorn

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
_HOST = "127.0.0.1"
_PORT = 17493
_BASE_URL = f"http://{_HOST}:{_PORT}"
_STARTUP_TIMEOUT = 300  # 5 min max for cold start model downloads
_STARTUP_POLL = 2  # seconds between health checks

# ── Server lifecycle ──────────────────────────────────────────
_server_ready = threading.Event()
_server_thread: threading.Thread | None = None


def _start_server():
    """Start the FastAPI/uvicorn server in a background thread."""
    global _server_thread

    if _server_thread is not None and _server_thread.is_alive():
        return

    _server_ready.clear()

    # Set SERVERLESS before importing the app so backends disable idle timers
    os.environ["SERVERLESS"] = "1"

    from backend import config, database
    from backend.main import app

    config.set_data_dir("/app/data")

    def _run():
        uvicorn.run(app, host=_HOST, port=_PORT, log_level="info")

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()


def _wait_for_server():
    """Block until /health responds 200 or timeout."""
    if _server_ready.is_set():
        return

    deadline = time.time() + _STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            r = httpx.get(f"{_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                logger.info("Voicebox server is ready")
                _server_ready.set()
                return
        except httpx.RequestError:
            pass
        time.sleep(_STARTUP_POLL)

    raise RuntimeError(
        f"Voicebox server did not become healthy within {_STARTUP_TIMEOUT}s"
    )


# ── RunPod handler ────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    RunPod serverless handler.

    Expected job["input"]:
        method  (str)  — HTTP method, default "POST"
        path    (str)  — required, e.g. "/generate"
        body    (dict) — optional, JSON body for POST/PUT
        params  (dict) — optional, query params
        headers (dict) — optional
    """
    _start_server()
    _wait_for_server()

    inp = job.get("input", {})

    path = inp.get("path")
    if not path:
        return {"error": "Missing 'path' in job input"}

    method = inp.get("method", "POST").upper()
    body = inp.get("body")
    params = inp.get("params")
    headers = inp.get("headers", {})

    url = f"{_BASE_URL}{path}"

    try:
        with httpx.Client(timeout=600) as client:
            response = client.request(
                method=method,
                url=url,
                json=body if method in ("POST", "PUT", "PATCH") else None,
                params=params,
                headers=headers,
            )

        content_type = response.headers.get("content-type", "")
        is_binary = (
            "audio/" in content_type
            or "application/octet-stream" in content_type
            or "application/zip" in content_type
        )

        if is_binary:
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body_base64": base64.b64encode(response.content).decode("ascii"),
                "is_binary": True,
            }

        try:
            result = response.json()
        except Exception:
            result = response.text

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": result,
        }

    except httpx.TimeoutException:
        return {"error": "Request to voicebox server timed out (600s)"}
    except Exception as e:
        return {"error": f"Request failed: {e}"}


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
