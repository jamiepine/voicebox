"""Voicebox Cloud device login routes.

The browser-based pairing flow:
  1. POST /cloud/login/start  — opens the browser to the cloud authorize page.
  2. GET  /cloud/callback     — the browser lands here with a one-time code;
                                the backend exchanges it for an API key.
  3. GET  /cloud/status       — the UI polls this to learn when it connected.
  4. POST /cloud/disconnect   — forget the local credential.
"""

import socket

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..services import cloud as cloud_service

router = APIRouter(prefix="/cloud", tags=["cloud"])


def _callback_url(request: Request) -> str:
    # Always loopback — the cloud only redirects codes to 127.0.0.1/localhost.
    port = request.url.port or 17493
    return f"http://127.0.0.1:{port}/cloud/callback"


@router.post("/login/start", response_model=models.CloudLoginStartResponse)
async def start_cloud_login(request: Request):
    device_name = socket.gethostname() or "Desktop"
    authorize_url = cloud_service.start_login(_callback_url(request), device_name)
    return models.CloudLoginStartResponse(authorize_url=authorize_url)


@router.get("/callback", response_class=HTMLResponse)
async def cloud_callback(
    request: Request,
    code: str = "",
    state: str = "",
    db: Session = Depends(get_db),
):
    ok, message = await cloud_service.handle_callback(db, code=code, state=state)
    heading = "You're connected" if ok else "Couldn't connect"
    accent = "#16a34a" if ok else "#dc2626"
    sub = (
        "Voicebox is now linked to your account. You can close this tab and return to the app."
        if ok
        else message
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Voicebox Cloud</title>
<style>
  body {{ margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background:#0b0b0d; color:#e7e7ea; }}
  .card {{ max-width:28rem; padding:2.5rem; text-align:center; }}
  h1 {{ font-size:1.5rem; margin:0 0 .5rem; color:{accent}; }}
  p {{ color:#a1a1aa; line-height:1.5; }}
</style></head>
<body><div class="card"><h1>{heading}</h1><p>{sub}</p></div></body></html>"""
    return HTMLResponse(content=html, status_code=200 if ok else 400)


@router.get("/status", response_model=models.CloudStatusResponse)
async def cloud_status(db: Session = Depends(get_db)):
    return models.CloudStatusResponse(**cloud_service.get_status(db))


@router.post("/disconnect", response_model=models.CloudStatusResponse)
async def cloud_disconnect(db: Session = Depends(get_db)):
    cloud_service.disconnect(db)
    return models.CloudStatusResponse(**cloud_service.get_status(db))
