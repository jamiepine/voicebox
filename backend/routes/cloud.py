"""Voicebox Cloud routes: device login + encrypted backup/sync.

The browser-based pairing flow:
  1. POST /cloud/login/start  — opens the browser to the cloud authorize page.
  2. GET  /cloud/callback     — the browser lands here with a one-time code;
                                the backend exchanges it for an API key.
  3. GET  /cloud/status       — the UI polls this to learn when it connected.
  4. POST /cloud/disconnect   — forget the local credential.

Sync, once logged in:
  5. POST /cloud/sync/setup    — register as an encryption device; on a keyless
                                 account this mints the master key and returns
                                 the recovery phrase (shown exactly once).
  6. POST /cloud/sync/restore  — recover the master key from the phrase.
  7. POST /cloud/sync/adopt    — pick up a wrapped key another device provisioned.
  8. GET  /cloud/sync/status   — identity state + cursor.
  9. POST /cloud/sync/run      — one full push+pull pass.
"""

import socket

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from .. import models
from ..database import CloudSettings as DBCloudSettings, get_db
from ..services import cloud as cloud_service, cloud_account, cloud_sync
from ..services.cloud_account import CloudAccountError
from ..services.cloud_api import CloudApiError
from ..services.cloud_crypto import CloudCryptoError
from ..services.cloud_keys import CloudKeyStoreError
from ..services.cloud_sync import CloudSyncError

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
    sub = "Voicebox is now linked to your account. You can close this tab and return to the app." if ok else message
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


# ─── Encrypted backup & sync ─────────────────────────────────────────────

_SYNC_ERRORS = (CloudAccountError, CloudApiError, CloudCryptoError, CloudKeyStoreError, CloudSyncError)


def _sync_status(db: Session) -> models.CloudSyncStatusResponse:
    identity = cloud_account.identity_status(db)
    row = db.query(DBCloudSettings).filter(DBCloudSettings.id == 1).first()
    return models.CloudSyncStatusResponse(
        status=identity.status,
        device_id=identity.device_id,
        sync_cursor=(row.sync_cursor or 0) if row else 0,
    )


@router.post("/sync/setup", response_model=models.CloudSyncSetupResponse)
async def cloud_sync_setup(db: Session = Depends(get_db)):
    try:
        phrase = await cloud_account.setup_device(db)
        identity = cloud_account.identity_status(db)
    except _SYNC_ERRORS as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return models.CloudSyncSetupResponse(
        status=identity.status,
        device_id=identity.device_id,
        recovery_phrase=phrase,
    )


@router.post("/sync/restore", response_model=models.CloudSyncStatusResponse)
async def cloud_sync_restore(body: models.CloudRestoreRequest, db: Session = Depends(get_db)):
    try:
        await cloud_account.restore_with_phrase(db, body.phrase)
    except _SYNC_ERRORS as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return _sync_status(db)


@router.post("/sync/adopt", response_model=models.CloudSyncStatusResponse)
async def cloud_sync_adopt(db: Session = Depends(get_db)):
    try:
        await cloud_account.adopt_wrapped_key(db)
    except _SYNC_ERRORS as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return _sync_status(db)


@router.get("/sync/status", response_model=models.CloudSyncStatusResponse)
async def cloud_sync_status(db: Session = Depends(get_db)):
    try:
        return _sync_status(db)
    except CloudAccountError:
        return models.CloudSyncStatusResponse(status="unregistered", device_id=None, sync_cursor=0)


@router.post("/sync/run", response_model=models.CloudSyncRunResponse)
async def cloud_sync_run(db: Session = Depends(get_db)):
    try:
        report = await cloud_sync.run_sync(db)
    except _SYNC_ERRORS as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return models.CloudSyncRunResponse(
        pushed=report.pushed,
        pushed_deletes=report.pushed_deletes,
        pulled=report.pulled,
        pulled_deletes=report.pulled_deletes,
        cursor=report.cursor,
    )
