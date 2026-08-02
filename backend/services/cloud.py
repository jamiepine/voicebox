"""
Voicebox Cloud device login — the "Log in with browser" flow.

The desktop opens the browser to ``{web}/connect``; the user authorizes while
signed in; the cloud redirects a single-use code back to this backend's loopback
callback. We exchange that code (server-to-server, over TLS) for a ``voicebox_…``
API key, verify the key against the API, and store it locally. The key never
travels through a browser URL, and an unfinished flow leaves nothing behind.

The ``state`` we mint and round-trip prevents login-CSRF: a callback whose state
we didn't issue (e.g. an attacker tricking the user into hitting the loopback
callback with their own code) is rejected.
"""

import logging
import secrets
import time
import webbrowser
from urllib.parse import urlencode

import httpx
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import config
from ..database import CloudSettings as DBCloudSettings

logger = logging.getLogger(__name__)

SINGLETON_ID = 1
PENDING_TTL_SECONDS = 600  # the whole browser flow must finish within 10 min

# state -> expiry epoch. In-memory: a single backend process owns the flow, and a
# dropped pairing should simply be restarted.
_pending: dict[str, float] = {}


def _prune() -> None:
    now = time.time()
    for state, expiry in list(_pending.items()):
        if expiry < now:
            _pending.pop(state, None)


def _json_dict(response: httpx.Response) -> dict | None:
    """Parsed JSON body, or None when it isn't a JSON object."""
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _consume_state(state: str) -> bool:
    """Validate and single-use-consume a pending state."""
    _prune()
    expiry = _pending.pop(state, None)
    return expiry is not None and expiry >= time.time()


def start_login(callback_url: str, device_name: str) -> str:
    """Mint a state, build the authorize URL, and open the browser.

    Returns the authorize URL (also opened here) so the caller can surface it as
    a fallback if the browser didn't open.
    """
    state = secrets.token_urlsafe(24)
    _prune()
    _pending[state] = time.time() + PENDING_TTL_SECONDS

    params = urlencode({"redirect_uri": callback_url, "state": state, "name": device_name})
    authorize_url = f"{config.get_cloud_web_url()}/connect?{params}"

    try:
        webbrowser.open(authorize_url)
    except Exception:  # pragma: no cover - platform dependent
        logger.exception("failed to open browser for cloud login")

    return authorize_url


async def handle_callback(db: Session, code: str, state: str) -> tuple[bool, str]:
    """Exchange the code for an API key and store it. Returns (ok, message)."""
    if not _consume_state(state):
        return False, "This sign-in link is invalid or has expired. Start again from the app."
    if not code:
        return False, "Missing authorization code."

    web = config.get_cloud_web_url()
    api = config.get_cloud_api_url()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            exchanged = await client.post(f"{web}/api/connect/exchange", json={"code": code})
            if exchanged.status_code != 200:
                logger.warning("cloud exchange rejected code: %s", exchanged.status_code)
                return False, "Could not complete sign-in — the code was rejected."
            payload = _json_dict(exchanged)
            if payload is None:
                logger.warning("cloud exchange returned a non-JSON payload")
                return False, "Voicebox Cloud returned an unexpected response."
            api_key = payload.get("key")
            device_name = payload.get("label")
            if not api_key:
                return False, "Voicebox Cloud did not return a key."

            # Confirm the freshly minted key actually authenticates the API.
            me = await client.get(
                f"{api}/v1/account/me",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if me.status_code != 200:
                logger.warning("minted key failed verification: %s", me.status_code)
                return False, "Sign-in succeeded but the key could not be verified."
            # The 200 above proves the key works; the user id is best-effort.
            data = (_json_dict(me) or {}).get("data")
            account_user_id = data.get("userId") if isinstance(data, dict) else None
    except httpx.HTTPError:
        logger.exception("network error during cloud exchange")
        return False, "Could not reach Voicebox Cloud. Check your connection and try again."

    _store_key(db, api_key=api_key, device_name=device_name, account_user_id=account_user_id)
    logger.info("connected to Voicebox Cloud as device %r", device_name)
    return True, "Connected"


def _get_or_create_row(db: Session) -> DBCloudSettings:
    row = db.query(DBCloudSettings).filter(DBCloudSettings.id == SINGLETON_ID).first()
    if row is None:
        row = DBCloudSettings(id=SINGLETON_ID)
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            # Another request created the singleton concurrently.
            db.rollback()
            row = db.query(DBCloudSettings).filter(DBCloudSettings.id == SINGLETON_ID).one()
        else:
            db.refresh(row)
    return row


def _store_key(db: Session, *, api_key: str, device_name: str | None, account_user_id: str | None):
    from datetime import datetime

    row = _get_or_create_row(db)
    row.api_key = api_key
    row.device_name = device_name
    row.account_user_id = account_user_id
    row.connected_at = datetime.utcnow()
    db.commit()


def get_status(db: Session) -> dict:
    """Local view of the cloud link — never returns the full key."""
    row = _get_or_create_row(db)
    connected = bool(row.api_key)
    # Prefix only: "voicebox_" (9) + 8 chars, matching the cloud's key_prefix.
    key_prefix = row.api_key[:17] if row.api_key else None
    return {
        "connected": connected,
        "device_name": row.device_name if connected else None,
        "account_user_id": row.account_user_id if connected else None,
        "key_prefix": key_prefix,
        "connected_at": row.connected_at if connected else None,
        "dashboard_url": f"{config.get_cloud_web_url()}/account",
    }


def disconnect(db: Session) -> None:
    """Forget the local credential. The key remains valid on the server until
    revoked from the account dashboard — surface that in the UI."""
    row = _get_or_create_row(db)
    row.api_key = None
    row.device_name = None
    row.account_user_id = None
    row.connected_at = None
    db.commit()


def get_api_key(db: Session) -> str | None:
    """The stored bearer key, for the (future) sync client. None if not linked."""
    row = _get_or_create_row(db)
    return row.api_key
