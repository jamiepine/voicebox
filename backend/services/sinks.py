"""Telnyx PSTN sink — dial a number and play a Voicebox generation on the call.

Thin HTTP-only client against ``api.telnyx.com/v2``. No SDK dependency, no
new requirements (httpx is already in ``requirements.txt``). Generation stays
local; only the PSTN leg touches Telnyx.

Flow:
    1. Caller hits ``POST /call`` (REST) or ``voicebox.call`` (MCP).
    2. We resolve a voice profile and either reuse an existing generation_id
       or run ``generate_speech`` to produce one (status="generating").
    3. We POST to Telnyx ``/v2/calls`` with ``webhook_url`` pointing back at
       this Voicebox server, scoped to the call via a per-call ``webhook_secret``.
    4. Telnyx returns ``call_control_id`` immediately; we persist a
       ``TelnyxCall`` row mapping it to the generation and return to the caller.
    5. Telnyx fires ``call.answered`` asynchronously → ``POST /sinks/telnyx/webhook``.
       The webhook handler waits for the generation to finish (if it hasn't yet),
       then issues ``POST /calls/{id}/actions/playback_start`` with a media
       URL pointing at the existing ``GET /audio/{generation_id}`` route.
    6. If ``auto_hangup`` is set, ``playback.ended`` triggers a hangup.

The user must expose their Voicebox server publicly (ngrok, Tailscale Funnel,
Cloudflare Tunnel, etc.) and set ``public_base_url`` in Telnyx settings so
both the outbound webhook_url and the playback media_url are reachable from
Telnyx. ``GET /audio/{generation_id}`` already exists and serves the file
with the right Content-Type; we just hand Telnyx its public URL.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session

from ..database import (
    Generation as DBGeneration,
    TelnyxCall as DBTelnyxCall,
    TelnyxSettings as DBTelnyxSettings,
    get_db,
)
from .. import models

logger = logging.getLogger(__name__)


TELNYX_API_BASE = "https://api.telnyx.com/v2"
# Wait this long for a generation to finish before giving up on the call.
# TTS for short clips finishes in 1-3s; the cap protects against a wedged
# worker leaving the line open in silence after the callee picks up.
PLAYBACK_WAIT_SECS = 30
POLL_INTERVAL_SECS = 0.5


SINGLETON_ID = 1


# ─── Settings singleton ────────────────────────────────────────────────────


def _get_or_create_settings_row(db: Session) -> DBTelnyxSettings:
    row = db.query(DBTelnyxSettings).filter(DBTelnyxSettings.id == SINGLETON_ID).first()
    if row is None:
        row = DBTelnyxSettings(id=SINGLETON_ID)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def get_settings(db: Session) -> DBTelnyxSettings:
    return _get_or_create_settings_row(db)


def update_settings(db: Session, patch: dict[str, Any]) -> DBTelnyxSettings:
    row = _get_or_create_settings_row(db)
    columns = type(row).__table__.columns
    for key, value in patch.items():
        col = columns.get(key)
        if col is None:
            continue
        if value is None and not col.nullable:
            continue
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return row


def is_configured(db: Session) -> bool:
    """True if the Telnyx sink has enough to dial: api_key + from_number +
    public_base_url. Used by the route and MCP tool to gate the "Call"
    action behind a friendly setup hint instead of a 500."""
    s = _get_or_create_settings_row(db)
    return bool(s.enabled and s.api_key and s.from_number and s.public_base_url)


def mask_key(key: str | None) -> str:
    """Mask all but the last 4 chars — same shape as the cloud settings UI."""
    if not key:
        return ""
    if len(key) <= 4:
        return "*" * len(key)
    return "*" * (len(key) - 4) + key[-4:]


# ─── Thin HTTP client ──────────────────────────────────────────────────────


class TelnyxClient:
    """Three endpoints, no SDK. Constructed per-call to avoid holding a
    connection across the long-lived webhook wait."""

    def __init__(self, api_key: str, *, timeout: float = 10.0):
        self._client = httpx.AsyncClient(
            base_url=TELNYX_API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "TelnyxClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self._client.aclose()

    async def create_call(
        self,
        *,
        to: str,
        from_: str,
        webhook_url: str,
    ) -> dict[str, Any]:
        """POST /calls — returns Telnyx's ``data`` payload."""
        resp = await self._client.post(
            "/calls",
            json={
                "to": to,
                "from": from_,
                "webhook_url": webhook_url,
            },
        )
        resp.raise_for_status()
        return resp.json()["data"]

    async def playback_start(self, call_control_id: str, media_url: str) -> None:
        resp = await self._client.post(
            f"/calls/{call_control_id}/actions/playback_start",
            json={"media_url": media_url, "overlay": False},
        )
        resp.raise_for_status()

    async def hangup(self, call_control_id: str) -> None:
        resp = await self._client.post(
            f"/calls/{call_control_id}/actions/hangup",
        )
        # 404 means the call already ended — not an error from our POV.
        if resp.status_code not in (200, 404):
            resp.raise_for_status()

    async def health(self) -> bool:
        """Verify the API key by listing phone numbers. Returns True on 200."""
        try:
            resp = await self._client.get("/phone_numbers", params={"page[size]": 1})
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


# ─── Dial-and-play orchestration ───────────────────────────────────────────


async def dial_and_play(
    *,
    to: str,
    generation_id: str,
    db: Session,
) -> DBTelnyxCall:
    """Dial ``to`` and arrange for ``generation_id`` to play once the call
    answers. Persists the pending call row and returns it. Caller is
    responsible for ensuring the generation exists; this function does not
    poll for generation completion (the webhook handler does that).
    """
    s = _get_or_create_settings_row(db)
    if not (s.api_key and s.from_number and s.public_base_url):
        raise ValueError(
            "Telnyx sink not configured. Set api_key, from_number, and "
            "public_base_url in Voicebox → Settings → Sinks → Telnyx."
        )

    base = s.public_base_url.rstrip("/")
    webhook_secret = secrets.token_urlsafe(24)
    webhook_url = f"{base}/sinks/telnyx/webhook?token={webhook_secret}"

    async with TelnyxClient(s.api_key) as client:
        data = await client.create_call(
            to=to,
            from_=s.from_number,
            webhook_url=webhook_url,
        )

    call_control_id = data.get("call_control_id") or data.get("call_session_id")
    if not call_control_id:
        raise ValueError(f"Telnyx returned no call_control_id: {data}")

    row = DBTelnyxCall(
        call_control_id=call_control_id,
        generation_id=generation_id,
        to_number=to,
        from_number=s.from_number,
        webhook_secret=webhook_secret,
        status="initiating",
        started_at=datetime.now(timezone.utc),
        auto_hangup=bool(s.auto_hangup),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ─── Webhook handler ──────────────────────────────────────────────────────


async def handle_telnyx_webhook(
    *,
    body: dict[str, Any],
    token: str | None,
    db: Session,
) -> dict[str, Any]:
    """Dispatch a Telnyx webhook event. Returns a small ack dict for the
    route to JSON back to Telnyx. Idempotent on ``call_control_id`` — replaying
    an event won't re-fire playback on a call that's already past that state.
    """
    data = body.get("data") or {}
    event_type = data.get("event_type") or ""
    payload = data.get("payload") or {}
    call_control_id = payload.get("call_control_id")

    if not call_control_id:
        return {"ok": True, "reason": "no_call_control_id"}

    row = (
        db.query(DBTelnyxCall)
        .filter(DBTelnyxCall.call_control_id == call_control_id)
        .first()
    )
    if row is None:
        # Unknown call — could be a stray from before sink tracking existed,
        # or someone probing the webhook. Either way, nothing to do.
        return {"ok": True, "reason": "unknown_call"}

    # Per-call webhook_secret is in the query string; we don't expose the
    # webhook endpoint publicly without it. If the row has no secret stored
    # (older rows), accept anyway — they were created before this gate.
    if row.webhook_secret and row.webhook_secret != token:
        return {"ok": False, "reason": "bad_token"}

    if event_type == "call.answered":
        await _on_call_answered(row, db)
    elif event_type == "playback.ended":
        await _on_playback_ended(row, db)
    # Other events (call.initiated, call.hangup, call.bridge, etc.) are
    # ignored for now — we don't need them to fulfill the basic dial+play.

    return {"ok": True, "event": event_type, "call_control_id": call_control_id}


async def _on_call_answered(row: DBTelnyxCall, db: Session) -> None:
    """Call was picked up. Wait for the generation to finish, then issue
    playback_start with the public audio URL."""
    if row.status in ("playing", "playback_ended", "hangup", "failed"):
        return  # already advanced past this state

    s = _get_or_create_settings_row(db)
    if not (s.api_key and s.public_base_url):
        row.status = "failed"
        row.error = "Telnyx settings missing on webhook"
        db.commit()
        return

    # Wait for generation to complete. The /call endpoint returns immediately
    # after Telnyx accepts the dial, so by the time the callee picks up the
    # generation is usually already done. Polling here covers the edge case.
    ok = await _wait_for_generation(row.generation_id, db)
    if not ok:
        row.status = "failed"
        row.error = "Generation did not complete in time"
        db.commit()
        return

    media_url = (
        f"{s.public_base_url.rstrip('/')}/audio/{row.generation_id}"
    )
    try:
        async with TelnyxClient(s.api_key) as client:
            await client.playback_start(row.call_control_id, media_url)
    except Exception as exc:
        row.status = "failed"
        row.error = f"playback_start failed: {exc}"
        logger.exception("Telnyx playback_start failed for call %s", row.call_control_id)
        db.commit()
        return

    row.status = "playing"
    db.commit()


async def _on_playback_ended(row: DBTelnyxCall, db: Session) -> None:
    """Playback finished. If auto_hangup is on, hang up the call."""
    row.status = "playback_ended"
    db.commit()

    if not row.auto_hangup:
        return

    s = _get_or_create_settings_row(db)
    if not s.api_key:
        return

    try:
        async with TelnyxClient(s.api_key) as client:
            await client.hangup(row.call_control_id)
        row.status = "hangup"
        row.completed_at = datetime.now(timezone.utc)
    except Exception as exc:
        row.error = f"hangup failed: {exc}"
        logger.exception("Telnyx hangup failed for call %s", row.call_control_id)
    db.commit()


async def _wait_for_generation(generation_id: str, db: Session) -> bool:
    """Poll the Generation row until it leaves the "generating" state, or
    PLAYBACK_WAIT_SECS elapses. Uses a fresh session each poll to avoid stale
    ORM cache across the long sleep window."""
    deadline = asyncio.get_event_loop().time() + PLAYBACK_WAIT_SECS
    while asyncio.get_event_loop().time() < deadline:
        # New session per check so we don't read cached state.
        fresh = next(get_db())
        try:
            gen = fresh.query(DBGeneration).filter_by(id=generation_id).first()
            if gen is None:
                return False
            status = gen.status or "completed"
            if status == "completed":
                return True
            if status == "failed":
                return False
        finally:
            fresh.close()
        await asyncio.sleep(POLL_INTERVAL_SECS)
    return False


# ─── Top-level call entry (used by REST + MCP) ──────────────────────────────


async def place_call(
    *,
    to: str,
    text: str | None,
    generation_id: str | None,
    profile_id: str,
    profile_name: str,
    engine: str | None,
    language: str | None,
    personality: bool,
    auto_hangup: bool | None,
    db: Session,
) -> dict[str, Any]:
    """Resolve the generation (existing or fresh), dial the call, return the
    shape both ``POST /call`` and ``voicebox.call`` return."""
    from ..routes.generations import generate_speech

    if bool(text) == bool(generation_id):
        raise ValueError(
            "Pass exactly one of `text` (generate inline) or `generation_id` "
            "(use an existing generation)."
        )

    if generation_id is not None:
        gen = db.query(DBGeneration).filter_by(id=generation_id).first()
        if gen is None:
            raise ValueError(f"Generation '{generation_id}' not found.")
        if (gen.status or "completed") != "completed":
            raise ValueError(
                f"Generation '{generation_id}' is not completed (status: "
                f"{gen.status}). Wait for it to finish before dialing."
            )
        used_generation_id = generation_id
    else:
        # Generate inline using the same path POST /generate takes.
        generation = await generate_speech(
            models.GenerationRequest(
                profile_id=profile_id,
                text=text or "",
                language=language or "en",
                engine=engine,
                personality=personality,
            ),
            db,
        )
        used_generation_id = generation.id

    call_row = await dial_and_play(
        to=to,
        generation_id=used_generation_id,
        db=db,
    )

    # Override the sink's default auto_hangup if the caller asked.
    if auto_hangup is not None and auto_hangup != call_row.auto_hangup:
        call_row.auto_hangup = bool(auto_hangup)
        db.commit()

    return {
        "call_control_id": call_row.call_control_id,
        "status": call_row.status,
        "generation_id": used_generation_id,
        "profile": profile_name,
        "to": to,
        "poll_url": f"/generate/{used_generation_id}/status",
    }
