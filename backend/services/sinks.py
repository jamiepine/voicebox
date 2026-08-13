"""Telnyx PSTN sink — dial a number and play a Voicebox generation on the call.

Thin HTTP-only client against ``api.telnyx.com/v2``. No SDK dependency; the
only network library is ``httpx``, which the cloud service already imports at
module scope. Generation stays local; only the PSTN leg touches Telnyx.

Flow:
    1. Caller hits ``POST /call`` (REST) or ``voicebox.call`` (MCP).
    2. We resolve a voice profile and either reuse an existing generation_id
       or run ``generate_speech`` to produce one (status="generating").
    3. We POST to Telnyx ``/v2/calls`` with ``connection_id`` (the Call Control
       Application) plus a ``webhook_url`` pointing back at this Voicebox
       server, scoped to the call via a per-call ``webhook_secret``.
    4. Telnyx returns ``call_control_id`` immediately; we persist a
       ``TelnyxCall`` row mapping it to the generation and return to the caller.
    5. Telnyx fires ``call.answered`` asynchronously → ``POST /sinks/telnyx/webhook``.
       The webhook acks immediately and finishes the work in a background task:
       wait for the generation, then ``POST /calls/{id}/actions/playback_start``
       with an ``audio_url`` pointing at ``GET /audio/{generation_id}``.
    6. If ``auto_hangup`` is set, ``playback.ended`` triggers a hangup.

The user must expose their Voicebox server publicly (ngrok, Tailscale Funnel,
Cloudflare Tunnel, etc.) and set ``public_base_url`` in Telnyx settings so
both the outbound webhook_url and the playback audio_url are reachable from
Telnyx. ``GET /audio/{generation_id}`` already exists and serves the file
with the right Content-Type; we just hand Telnyx its public URL.

Security note: that tunnel exposes *every* Voicebox route, not just the two
Telnyx needs. Voicebox binds to 127.0.0.1 and has no request auth, so a naive
``ngrok http 17493`` publishes the profile, generation, and settings APIs to
the internet. The docs tell users to put a reverse proxy in front that only
allows ``/sinks/telnyx/webhook`` and ``/audio/``.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
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
# This runs in a background task, not inside the webhook request, so it does
# not hold Telnyx's webhook connection open.
PLAYBACK_WAIT_SECS = 30
POLL_INTERVAL_SECS = 0.5

# Statuses that mean "call.answered has already been handled (or is being
# handled right now)". Telnyx retries webhooks it considers failed, so the
# answered handler claims the row before doing any slow work.
_ANSWERED_CLAIMED = ("preparing", "playing", "playback_ended", "hangup", "failed")

# Settings fields that must be present before we can dial.
_REQUIRED_SETTINGS = ("api_key", "connection_id", "from_number", "public_base_url")


SINGLETON_ID = 1

# Strong refs to detached webhook tasks so the loop can't collect them midway.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


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


def missing_settings(row: DBTelnyxSettings) -> list[str]:
    """Names of the settings fields still needed before the sink can dial.

    Single source of truth for "is this thing set up?" — both the request-time
    gate and ``dial_and_play`` call it, so the two can't drift apart and let a
    request past one check only to fail the other deeper in.
    """
    missing = [f for f in _REQUIRED_SETTINGS if not getattr(row, f, None)]
    if not row.enabled:
        # Explicit toggle: creds can be filled in while the sink stays off.
        missing.append("enabled")
    return missing


def setup_hint(missing: list[str]) -> str:
    """Error text naming exactly which fields are missing."""
    return (
        "Telnyx sink not configured (missing: "
        + ", ".join(missing)
        + "). Set these in Voicebox → Settings → Sinks → Telnyx."
    )


def mask_key(key: str | None) -> str:
    """Mask all but the last 4 chars — same shape as the cloud settings UI."""
    if not key:
        return ""
    if len(key) <= 4:
        return "*" * len(key)
    return "*" * (len(key) - 4) + key[-4:]


# ─── Thin HTTP client ──────────────────────────────────────────────────────


def _raise_for_telnyx_status(resp: httpx.Response, action: str) -> None:
    """Turn a Telnyx error response into a ``ValueError`` carrying its detail.

    Misconfiguration is the common failure here (wrong connection_id, number
    not on the account, unreachable webhook URL), and Telnyx explains each one
    in its JSON error body. ``raise_for_status`` would throw that away and the
    caller would surface a bare 500, so unpack ``errors[].detail`` instead and
    let the route map it to a 400 the user can act on.
    """
    if resp.status_code < 400:
        return

    detail = ""
    try:
        errors = resp.json().get("errors") or []
        detail = "; ".join(
            " ".join(
                part
                for part in (e.get("title"), e.get("detail"))
                if part
            )
            for e in errors
            if isinstance(e, dict)
        )
    except Exception:  # noqa: BLE001 - non-JSON error body
        detail = (resp.text or "").strip()[:300]

    raise ValueError(
        f"Telnyx {action} failed ({resp.status_code})"
        + (f": {detail}" if detail else "")
    )


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
        connection_id: str,
        webhook_url: str,
    ) -> dict[str, Any]:
        """POST /calls — returns Telnyx's ``data`` payload.

        ``connection_id`` (the Call Control Application ID) is required by the
        API alongside ``to`` and ``from``; omitting it fails the request.
        """
        resp = await self._client.post(
            "/calls",
            json={
                "to": to,
                "from": from_,
                "connection_id": connection_id,
                "webhook_url": webhook_url,
            },
        )
        _raise_for_telnyx_status(resp, "dial")
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected Telnyx dial response: {payload!r}")
        return data

    async def playback_start(self, call_control_id: str, audio_url: str) -> None:
        """Play a file on the call.

        The API field is ``audio_url`` — there is no ``media_url`` on this
        action (``media_name`` refers to previously uploaded media instead).
        """
        resp = await self._client.post(
            f"/calls/{call_control_id}/actions/playback_start",
            json={"audio_url": audio_url},
        )
        _raise_for_telnyx_status(resp, "playback_start")

    async def hangup(self, call_control_id: str) -> None:
        resp = await self._client.post(
            f"/calls/{call_control_id}/actions/hangup",
        )
        # 404/422 mean the call already ended — not an error from our POV.
        if resp.status_code in (404, 422):
            return
        _raise_for_telnyx_status(resp, "hangup")

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
    auto_hangup: bool | None,
    db: Session,
) -> DBTelnyxCall:
    """Dial ``to`` and arrange for ``generation_id`` to play once the call
    answers. Persists the pending call row and returns it. Caller is
    responsible for ensuring the generation exists; this function does not
    poll for generation completion (the webhook handler does that).
    """
    s = _get_or_create_settings_row(db)
    missing = missing_settings(s)
    if missing:
        raise ValueError(setup_hint(missing))

    base = s.public_base_url.rstrip("/")
    webhook_secret = secrets.token_urlsafe(24)
    webhook_url = f"{base}/sinks/telnyx/webhook?token={webhook_secret}"

    async with TelnyxClient(s.api_key) as client:
        data = await client.create_call(
            to=to,
            from_=s.from_number,
            connection_id=s.connection_id,
            webhook_url=webhook_url,
        )

    # Only call_control_id addresses call-control actions. call_session_id is
    # a different identifier and every playback/hangup against it would 404,
    # so fail loudly here rather than dial a call we can never control.
    call_control_id = data.get("call_control_id")
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
        auto_hangup=bool(s.auto_hangup if auto_hangup is None else auto_hangup),
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

    # Per-call webhook_secret is in the query string. Compared in constant
    # time, and a row with no stored secret fails closed — every row this
    # code path creates has one, so a missing secret means something is wrong
    # rather than "legacy row worth trusting".
    if not row.webhook_secret or not secrets.compare_digest(
        str(row.webhook_secret), str(token or "")
    ):
        return {"ok": False, "reason": "bad_token"}

    if event_type == "call.answered":
        # Claim the row before returning so a Telnyx retry (it re-POSTs when a
        # webhook is slow or errors) can't start a second playback on the same
        # call. The claim commits synchronously; the slow work runs detached.
        if _claim_for_playback(row, db):
            _spawn(_run_call_answered(row.call_control_id))
    elif event_type == "playback.ended":
        _spawn(_run_playback_ended(row.call_control_id))
    # Other events (call.initiated, call.hangup, call.bridge, etc.) are
    # ignored for now — we don't need them to fulfill the basic dial+play.

    return {"ok": True, "event": event_type, "call_control_id": call_control_id}


def _spawn(coro) -> None:
    """Run a coroutine detached from the webhook request.

    Telnyx wants a 2xx within 2000 ms and retries the delivery otherwise, so
    the handler acknowledges first and does the slow part here. Waiting for a
    generation can take tens of seconds; doing that inline guaranteed a
    timeout, a retry, and a second playback on the same call. The task holds a
    reference until done so it isn't garbage collected mid-flight.
    """
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def _claim_for_playback(row: DBTelnyxCall, db: Session) -> bool:
    """Transition the row into ``preparing`` exactly once.

    Returns True for the caller that won the claim, False if another webhook
    delivery already took it. Guards against Telnyx's at-least-once delivery
    turning into two ``playback_start`` calls and doubled audio.
    """
    if row.status in _ANSWERED_CLAIMED:
        return False
    row.status = "preparing"
    db.commit()
    return True


async def _run_call_answered(call_control_id: str) -> None:
    """Background half of ``call.answered``: wait for audio, then play it.

    Opens its own session — the webhook request's session is closed by the
    time this runs.
    """
    db = next(get_db())
    try:
        row = (
            db.query(DBTelnyxCall)
            .filter(DBTelnyxCall.call_control_id == call_control_id)
            .first()
        )
        if row is None:
            return

        s = _get_or_create_settings_row(db)
        if not (s.api_key and s.public_base_url):
            row.status = "failed"
            row.error = "Telnyx settings missing on webhook"
            db.commit()
            return

        # Wait for generation to complete. The /call endpoint returns
        # immediately after Telnyx accepts the dial, so by the time the callee
        # picks up the generation is usually already done. Polling here covers
        # the edge case.
        if not await _wait_for_generation(row.generation_id):
            row.status = "failed"
            row.error = "Generation did not complete in time"
            db.commit()
            return

        audio_url = f"{s.public_base_url.rstrip('/')}/audio/{row.generation_id}"
        try:
            async with TelnyxClient(s.api_key) as client:
                await client.playback_start(row.call_control_id, audio_url)
        except Exception as exc:  # noqa: BLE001 - recorded on the row
            row.status = "failed"
            row.error = f"playback_start failed: {exc}"
            logger.exception(
                "Telnyx playback_start failed for call %s", row.call_control_id
            )
            db.commit()
            return

        row.status = "playing"
        db.commit()
    finally:
        db.close()


async def _run_playback_ended(call_control_id: str) -> None:
    """Background half of ``playback.ended``: hang up when configured to."""
    db = next(get_db())
    try:
        row = (
            db.query(DBTelnyxCall)
            .filter(DBTelnyxCall.call_control_id == call_control_id)
            .first()
        )
        if row is None or row.status in ("playback_ended", "hangup"):
            return

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
        except Exception as exc:  # noqa: BLE001 - recorded on the row
            row.error = f"hangup failed: {exc}"
            logger.exception("Telnyx hangup failed for call %s", row.call_control_id)
        db.commit()
    finally:
        db.close()


async def _wait_for_generation(generation_id: str) -> bool:
    """Poll the Generation row until it leaves the "generating" state, or
    PLAYBACK_WAIT_SECS elapses. Uses a fresh session each poll to avoid stale
    ORM cache across the long sleep window."""
    deadline = time.monotonic() + PLAYBACK_WAIT_SECS
    while time.monotonic() < deadline:
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

    # auto_hangup goes in at row creation — patching it afterwards left a
    # window where an early playback.ended could read the sink default.
    call_row = await dial_and_play(
        to=to,
        generation_id=used_generation_id,
        auto_hangup=auto_hangup,
        db=db,
    )

    return {
        "call_control_id": call_row.call_control_id,
        "status": call_row.status,
        "generation_id": used_generation_id,
        "profile": profile_name,
        "to": to,
        "poll_url": f"/generate/{used_generation_id}/status",
    }
