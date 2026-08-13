"""REST endpoints for the Telnyx PSTN sink.

Mirrors the pattern from ``routes/speak.py``: a REST surface that wraps the
same logic as the ``voicebox.call`` MCP tool, plus CRUD for the Telnyx settings
singleton, plus the webhook receiver Telnyx calls back into when a call is
answered or playback finishes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from .. import models
from ..database import MCPClientBinding, get_db
from ..mcp_server import events as mcp_events
from ..mcp_server.resolve import resolve_profile
from ..services import sinks as sinks_service


logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Settings CRUD ─────────────────────────────────────────────────────────


@router.get(
    "/sinks/telnyx",
    response_model=models.TelnyxSettingsResponse,
    tags=["sinks"],
)
async def get_telnyx_settings(db: Session = Depends(get_db)):
    """Read the Telnyx sink settings. The API key is masked in the response
    so the Settings UI can render a "configured" hint without ever shipping
    the secret to the browser."""
    return _settings_response(sinks_service.get_settings(db))


@router.put(
    "/sinks/telnyx",
    response_model=models.TelnyxSettingsResponse,
    tags=["sinks"],
)
async def update_telnyx_settings(
    patch: models.TelnyxSettingsUpdate,
    db: Session = Depends(get_db),
):
    """Partial update. Passing ``api_key`` overwrites the stored key; passing
    ``api_key=None`` is a no-op (so the UI can save other fields without
    clearing the key it never received)."""
    payload = patch.model_dump(exclude_unset=True)
    # Don't null the API key if the caller just omitted it.
    if "api_key" in payload and payload["api_key"] is None:
        payload.pop("api_key")
    row = sinks_service.update_settings(db, payload)
    return _settings_response(row)


def _settings_response(row) -> models.TelnyxSettingsResponse:
    """Shared serializer so GET and PUT can't drift out of sync.

    ``missing`` is included so the Settings UI can tell the user exactly what
    is still needed instead of only discovering it when a call fails.
    """
    return models.TelnyxSettingsResponse(
        enabled=bool(row.enabled),
        api_key_masked=sinks_service.mask_key(row.api_key),
        api_key_set=bool(row.api_key),
        connection_id=row.connection_id,
        from_number=row.from_number,
        public_base_url=row.public_base_url,
        default_profile_id=row.default_profile_id,
        auto_hangup=bool(row.auto_hangup),
        missing=sinks_service.missing_settings(row),
    )


@router.get("/sinks/telnyx/health", tags=["sinks"])
async def telnyx_health(db: Session = Depends(get_db)):
    """Verify the API key by listing phone numbers. Returns 200 with
    ``{ok: true/false}`` — never raises, so the UI can show a green/red dot."""
    row = sinks_service.get_settings(db)
    if not row.api_key:
        return {"ok": False, "reason": "no_api_key"}
    try:
        async with sinks_service.TelnyxClient(row.api_key) as client:
            ok = await client.health()
        return {"ok": ok}
    except Exception:
        # Log the detail locally; the response stays generic so a failed probe
        # can't echo internal state back through the API.
        logger.exception("Telnyx health check failed")
        return {"ok": False, "reason": "request_failed"}


# ─── Call endpoint (REST mirror of voicebox.call) ──────────────────────────


@router.post("/call", response_model=models.CallResponse, tags=["sinks"])
async def call_endpoint(
    data: models.CallRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Dial a number and play a generation on the call. Mirrors the
    ``voicebox.call`` MCP tool — same profile resolution, same generation
    path, same webhook-driven playback.

    Returns a ``CallResponse`` with ``status="dialing"`` and a
    ``call_control_id``. The caller can poll ``GET /generate/{id}/status``
    to know when the generation finished; the actual playback happens
    asynchronously once Telnyx fires the ``call.answered`` webhook.
    """
    row = sinks_service.get_settings(db)
    missing = sinks_service.missing_settings(row)
    if missing:
        raise HTTPException(status_code=400, detail=sinks_service.setup_hint(missing))

    if bool(data.text) == bool(data.generation_id):
        raise HTTPException(
            status_code=400,
            detail="Pass exactly one of `text` or `generation_id`.",
        )

    client_id = request.headers.get("X-Voicebox-Client-Id")

    # Resolve profile the same way POST /speak does: one resolve_profile call
    # handles explicit arg → per-client binding → global capture default. Only
    # if all of those miss do we fall back to the sink's own default, so a
    # caller with an MCP binding gets the same voice on /call as on /speak.
    profile = resolve_profile(data.profile, client_id, db)
    if profile is None and data.profile:
        raise HTTPException(
            status_code=404,
            detail=f"Voice profile '{data.profile}' not found.",
        )
    if profile is None and row.default_profile_id:
        profile = resolve_profile(row.default_profile_id, client_id, db)
    if profile is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No voice profile resolved. Pass `profile`, set a default "
                "in Voicebox → Settings → Sinks → Telnyx, or bind a profile "
                "to this client in Settings → MCP."
            ),
        )

    # Per-client defaults for personality/engine, matching POST /speak.
    binding = None
    if client_id:
        binding = (
            db.query(MCPClientBinding)
            .filter(MCPClientBinding.client_id == client_id)
            .first()
        )

    personality_flag = data.personality
    if personality_flag is None and binding is not None:
        personality_flag = bool(binding.default_personality)

    engine = data.engine
    if engine is None and binding is not None:
        engine = binding.default_engine

    try:
        result = await sinks_service.place_call(
            to=data.to,
            text=data.text,
            generation_id=data.generation_id,
            profile_id=profile.id,
            profile_name=profile.name,
            engine=engine,
            language=data.language,
            personality=bool(personality_flag),
            auto_hangup=data.auto_hangup,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    mcp_events.publish(
        "call-start",
        {
            "call_control_id": result["call_control_id"],
            "generation_id": result["generation_id"],
            "to": data.to,
            "profile_name": profile.name,
            "source": "rest",
            "client_id": client_id,
        },
    )
    return result


# ─── Webhook receiver ──────────────────────────────────────────────────────


@router.post("/sinks/telnyx/webhook", tags=["sinks"])
async def telnyx_webhook(
    request: Request,
    token: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Telnyx call-control webhook receiver. Telnyx POSTs here on
    ``call.answered`` and ``playback.ended`` (and several other events we
    ignore). The ``token`` query param is a per-call secret embedded in the
    ``webhook_url`` we sent Telnyx at call-creation time; verifying it stops
    a third party who knows the URL from triggering playback on a live call.

    This endpoint is unauthenticated by design — Telnyx can't be asked to
    authenticate. The per-call token is the only gate. Future hardening:
    verify the Telnyx webhook signature header against the application's
    public key.
    """
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "invalid_json"}

    return await sinks_service.handle_telnyx_webhook(
        body=body,
        token=token,
        db=db,
    )
