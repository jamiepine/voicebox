"""POST /speak — REST wrapper around voicebox.speak for non-MCP callers.

Shell scripts, ACP, A2A, or any agent that doesn't speak MCP can hit this
endpoint to play text through a cloned voice. Uses the same profile
resolution and generation pipeline as the MCP tool, so per-client
bindings (via X-Voicebox-Client-Id) work identically.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from .. import models
from ..database import MCPClientBinding, get_db
from ..mcp_server import events as mcp_events
from ..mcp_server.resolve import resolve_profile
from ..services import history


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/speak", response_model=models.GenerationResponse)
async def speak(
    data: models.SpeakRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Speak text in a voice profile. Mirrors voicebox.speak (MCP).

    Response shape matches POST /generate — a ``GenerationResponse`` with
    ``status="generating"`` and an ``id`` the caller polls at
    ``GET /generate/{id}/status``.
    """
    client_id = request.headers.get("X-Voicebox-Client-Id")
    profile = resolve_profile(data.profile, client_id, db)
    if profile is None:
        if data.profile:
            raise HTTPException(
                status_code=404,
                detail=f"Voice profile '{data.profile}' not found.",
            )
        raise HTTPException(
            status_code=400,
            detail=(
                "No voice profile resolved. Pass `profile` (name or id), "
                "or configure a default in Voicebox → Settings → MCP."
            ),
        )

    binding = None
    if client_id:
        binding = (
            db.query(MCPClientBinding)
            .filter(MCPClientBinding.client_id == client_id)
            .first()
        )

    # Resolve per-client personality default when the caller didn't pin it.
    personality_flag = data.personality
    if personality_flag is None and binding is not None:
        personality_flag = bool(binding.default_personality)

    engine = data.engine
    if engine is None and binding is not None:
        engine = binding.default_engine

    from .generations import generate_speech

    generation = await generate_speech(
        models.GenerationRequest(
            profile_id=profile.id,
            text=data.text,
            language=data.language or "en",
            engine=engine,
            personality=bool(personality_flag),
        ),
        db,
    )

    mcp_events.publish(
        "speak-start",
        {
            "generation_id": getattr(generation, "id", None),
            "profile_name": profile.name,
            "source": "rest",
            "client_id": client_id,
        },
    )
    return generation


@router.get("/tts")
async def tts_get(
    request: Request,
    text: str = Query(..., min_length=1, max_length=10000),
    voice: str | None = Query(
        default=None,
        description="Alias for profile. Voice profile name or id.",
    ),
    profile: str | None = Query(
        default=None,
        description="Voice profile name or id. Overrides voice when both are set.",
    ),
    model: str | None = Query(
        default=None,
        pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$",
        description="Alias for engine.",
    ),
    engine: str | None = Query(
        default=None,
        pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$",
    ),
    personality: bool | None = Query(default=None),
    language: str = Query(
        default="en",
        pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$",
    ),
    stream: bool = Query(
        default=False,
        description=(
            "When true, stream WAV audio directly instead of returning "
            "JSON/status links. Ignores wait/timeout_s/poll_ms/response."
        ),
    ),
    wait: bool = Query(
        default=False,
        description="Wait for completion. false returns links immediately.",
    ),
    timeout_s: int = Query(
        default=120,
        ge=1,
        le=1800,
        description="Max wait time when wait=true.",
    ),
    poll_ms: int = Query(
        default=500,
        ge=100,
        le=5000,
        description="Polling interval when wait=true.",
    ),
    response: str = Query(
        default="json",
        pattern="^(json|redirect|audio)$",
        description="json: metadata, redirect: 307 to /audio/{id}, audio: stream file.",
    ),
    db: Session = Depends(get_db),
):
    """GET-friendly TTS endpoint for local device integrations.

    Supports URL query params (text/voice/model) and can either:
    - return links immediately (default),
    - stream WAV immediately while generating (stream=true),
    - wait then redirect to the generated audio URL,
    - wait then stream the generated audio bytes.
    """
    selected_profile = profile or voice
    # Keep caller-provided engine/model precedence, but make /tts default to
    # LuxTTS when neither alias is provided.
    selected_engine = engine or model or "luxtts"

    if stream:
        client_id = request.headers.get("X-Voicebox-Client-Id")
        resolved_profile = resolve_profile(selected_profile, client_id, db)
        if resolved_profile is None:
            if selected_profile:
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice profile '{selected_profile}' not found.",
                )
            raise HTTPException(
                status_code=400,
                detail=(
                    "No voice profile resolved. Pass `profile` (name or id), "
                    "or configure a default in Voicebox -> Settings -> MCP."
                ),
            )

        binding = None
        if client_id:
            binding = (
                db.query(MCPClientBinding)
                .filter(MCPClientBinding.client_id == client_id)
                .first()
            )

        personality_flag = personality
        if personality_flag is None and binding is not None:
            personality_flag = bool(binding.default_personality)

        from .generations import stream_speech

        return await stream_speech(
            models.GenerationRequest(
                profile_id=resolved_profile.id,
                text=text,
                language=language,
                engine=selected_engine,
                personality=bool(personality_flag),
            ),
            db,
        )

    generation = await speak(
        models.SpeakRequest(
            text=text,
            profile=selected_profile,
            engine=selected_engine,
            personality=personality,
            language=language,
        ),
        request,
        db,
    )

    base = str(request.base_url).rstrip("/")
    payload = {
        "id": generation.id,
        "status": generation.status,
        "status_url": f"{base}/generate/{generation.id}/status",
        "audio_url": f"{base}/audio/{generation.id}",
    }

    if not wait:
        return payload

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        row = await history.get_generation(generation.id, db)
        if row is None:
            raise HTTPException(status_code=404, detail="Generation not found")

        if row.status == "completed":
            audio_url = f"/audio/{generation.id}"
            if response == "redirect":
                return RedirectResponse(url=audio_url, status_code=307)
            if response == "audio":
                from .audio import get_audio

                return await get_audio(generation.id, db)
            payload["status"] = "completed"
            return payload

        if row.status == "failed":
            raise HTTPException(
                status_code=500,
                detail=row.error or "Generation failed",
            )

        await asyncio.sleep(poll_ms / 1000)

    payload["status"] = "generating"
    return JSONResponse(status_code=202, content=payload)
