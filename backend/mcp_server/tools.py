"""Voicebox MCP tool implementations.

Thin wrappers over existing services/routes. Tools are registered with dotted
names (``voicebox.speak`` etc.) so they look natural in agent logs —
the Python function name stays snake_case.
"""

from __future__ import annotations

import asyncio
import base64 as b64
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .. import config, models
from ..database import Generation as DBGeneration, get_db
from ..services import captures as captures_service
from ..services import profiles as profiles_service
from . import events as mcp_events
from .context import current_client_id, request_is_loopback
from .resolve import resolve_profile


logger = logging.getLogger(__name__)

# Absolute-path transcribes are bounded to keep a bad client from
# asking us to ingest a 20 GB file.
MAX_TRANSCRIBE_BYTES = 200 * 1024 * 1024  # 200 MB

# How long voicebox.speak_audio will poll before giving up on a generation
# that never reaches "completed"/"failed".
DEFAULT_SPEAK_AUDIO_TIMEOUT_SECONDS = 60.0


def register_tools(mcp: FastMCP) -> None:
    """Attach all Voicebox tools to the given FastMCP instance."""

    @mcp.tool(
        name="voicebox.speak",
        description=(
            "Speak text in a Voicebox voice profile. Returns a generation id "
            "the caller can poll at /generate/{id}/status. Audio plays on the "
            "user's speakers and is saved to the Captures / History tab. "
            "Requires a Voicebox browser tab (or the floating pill) to "
            "already be open to actually hear it — for headless/background "
            "callers with no Voicebox UI open, use `voicebox.speak_audio` "
            "instead, which waits for generation and returns the audio "
            "bytes directly."
        ),
    )
    async def voicebox_speak(
        text: str,
        profile: str | None = None,
        engine: str | None = None,
        personality: bool | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Speak ``text`` in a voice profile.

        ``profile`` accepts a voice profile name (e.g. "Morgan") or id. If
        omitted, the server looks up the per-client binding for the calling
        MCP client, then falls back to the global default voice.

        ``personality`` only matters for profiles that have a personality
        prompt — when true, the text is first rewritten in character by the
        LLM before TTS. When omitted, the per-client binding's
        ``default_personality`` flag decides; when that is unset, the
        default is plain TTS.
        """
        from ..database.models import MCPClientBinding

        db = next(get_db())
        try:
            client_id = current_client_id.get()
            vp = resolve_profile(profile, client_id, db)
            if vp is None:
                raise ValueError(
                    "No voice profile resolved. Pass `profile=` with a "
                    "voice profile name or id, or set a default voice in "
                    "Voicebox → Settings → MCP."
                )

            binding = None
            if client_id:
                binding = (
                    db.query(MCPClientBinding)
                    .filter(MCPClientBinding.client_id == client_id)
                    .first()
                )

            resolved_personality = personality
            if resolved_personality is None and binding is not None:
                resolved_personality = bool(binding.default_personality)

            resolved_engine = engine
            if resolved_engine is None and binding is not None:
                resolved_engine = binding.default_engine

            use_persona = bool(resolved_personality) and bool(vp.personality)
            return await _speak(
                profile_id=vp.id,
                profile_name=vp.name,
                text=text,
                engine=resolved_engine,
                language=language,
                personality=use_persona,
                db=db,
            )
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.speak_audio",
        description=(
            "Like voicebox.speak, but blocks until generation finishes and "
            "returns the result as base64-encoded audio bytes in the tool "
            "actual audio, instead of relying on an open Voicebox browser "
            "tab / floating pill to play it. Use this for headless or "
            "background callers (e.g. a chat agent with no Voicebox UI "
            "open) that need the audio itself — to play it, attach it to a "
            "message, save it, etc. Still fires the same speak-start event, "
            "so an open Voicebox tab will *also* play it and save it to "
            "Captures / History, same as voicebox.speak. Accepts the same "
            "`profile`/`engine`/`personality`/`language` parameters as "
            "voicebox.speak (see its description for details on those). "
            "Raises if the generation doesn't complete within "
            "`timeout_seconds` (default 60s) — pass a higher value for "
            "long text or a slow/cold-loading model. By default returns "
            "the audio as base64 (`audio_base64`, `audio_format`) in the "
            "tool result — note this can be tens of KB of text and may get "
            "truncated by clients/agents that cap tool-result size. "
            "Loopback callers can instead pass `output_path` (an absolute "
            "local path) to have the server write the file there directly "
            "and return just `saved_path` (no base64), which avoids that "
            "risk entirely for large audio."
        ),
    )
    async def voicebox_speak_audio(
        text: str,
        profile: str | None = None,
        engine: str | None = None,
        personality: bool | None = None,
        language: str | None = None,
        timeout_seconds: float = DEFAULT_SPEAK_AUDIO_TIMEOUT_SECONDS,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Like ``voicebox_speak``, but waits for the generation and returns
        the resulting audio inline instead of only a poll URL.

        See ``voicebox_speak`` for parameter semantics — this wraps the same
        ``_speak`` helper (identical profile/engine/personality resolution)
        and adds polling + reading the finished audio file off disk.

        ``output_path``, if given, must be an absolute path and is only
        honored for loopback callers (same restriction as ``audio_path`` on
        ``voicebox_transcribe``, for the same reason: a Voicebox bound on
        0.0.0.0 shouldn't double as a remote arbitrary-local-file-write
        primitive). When honored, the response omits ``audio_base64`` /
        ``audio_format`` entirely and returns ``saved_path`` instead.
        """
        from ..database.models import MCPClientBinding

        if timeout_seconds <= 0:
            raise ValueError("`timeout_seconds` must be > 0.")

        if output_path is not None:
            if not request_is_loopback():
                raise ValueError(
                    "`output_path` is only available to loopback callers — "
                    "remote callers must omit it and use the base64 "
                    "response instead."
                )
            if not Path(output_path).is_absolute():
                raise ValueError("`output_path` must be absolute.")

        db = next(get_db())
        try:
            client_id = current_client_id.get()
            vp = resolve_profile(profile, client_id, db)
            if vp is None:
                raise ValueError(
                    "No voice profile resolved. Pass `profile=` with a "
                    "voice profile name or id, or set a default voice in "
                    "Voicebox → Settings → MCP."
                )

            binding = None
            if client_id:
                binding = (
                    db.query(MCPClientBinding)
                    .filter(MCPClientBinding.client_id == client_id)
                    .first()
                )

            resolved_personality = personality
            if resolved_personality is None and binding is not None:
                resolved_personality = bool(binding.default_personality)

            resolved_engine = engine
            if resolved_engine is None and binding is not None:
                resolved_engine = binding.default_engine

            use_persona = bool(resolved_personality) and bool(vp.personality)
            speak_result = await _speak(
                profile_id=vp.id,
                profile_name=vp.name,
                text=text,
                engine=resolved_engine,
                language=language,
                personality=use_persona,
                db=db,
            )

            generation_id = speak_result.get("generation_id")
            if not generation_id:
                raise ValueError(
                    "Generation failed to start; no generation_id returned."
                )

            deadline = time.monotonic() + timeout_seconds
            gen = None
            while True:
                db.expire_all()
                gen = db.query(DBGeneration).filter_by(id=generation_id).first()
                status = (gen.status if gen else None) or "generating"
                if status == "completed":
                    break
                if status == "failed":
                    error_detail = gen.error if gen else "unknown error"
                    raise ValueError(
                        f"Generation {generation_id} failed: {error_detail}"
                    )
                if time.monotonic() >= deadline:
                    raise ValueError(
                        f"Generation {generation_id} did not complete within "
                        f"{timeout_seconds}s (last status: {status}). Pass a "
                        "higher `timeout_seconds` for long text or a "
                        "slow/cold-loading model."
                    )
                await asyncio.sleep(0.5)

            audio_path = config.resolve_storage_path(gen.audio_path)
            if audio_path is None or not audio_path.is_file():
                raise ValueError(
                    f"Generation {generation_id} completed but its audio "
                    "file is missing on disk."
                )

            if output_path is not None:
                dest = Path(output_path)

                def _copy_to_output_path() -> None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(audio_path.read_bytes())

                await asyncio.to_thread(_copy_to_output_path)
                return {
                    **speak_result,
                    "status": "completed",
                    "duration": gen.duration,
                    "saved_path": str(dest),
                }

            audio_bytes = await asyncio.to_thread(audio_path.read_bytes)

            return {
                **speak_result,
                "status": "completed",
                "duration": gen.duration,
                "audio_base64": b64.b64encode(audio_bytes).decode("ascii"),
                "audio_format": audio_path.suffix.lstrip("."),
            }
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.transcribe",
        description=(
            "Transcribe an audio clip to text using Voicebox's local Whisper. "
            "Pass exactly one of `audio_base64` (bytes as base64) or "
            "`audio_path` (absolute local file path — loopback callers only)."
        ),
    )
    async def voicebox_transcribe(
        audio_base64: str | None = None,
        audio_path: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        if bool(audio_base64) == bool(audio_path):
            raise ValueError(
                "Pass exactly one of `audio_base64` or `audio_path`."
            )

        # Absolute-path mode: validate and transcribe in place. Restricted
        # to loopback callers so a Voicebox bound on 0.0.0.0 doesn't double
        # as an unauthenticated arbitrary-local-file read primitive.
        if audio_path is not None:
            if not request_is_loopback():
                raise ValueError(
                    "`audio_path` is only available to loopback callers — "
                    "remote callers must use `audio_base64`."
                )
            path = Path(audio_path)
            if not path.is_absolute():
                raise ValueError("`audio_path` must be absolute.")
            if not path.is_file():
                raise ValueError(f"File not found: {audio_path}")
            if path.stat().st_size > MAX_TRANSCRIBE_BYTES:
                raise ValueError(
                    f"File exceeds {MAX_TRANSCRIBE_BYTES // (1024 * 1024)} MB limit."
                )
            return await _transcribe_file(path, language, model)

        # Base64 mode: decode into a temp file, transcribe, clean up.
        try:
            raw = b64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid audio_base64: {exc}") from exc
        if len(raw) > MAX_TRANSCRIBE_BYTES:
            raise ValueError(
                f"Audio exceeds {MAX_TRANSCRIBE_BYTES // (1024 * 1024)} MB limit."
            )
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            return await _transcribe_file(tmp_path, language, model)
        finally:
            tmp_path.unlink(missing_ok=True)

    @mcp.tool(
        name="voicebox.list_captures",
        description=(
            "List recent voice captures (dictations, recordings, uploads) "
            "with their transcripts. Most-recent first."
        ),
    )
    async def voicebox_list_captures(
        limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        if not (1 <= limit <= 200):
            raise ValueError("`limit` must be between 1 and 200.")
        if offset < 0:
            raise ValueError("`offset` must be >= 0.")
        db = next(get_db())
        try:
            items, total = captures_service.list_captures(
                db, limit=limit, offset=offset
            )
            return {
                "captures": [
                    item.model_dump(mode="json") for item in items
                ],
                "total": total,
            }
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.list_profiles",
        description=(
            "List available voice profiles (both cloned voices and presets). "
            "Use the returned `name` with voicebox.speak(profile=...)."
        ),
    )
    async def voicebox_list_profiles() -> dict[str, Any]:
        db = next(get_db())
        try:
            profiles = await profiles_service.list_profiles(db)
            return {
                "profiles": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "voice_type": p.voice_type,
                        "language": p.language,
                        "has_personality": bool(getattr(p, "personality", None)),
                    }
                    for p in profiles
                ]
            }
        finally:
            db.close()


# ─── Speak helper ──────────────────────────────────────────────────────────


async def _speak(
    *,
    profile_id: str,
    profile_name: str,
    text: str,
    engine: str | None,
    language: str | None,
    personality: bool,
    db,
) -> dict[str, Any]:
    """Delegate to POST /generate — the route handles personality-rewrite
    internally when ``personality=true`` and the profile has a prompt."""
    from ..routes.generations import generate_speech

    req = models.GenerationRequest(
        profile_id=profile_id,
        text=text,
        language=language or "en",
        engine=engine,
        personality=personality,
    )
    generation = await generate_speech(req, db)
    return _speak_response(generation, profile_name, source="mcp")


def _speak_response(
    generation, profile_name: str, *, source: str
) -> dict[str, Any]:
    """Normalize a GenerationResponse into the MCP tool's return shape.

    Also fires a speak-start event so the DictateWindow pill surfaces
    the agent's speech. Speak-end is fired from run_generation's
    completion hook.
    """
    payload = generation.model_dump(mode="json") if hasattr(
        generation, "model_dump"
    ) else dict(generation)
    generation_id = payload.get("id")
    mcp_events.publish(
        "speak-start",
        {
            "generation_id": generation_id,
            "profile_name": profile_name,
            "source": source,
            "client_id": current_client_id.get(),
        },
    )
    return {
        "generation_id": generation_id,
        "status": payload.get("status"),
        "profile": profile_name,
        "source": source,
        "poll_url": f"/generate/{generation_id}/status"
        if generation_id
        else None,
    }


# ─── Transcribe helper ─────────────────────────────────────────────────────


async def _transcribe_file(
    path: Path, language: str | None, model: str | None
) -> dict[str, Any]:
    from ..backends import WHISPER_HF_REPOS
    from ..services import transcribe as transcribe_service
    from ..utils.audio import load_audio

    whisper = transcribe_service.get_whisper_model()
    model_size = model or whisper.model_size
    valid = list(WHISPER_HF_REPOS.keys())
    if model_size not in valid:
        raise ValueError(
            f"Invalid STT model '{model_size}'. Must be one of: {', '.join(valid)}"
        )

    # load_audio is sync; keep the event loop responsive.
    audio, sr = await asyncio.to_thread(load_audio, str(path))
    duration = len(audio) / sr

    if (
        not whisper.is_loaded() or whisper.model_size != model_size
    ) and not whisper._is_model_cached(model_size):
        raise ValueError(
            f"Whisper model '{model_size}' is not yet downloaded. Open "
            "Voicebox → Settings → Models to download it first."
        )

    text = await whisper.transcribe(str(path), language, model_size)
    return {
        "text": text,
        "duration": duration,
        "language": language,
        "model": model_size,
    }
