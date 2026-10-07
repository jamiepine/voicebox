"""Idle model unloading — free VRAM when no generation has run for a while.

Models load on demand at generate time but are only ever unloaded explicitly
(POST /models/unload) or at shutdown, so an idle instance pins its weights
indefinitely — ~5.6GB for qwen-tts-1.7B, which is most of an 8GB card.

This reaper unloads after a period of inactivity. It is OFF by default; set
VOICEBOX_IDLE_UNLOAD_SECONDS to a positive number of seconds to enable it.

Safety: idleness is read from the serial generation queue's own state, so a
model is never pulled out from under queued or in-flight work. The next request
simply reloads the model (the same cold-start cost already paid on first use).
"""

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

# Seconds of inactivity before unloading. <= 0 disables the reaper entirely.
IDLE_UNLOAD_SECONDS = int(os.environ.get("VOICEBOX_IDLE_UNLOAD_SECONDS", "0") or 0)

# How often to check. Kept well under the timeout so the granularity is decent.
_POLL_SECONDS = 15

_last_activity: float = time.monotonic()


def mark_activity() -> None:
    """Record generation activity, resetting the idle countdown."""
    global _last_activity
    _last_activity = time.monotonic()


def _is_busy() -> bool:
    """True if any generation is queued or running."""
    from . import task_queue

    return bool(
        task_queue._queued_generation_ids or task_queue._running_generation_tasks
    )


def _anything_loaded() -> bool:
    """True if any model _unload_all() frees is currently resident.

    Checks every backend type, not just TTS: dictation loads Whisper and chat
    loads the LLM without ever touching TTS, and those hold VRAM too. Each
    getter is isolated so one failing backend can't mask a loaded sibling —
    on error we assume loaded, letting _unload_all() (which isolates its own
    failures) sort it out rather than pinning the weights forever.
    """
    from ..backends import get_llm_backend, get_stt_backend, get_tts_backend

    for name, get_backend in (
        ("TTS", get_tts_backend),
        ("Whisper", get_stt_backend),
        ("LLM", get_llm_backend),
    ):
        try:
            if get_backend().is_loaded():
                return True
        except Exception:
            logger.exception("Idle unload: failed to check %s backend", name)
            return True

    return False


def _unload_all() -> None:
    """Unload every model type, isolating failures so one can't block the others
    (mirrors the shutdown path in app.py)."""
    from . import llm, transcribe, tts

    for name, unload in (
        ("TTS", tts.unload_tts_model),
        ("Whisper", transcribe.unload_whisper_model),
        ("LLM", llm.unload_llm_model),
    ):
        try:
            unload()
        except Exception:
            logger.exception("Idle unload: failed to unload %s model", name)


async def _reaper() -> None:
    while True:
        await asyncio.sleep(_POLL_SECONDS)

        # Busy work keeps the model alive and re-arms the countdown, so a long
        # generation can never be interrupted by the reaper.
        if _is_busy():
            mark_activity()
            continue

        idle_for = time.monotonic() - _last_activity
        if idle_for < IDLE_UNLOAD_SECONDS:
            continue

        if not _anything_loaded():
            continue  # nothing loaded — nothing to do

        logger.info("Idle for %ds — unloading models to free memory", int(idle_for))
        await asyncio.to_thread(_unload_all)
        mark_activity()  # don't re-run every poll while still idle


def start() -> None:
    """Start the idle reaper if enabled. Call once at startup."""
    if IDLE_UNLOAD_SECONDS <= 0:
        return

    from .task_queue import create_background_task

    mark_activity()
    create_background_task(_reaper())
    logger.info("Idle model unloading enabled (%ds)", IDLE_UNLOAD_SECONDS)
