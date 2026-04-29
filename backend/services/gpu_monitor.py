"""
GPU auto-offload monitor — tracks backend idle time and unloads models
that haven't been used for a configurable duration.
"""

import asyncio
import gc
import logging
import os
import time
from typing import Any

from ..backends.base import empty_device_cache

logger = logging.getLogger(__name__)


class IdleTracker:
    """Proxy wrapper around a backend that updates last_used_at on every call."""

    def __init__(self, backend: Any, engine: str) -> None:
        self._backend = backend
        self._engine = engine
        self.last_used_at: float = time.time()
        self.is_generating: bool = False

    # -- tracked methods --

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.last_used_at = time.time()
        return await self._backend.generate(*args, **kwargs)

    async def transcribe(self, *args: Any, **kwargs: Any) -> Any:
        self.last_used_at = time.time()
        return await self._backend.transcribe(*args, **kwargs)

    async def create_voice_prompt(self, *args: Any, **kwargs: Any) -> Any:
        self.last_used_at = time.time()
        return await self._backend.create_voice_prompt(*args, **kwargs)

    # -- transparent attribute forwarding --

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def __repr__(self) -> str:
        return f"IdleTracker({self._engine})"


class GPUMonitor:
    """Background asyncio sweep that auto-unloads idle GPU models."""

    def __init__(
        self,
        idle_timeout: int | None = None,
        sweep_interval: int | None = None,
    ) -> None:
        self.idle_timeout = idle_timeout or int(os.environ.get("VOICEBOX_GPU_IDLE_TIMEOUT_SECONDS", "600"))
        self.sweep_interval = sweep_interval or int(os.environ.get("VOICEBOX_GPU_SWEEP_INTERVAL_SECONDS", "30"))
        self._sweep_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background sweep loop as an asyncio task."""
        if self._sweep_task is not None and not self._sweep_task.done():
            logger.warning("GPUMonitor already running")
            return
        self._sweep_task = asyncio.create_task(self._sweep_loop())
        logger.info(
            "GPU monitor started (idle_timeout=%ds, sweep_interval=%ds)",
            self.idle_timeout,
            self.sweep_interval,
        )

    async def stop(self) -> None:
        """Cancel the sweep task cleanly."""
        if self._sweep_task is not None and not self._sweep_task.done():
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None
            logger.info("GPU monitor stopped")

    async def _sweep_loop(self) -> None:
        """Every sweep_interval seconds, check backends and unload idle ones."""
        # Lazy import to avoid circular dependency
        from .. import backends as backends_module

        while True:
            await asyncio.sleep(self.sweep_interval)

            # Check TTS backends
            for engine, backend in list(backends_module._tts_backends.items()):
                await self._check_backend(backend, engine)

            # Check STT backend
            stt = getattr(backends_module, "_stt_backend", None)
            if stt is not None:
                await self._check_backend(stt, "stt")

    async def _check_backend(self, backend: Any, engine: str) -> None:
        """Check a single backend for idle timeout and unload if eligible."""
        # Graceful fallback: skip if idle tracking flags aren't present
        last_used = getattr(backend, "last_used_at", None)
        is_generating = getattr(backend, "is_generating", None)

        if last_used is None:
            # No idle tracking — don't auto-unload
            return

        idle_seconds = time.time() - last_used

        if idle_seconds <= self.idle_timeout:
            return

        if is_generating:
            return

        if not getattr(backend, "is_loaded", lambda: False)():
            return

        display_name = _engine_display_name(engine)
        try:
            backend.unload_model()
            device = getattr(backend, "device", "cpu")
            empty_device_cache(device)
            gc.collect()
            logger.info(
                "Auto-unloaded %s after %.0fs idle",
                display_name,
                idle_seconds,
            )
        except Exception:
            logger.exception("Failed to auto-unload %s", display_name)

    def get_loaded_models_status(self) -> list[dict]:
        """Return status of all loaded models."""
        from .. import backends as backends_module

        results: list[dict] = []

        for engine, backend in backends_module._tts_backends.items():
            if not getattr(backend, "is_loaded", lambda: False)():
                continue
            last_used = getattr(backend, "last_used_at", None)
            idle = (time.time() - last_used) if last_used is not None else None
            results.append(
                {
                    "engine": _engine_display_name(engine),
                    "model": getattr(backend, "model_size", "unknown"),
                    "last_used_seconds_ago": round(idle, 1) if idle is not None else None,
                    "loaded": True,
                }
            )

        stt = getattr(backends_module, "_stt_backend", None)
        if stt is not None and getattr(stt, "is_loaded", lambda: False)():
            last_used = getattr(stt, "last_used_at", None)
            idle = (time.time() - last_used) if last_used is not None else None
            results.append(
                {
                    "engine": "Whisper",
                    "model": getattr(stt, "model_size", "unknown"),
                    "last_used_seconds_ago": round(idle, 1) if idle is not None else None,
                    "loaded": True,
                }
            )

        return results


def _engine_display_name(engine: str) -> str:
    """Map engine name to human-readable display name."""
    mapping = {
        "qwen": "Qwen TTS",
        "qwen_custom_voice": "Qwen CustomVoice",
        "luxtts": "LuxTTS",
        "chatterbox_turbo": "Chatterbox Turbo",
        "kokoro": "Kokoro",
        "stt": "Whisper STT",
    }
    return mapping.get(engine, engine)


# -- Singleton --

_gpu_monitor: GPUMonitor | None = None


def get_gpu_monitor() -> GPUMonitor:
    """Return the singleton GPUMonitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor


def wrap_backend_with_idle_tracking(backend: Any, engine: str) -> Any:
    """Wrap a backend with IdleTracker proxy for idle-time monitoring."""
    return IdleTracker(backend, engine)


def mark_generating(backend: Any, is_active: bool) -> None:
    """Set the is_generating flag on a backend to prevent mid-generation unload."""
    backend.is_generating = is_active
