"""
Progress tracking for audio generation using Server-Sent Events.
"""

import asyncio
from datetime import datetime
import json
import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger("voicebox.generation-progress")


class GenerationProgressManager:
    """Manages audio generation progress events across long-form synthesis.

    Thread-safe: can be updated from worker threads and subscribed to from async endpoints.
    """

    THROTTLE_INTERVAL_SECONDS = 0.2
    THROTTLE_PROGRESS_DELTA = 1.0

    def __init__(self):
        self._progress: Dict[str, Dict] = {}
        self._listeners: Dict[str, list] = {}
        self._lock = threading.Lock()
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_notify_time: Dict[str, float] = {}
        self._last_notify_progress: Dict[str, float] = {}
        self._last_notify_chunk: Dict[str, Optional[int]] = {}

    def _notify_listeners_threadsafe(self, generation_id: str, progress_data: Dict):
        with self._lock:
            listeners = list(self._listeners.get(generation_id, []))

        if not listeners:
            return

        for queue in listeners:
            try:
                try:
                    asyncio.get_running_loop()
                    queue.put_nowait(progress_data.copy())
                except RuntimeError:
                    if self._main_loop and self._main_loop.is_running():
                        self._main_loop.call_soon_threadsafe(
                            lambda q=queue, d=progress_data.copy(): q.put_nowait(d) if not q.full() else None
                        )
            except asyncio.QueueFull:
                logger.warning("Queue full for generation %s, dropping progress update", generation_id)
            except Exception as e:
                logger.warning("Error notifying listener for generation %s: %s", generation_id, e)

    def update_progress(
        self,
        generation_id: str,
        progress: float,
        current_chunk: Optional[int] = None,
        total_chunks: Optional[int] = None,
        status: str = "generating",
        message: Optional[str] = None,
    ):
        """Update generation progress and notify SSE listeners.

        Clamps percentage between 0 and 100. Logs structured outputs to logger.
        """
        import time

        progress_pct = min(100.0, max(0.0, float(progress)))

        progress_data = {
            "id": generation_id,
            "status": status,
            "progress": round(progress_pct, 1),
            "current_chunk": current_chunk,
            "total_chunks": total_chunks,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        with self._lock:
            self._progress[generation_id] = progress_data

        current_time = time.time()
        last_time = self._last_notify_time.get(generation_id, 0)
        last_progress = self._last_notify_progress.get(generation_id, -100)
        last_chunk = self._last_notify_chunk.get(generation_id)

        time_delta = current_time - last_time
        progress_delta = abs(progress_pct - last_progress)
        chunk_changed = current_chunk is not None and current_chunk != last_chunk

        should_notify = (
            status in ("completed", "failed")
            or chunk_changed
            or time_delta >= self.THROTTLE_INTERVAL_SECONDS
            or progress_delta >= self.THROTTLE_PROGRESS_DELTA
        )

        if should_notify:
            self._last_notify_time[generation_id] = current_time
            self._last_notify_progress[generation_id] = progress_pct
            self._last_notify_chunk[generation_id] = current_chunk

            chunk_info = (
                f" ({current_chunk}/{total_chunks} chunks)"
                if current_chunk is not None and total_chunks is not None
                else ""
            )
            logger.info(
                "[Generation %s] Status: %s - Progress: %.1f%%%s",
                generation_id,
                status,
                progress_pct,
                chunk_info,
            )
            self._notify_listeners_threadsafe(generation_id, progress_data)

    def get_progress(self, generation_id: str) -> Optional[Dict]:
        """Get current progress for a generation. Thread-safe."""
        with self._lock:
            progress = self._progress.get(generation_id)
            return progress.copy() if progress else None

    async def subscribe(self, generation_id: str):
        """Subscribe to SSE progress updates for a generation."""
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        queue = asyncio.Queue(maxsize=20)

        with self._lock:
            if generation_id not in self._listeners:
                self._listeners[generation_id] = []
            self._listeners[generation_id].append(queue)
            initial_progress = self._progress.get(generation_id)
            if initial_progress:
                initial_progress = initial_progress.copy()

        try:
            if initial_progress:
                yield f"data: {json.dumps(initial_progress)}\n\n"

            while True:
                try:
                    progress = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(progress)}\n\n"

                    if progress.get("status") in ("completed", "failed"):
                        break
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("SSE client disconnected from generation %s", generation_id)
        finally:
            with self._lock:
                if generation_id in self._listeners:
                    if queue in self._listeners[generation_id]:
                        self._listeners[generation_id].remove(queue)
                    if not self._listeners[generation_id]:
                        del self._listeners[generation_id]

    def mark_complete(self, generation_id: str):
        """Mark a generation as complete. Thread-safe."""
        with self._lock:
            if generation_id in self._progress:
                self._progress[generation_id]["status"] = "completed"
                self._progress[generation_id]["progress"] = 100.0
                progress_data = self._progress[generation_id].copy()
            else:
                progress_data = {
                    "id": generation_id,
                    "status": "completed",
                    "progress": 100.0,
                    "timestamp": datetime.now().isoformat(),
                }
                self._progress[generation_id] = progress_data

        logger.info("[Generation %s] Completed (100%%)", generation_id)
        self._notify_listeners_threadsafe(generation_id, progress_data)

    def mark_error(self, generation_id: str, error: str):
        """Mark a generation as failed. Thread-safe."""
        with self._lock:
            if generation_id in self._progress:
                self._progress[generation_id]["status"] = "failed"
                self._progress[generation_id]["progress"] = 0.0
                self._progress[generation_id]["error"] = error
                progress_data = self._progress[generation_id].copy()
            else:
                progress_data = {
                    "id": generation_id,
                    "status": "failed",
                    "progress": 0.0,
                    "error": error,
                    "timestamp": datetime.now().isoformat(),
                }
                self._progress[generation_id] = progress_data

        logger.error("[Generation %s] Failed: %s", generation_id, error)
        self._notify_listeners_threadsafe(generation_id, progress_data)

    def clear(self, generation_id: str):
        """Clean up progress tracking for a generation."""
        with self._lock:
            self._progress.pop(generation_id, None)
            self._last_notify_time.pop(generation_id, None)
            self._last_notify_progress.pop(generation_id, None)
            self._last_notify_chunk.pop(generation_id, None)


_generation_progress_manager: Optional[GenerationProgressManager] = None
_manager_lock = threading.Lock()


def get_generation_progress_manager() -> GenerationProgressManager:
    """Get or create global generation progress manager (thread-safe singleton)."""
    global _generation_progress_manager
    if _generation_progress_manager is None:
        with _manager_lock:
            if _generation_progress_manager is None:
                _generation_progress_manager = GenerationProgressManager()
    return _generation_progress_manager
