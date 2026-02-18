"""
Task tracking for active downloads and generations.
"""

import asyncio
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class DownloadTask:
    """Represents an active download task."""
    model_name: str
    status: str = "downloading"  # downloading, extracting, complete, error
    started_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    asyncio_task: Optional[asyncio.Task] = field(default=None, repr=False)


@dataclass
class GenerationTask:
    """Represents an active generation task."""
    task_id: str
    profile_id: str
    text_preview: str  # First 50 chars of text
    started_at: datetime = field(default_factory=datetime.utcnow)
    progress: float = 0.0  # 0-100


class TaskManager:
    """Manages active downloads and generations."""

    def __init__(self):
        self._active_downloads: Dict[str, DownloadTask] = {}
        self._active_generations: Dict[str, GenerationTask] = {}

    def start_download(self, model_name: str, asyncio_task: Optional[asyncio.Task] = None) -> None:
        """Mark a download as started. Does not overwrite an existing entry (preserves asyncio_task)."""
        if model_name in self._active_downloads:
            return
        self._active_downloads[model_name] = DownloadTask(
            model_name=model_name,
            status="downloading",
            asyncio_task=asyncio_task,
        )

    def set_download_task(self, model_name: str, asyncio_task: asyncio.Task) -> None:
        """Attach the asyncio task to an existing download."""
        if model_name in self._active_downloads:
            self._active_downloads[model_name].asyncio_task = asyncio_task

    def cancel_download(self, model_name: str) -> bool:
        """Cancel an active download. Returns True if cancelled."""
        task = self._active_downloads.get(model_name)
        if task and task.asyncio_task:
            task.asyncio_task.cancel()
            del self._active_downloads[model_name]
            return True
        if task:
            del self._active_downloads[model_name]
            return True
        return False

    def complete_download(self, model_name: str) -> None:
        """Mark a download as complete."""
        if model_name in self._active_downloads:
            del self._active_downloads[model_name]

    def error_download(self, model_name: str, error: str) -> None:
        """Mark a download as failed and remove from active downloads."""
        if model_name in self._active_downloads:
            del self._active_downloads[model_name]

    def start_generation(self, task_id: str, profile_id: str, text: str) -> None:
        """Mark a generation as started."""
        text_preview = text[:50] + "..." if len(text) > 50 else text
        self._active_generations[task_id] = GenerationTask(
            task_id=task_id,
            profile_id=profile_id,
            text_preview=text_preview,
        )

    def update_generation_progress(self, task_id: str, progress: float) -> None:
        """Update generation progress."""
        if task_id in self._active_generations:
            self._active_generations[task_id].progress = progress

    def complete_generation(self, task_id: str) -> None:
        """Mark a generation as complete."""
        if task_id in self._active_generations:
            del self._active_generations[task_id]

    def get_active_downloads(self) -> List[DownloadTask]:
        """Get all active downloads."""
        return list(self._active_downloads.values())

    def get_active_generations(self) -> List[GenerationTask]:
        """Get all active generations."""
        return list(self._active_generations.values())

    def is_download_active(self, model_name: str) -> bool:
        """Check if a download is active."""
        return model_name in self._active_downloads

    def is_generation_active(self, task_id: str) -> bool:
        """Check if a generation is active."""
        return task_id in self._active_generations


# Global task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
