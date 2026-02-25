"""
Task tracking for active downloads and generations.
"""

from typing import Optional, Dict, List
from datetime import datetime, timezone
from dataclasses import dataclass, field


@dataclass
class DownloadTask:
    """Represents an active download task."""
    model_name: str
    status: str = "downloading"  # downloading, extracting, complete, error
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


@dataclass
class GenerationTask:
    """Represents an active generation task."""
    task_id: str
    profile_id: str
    text_preview: str  # First 50 chars of text
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FinetuneTask:
    """Represents an active finetune training task."""
    job_id: str
    profile_id: str
    status: str = "pending"  # pending|preparing|training|completed|failed|cancelled
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    current_loss: Optional[float] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TaskManager:
    """Manages active downloads, generations, and finetune tasks."""

    def __init__(self):
        self._active_downloads: Dict[str, DownloadTask] = {}
        self._active_generations: Dict[str, GenerationTask] = {}
        self._active_finetunes: Dict[str, FinetuneTask] = {}
    
    def start_download(self, model_name: str) -> None:
        """Mark a download as started."""
        self._active_downloads[model_name] = DownloadTask(
            model_name=model_name,
            status="downloading",
        )
    
    def complete_download(self, model_name: str) -> None:
        """Mark a download as complete."""
        if model_name in self._active_downloads:
            del self._active_downloads[model_name]
    
    def error_download(self, model_name: str, error: str) -> None:
        """Mark a download as failed."""
        if model_name in self._active_downloads:
            self._active_downloads[model_name].status = "error"
            self._active_downloads[model_name].error = error
    
    def start_generation(self, task_id: str, profile_id: str, text: str) -> None:
        """Mark a generation as started."""
        text_preview = text[:50] + "..." if len(text) > 50 else text
        self._active_generations[task_id] = GenerationTask(
            task_id=task_id,
            profile_id=profile_id,
            text_preview=text_preview,
        )
    
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

    # Finetune task management
    def start_finetune(self, job_id: str, profile_id: str, total_epochs: int = 0) -> None:
        """Mark a finetune task as started."""
        self._active_finetunes[job_id] = FinetuneTask(
            job_id=job_id,
            profile_id=profile_id,
            status="preparing",
            total_epochs=total_epochs,
        )

    def update_finetune(self, job_id: str, **kwargs) -> None:
        """Update finetune task progress."""
        if job_id in self._active_finetunes:
            task = self._active_finetunes[job_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

    def complete_finetune(self, job_id: str) -> None:
        """Mark a finetune task as complete."""
        if job_id in self._active_finetunes:
            del self._active_finetunes[job_id]

    def error_finetune(self, job_id: str, error: str) -> None:
        """Mark a finetune task as failed."""
        if job_id in self._active_finetunes:
            self._active_finetunes[job_id].status = "failed"

    def get_active_finetunes(self) -> List[FinetuneTask]:
        """Get all active finetune tasks."""
        return list(self._active_finetunes.values())

    def is_finetune_active(self) -> bool:
        """Check if any finetune task is active."""
        return len(self._active_finetunes) > 0


# Global task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
