"""
Progress tracking for model downloads using Server-Sent Events.
"""

from typing import Optional, Callable, Dict, List
from fastapi.responses import StreamingResponse
import asyncio
import json
from datetime import datetime


class ProgressManager:
    """Manages download progress for multiple models."""
    
    def __init__(self):
        self._progress: Dict[str, Dict] = {}
        self._listeners: Dict[str, list] = {}
    
    def update_progress(
        self,
        model_name: str,
        current: int,
        total: int,
        filename: Optional[str] = None,
        status: str = "downloading",
    ):
        """
        Update progress for a model download.

        Args:
            model_name: Name of the model (e.g., "qwen-tts-1.7B", "whisper-base")
            current: Current bytes downloaded
            total: Total bytes to download
            filename: Current file being downloaded
            status: Status string (downloading, extracting, complete, error)
        """
        import logging
        logger = logging.getLogger(__name__)

        progress_pct = (current / total * 100) if total > 0 else 0

        self._progress[model_name] = {
            "model_name": model_name,
            "current": current,
            "total": total,
            "progress": progress_pct,
            "filename": filename,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

        # Notify all listeners
        listener_count = len(self._listeners.get(model_name, []))
        if listener_count > 0:
            logger.debug(f"Notifying {listener_count} listeners for {model_name}: {progress_pct:.1f}% ({filename})")
            for queue in self._listeners[model_name]:
                try:
                    queue.put_nowait(self._progress[model_name].copy())
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for {model_name}, dropping update")
        else:
            logger.debug(f"No listeners for {model_name}, progress update stored: {progress_pct:.1f}%")
    
    def get_progress(self, model_name: str) -> Optional[Dict]:
        """Get current progress for a model."""
        return self._progress.get(model_name)
    
    def get_all_active(self) -> List[Dict]:
        """Get all active downloads (status is 'downloading' or 'extracting')."""
        active = []
        for model_name, progress in self._progress.items():
            status = progress.get("status", "")
            if status in ("downloading", "extracting"):
                active.append(progress.copy())
        return active
    
    def create_progress_callback(self, model_name: str, filename: Optional[str] = None):
        """
        Create a progress callback function for HuggingFace downloads.
        
        Args:
            model_name: Name of the model
            filename: Optional filename filter
            
        Returns:
            Callback function
        """
        def callback(progress: Dict):
            """HuggingFace Hub progress callback."""
            if "total" in progress and "current" in progress:
                current = progress.get("current", 0)
                total = progress.get("total", 0)
                file_name = progress.get("filename", filename)
                
                self.update_progress(
                    model_name=model_name,
                    current=current,
                    total=total,
                    filename=file_name,
                    status="downloading",
                )
        
        return callback
    
    async def subscribe(self, model_name: str):
        """
        Subscribe to progress updates for a model.

        Yields progress updates as Server-Sent Events.
        """
        import logging
        logger = logging.getLogger(__name__)

        queue = asyncio.Queue(maxsize=10)

        # Add to listeners
        if model_name not in self._listeners:
            self._listeners[model_name] = []
        self._listeners[model_name].append(queue)

        logger.info(f"SSE client subscribed to {model_name}, total listeners: {len(self._listeners[model_name])}")

        try:
            # Send initial progress if available
            if model_name in self._progress:
                logger.info(f"Sending initial progress for {model_name}: {self._progress[model_name].get('status')}")
                yield f"data: {json.dumps(self._progress[model_name])}\n\n"
            else:
                logger.info(f"No initial progress available for {model_name}")

            # Stream updates
            while True:
                try:
                    # Wait for update with timeout
                    progress = await asyncio.wait_for(queue.get(), timeout=1.0)
                    logger.debug(f"Sending progress update for {model_name}: {progress.get('status')} - {progress.get('progress', 0):.1f}%")
                    yield f"data: {json.dumps(progress)}\n\n"

                    # Stop if complete or error
                    if progress.get("status") in ("complete", "error"):
                        logger.info(f"Download {progress.get('status')} for {model_name}, closing SSE connection")
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
                    continue
        finally:
            # Remove from listeners
            if model_name in self._listeners:
                self._listeners[model_name].remove(queue)
                if not self._listeners[model_name]:
                    del self._listeners[model_name]
                logger.info(f"SSE client unsubscribed from {model_name}, remaining listeners: {len(self._listeners.get(model_name, []))}")
    
    def mark_complete(self, model_name: str):
        """Mark a model download as complete."""
        import logging
        logger = logging.getLogger(__name__)

        if model_name in self._progress:
            self._progress[model_name]["status"] = "complete"
            self._progress[model_name]["progress"] = 100.0
            logger.info(f"Marked {model_name} as complete")
            # Notify listeners
            if model_name in self._listeners:
                for queue in self._listeners[model_name]:
                    try:
                        queue.put_nowait(self._progress[model_name].copy())
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full when marking {model_name} complete")
    
    def mark_error(self, model_name: str, error: str):
        """Mark a model download as failed."""
        import logging
        logger = logging.getLogger(__name__)

        if model_name in self._progress:
            self._progress[model_name]["status"] = "error"
            self._progress[model_name]["error"] = error
            logger.error(f"Marked {model_name} as error: {error}")
            # Notify listeners
            if model_name in self._listeners:
                for queue in self._listeners[model_name]:
                    try:
                        queue.put_nowait(self._progress[model_name].copy())
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full when marking {model_name} error")


# Global progress manager instance
_progress_manager: Optional[ProgressManager] = None


def get_progress_manager() -> ProgressManager:
    """Get or create the global progress manager."""
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager
