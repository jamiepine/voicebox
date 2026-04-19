"""Task and cache management endpoints."""

from datetime import datetime

from fastapi import APIRouter

from .. import models
from ..utils.cache import clear_voice_prompt_cache
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager
from fastapi import HTTPException

router = APIRouter()


@router.post("/tasks/clear")
async def clear_all_tasks():
    """Clear all download tasks and progress state."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    task_manager.clear_all()

    with progress_manager._lock:
        progress_manager._progress.clear()
        progress_manager._last_notify_time.clear()
        progress_manager._last_notify_progress.clear()

    return {"message": "All task state cleared"}


@router.post("/cache/clear")
async def clear_cache():
    """Clear all voice prompt caches (memory and disk)."""
    try:
        deleted_count = clear_voice_prompt_cache()
        return {
            "message": "Voice prompt cache cleared successfully",
            "files_deleted": deleted_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/tasks/active", response_model=models.ActiveTasksResponse)
async def get_active_tasks():
    """Return all currently active downloads and generations."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    active_downloads = []
    task_manager_downloads = task_manager.get_active_downloads()
    progress_active = progress_manager.get_all_active()

    download_map = {task.model_name: task for task in task_manager_downloads}
    progress_map = {p["model_name"]: p for p in progress_active}

    all_model_names = set(download_map.keys()) | set(progress_map.keys())
    for model_name in all_model_names:
        task = download_map.get(model_name)
        progress = progress_map.get(model_name)

        if task:
            error = task.error
            if not error:
                with progress_manager._lock:
                    pm_data = progress_manager._progress.get(model_name)
                    if pm_data:
                        error = pm_data.get("error")
            prog = progress or {}
            if not prog:
                with progress_manager._lock:
                    pm_data = progress_manager._progress.get(model_name)
                    if pm_data:
                        prog = pm_data
            active_downloads.append(
                models.ActiveDownloadTask(
                    model_name=model_name,
                    status=task.status,
                    started_at=task.started_at,
                    error=error,
                    progress=prog.get("progress"),
                    current=prog.get("current"),
                    total=prog.get("total"),
                    filename=prog.get("filename"),
                )
            )
        elif progress:
            timestamp_str = progress.get("timestamp")
            if timestamp_str:
                try:
                    started_at = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    started_at = datetime.utcnow()
            else:
                started_at = datetime.utcnow()

            active_downloads.append(
                models.ActiveDownloadTask(
                    model_name=model_name,
                    status=progress.get("status", "downloading"),
                    started_at=started_at,
                    error=progress.get("error"),
                    progress=progress.get("progress"),
                    current=progress.get("current"),
                    total=progress.get("total"),
                    filename=progress.get("filename"),
                )
            )

    active_generations = []
    for gen_task in task_manager.get_active_generations():
        active_generations.append(
            models.ActiveGenerationTask(
                task_id=gen_task.task_id,
                profile_id=gen_task.profile_id,
                text_preview=gen_task.text_preview,
                started_at=gen_task.started_at,
            )
        )

    return models.ActiveTasksResponse(
        downloads=active_downloads,
        generations=active_generations,
    )
