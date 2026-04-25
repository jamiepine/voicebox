"""ROCm backend management endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..services.task_queue import create_background_task
from ..utils.progress import get_progress_manager

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/backend/rocm-status")
async def get_rocm_status():
    """Get ROCm backend download/availability status."""
    from ..services import rocm

    return rocm.get_rocm_status()


@router.post("/backend/download-rocm")
async def download_rocm_backend():
    """Download the ROCm backend binary."""
    from ..services import rocm

    progress_manager = get_progress_manager()
    existing = progress_manager.get_progress(rocm.PROGRESS_KEY)
    if existing and existing.get("status") in {"downloading", "extracting"}:
        raise HTTPException(status_code=409, detail="ROCm backend download already in progress")

    async def _download():
        try:
            await rocm.download_rocm_binary()
        except Exception as e:
            logger.error("ROCm download failed: %s", e)

    create_background_task(_download())
    return {"message": "ROCm backend download started", "progress_key": rocm.PROGRESS_KEY}


@router.delete("/backend/rocm")
async def delete_rocm_backend():
    """Delete the downloaded ROCm backend binary."""
    from ..services import rocm

    if rocm.is_rocm_active():
        raise HTTPException(
            status_code=409,
            detail="Cannot delete ROCm backend while it is active. Switch to CPU first.",
        )

    deleted = await rocm.delete_rocm_binary()
    if not deleted:
        raise HTTPException(status_code=404, detail="No ROCm backend found to delete")

    return {"message": "ROCm backend deleted"}


@router.get("/backend/rocm-progress")
async def get_rocm_download_progress():
    """Get ROCm backend download progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe("rocm-backend"):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
