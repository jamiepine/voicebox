"""CUDA backend management endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..services.task_queue import create_background_task
from ..utils.progress import get_progress_manager

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/backend/cuda-status")
async def get_cuda_status():
    """Get CUDA backend download/availability status."""
    from ..services import cuda

    return cuda.get_cuda_status()


@router.post("/backend/download-cuda")
async def download_cuda_backend():
    """Download the CUDA backend binary."""
    from ..services import cuda

    if cuda.get_cuda_binary_path() is not None:
        raise HTTPException(status_code=409, detail="CUDA backend already downloaded")

    progress_manager = get_progress_manager()
    existing = progress_manager.get_progress(cuda.PROGRESS_KEY)
    if existing and existing.get("status") == "downloading":
        raise HTTPException(status_code=409, detail="CUDA backend download already in progress")

    async def _download():
        try:
            await cuda.download_cuda_binary()
        except Exception as e:
            logger.error("CUDA download failed: %s", e)

    create_background_task(_download())
    return {"message": "CUDA backend download started", "progress_key": "cuda-backend"}


@router.delete("/backend/cuda")
async def delete_cuda_backend():
    """Delete the downloaded CUDA backend binary."""
    from ..services import cuda

    if cuda.is_cuda_active():
        raise HTTPException(
            status_code=409,
            detail="Cannot delete CUDA backend while it is active. Switch to CPU first.",
        )

    deleted = await cuda.delete_cuda_binary()
    if not deleted:
        raise HTTPException(status_code=404, detail="No CUDA backend found to delete")

    return {"message": "CUDA backend deleted"}


@router.get("/backend/cuda-progress")
async def get_cuda_download_progress():
    """Get CUDA backend download progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe("cuda-backend"):
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
