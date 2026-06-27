"""Transcription endpoints."""

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)

from .. import models
from ..services import transcribe
from ..services.task_queue import create_background_task
from ..utils.tasks import get_task_manager

router = APIRouter()

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


@router.post("/transcribe", response_model=models.TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    model: str | None = Form(None),
):
    """Transcribe audio file to text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        from ..utils.audio import load_audio
        from ..backends import STT_HF_REPOS, normalize_stt_model_name

        audio, sr = await asyncio.to_thread(load_audio, tmp_path)
        duration = len(audio) / sr

        stt_model = transcribe.get_stt_model()

        # Resolve model: explicit param > user's saved setting > backend default
        if not model:
            from ..services import settings as settings_service
            from ..database import get_db
            db = next(get_db())
            saved = settings_service.get_capture_settings(db)
            model = saved.stt_model if saved else None
        model_name = normalize_stt_model_name(model or stt_model.model_name)

        valid_models = list(STT_HF_REPOS.keys())
        if model_name not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model '{model_name}'. Must be one of: {', '.join(valid_models)}",
            )

        already_loaded = stt_model.is_loaded() and stt_model.model_name == model_name
        if not already_loaded and not stt_model._is_model_cached(model_name):
            task_manager = get_task_manager()

            async def download_stt_background():
                try:
                    await stt_model.load_model_async(model_name)
                    task_manager.complete_download(model_name)
                except Exception as e:
                    task_manager.error_download(model_name, str(e))

            task_manager.start_download(model_name)
            create_background_task(download_stt_background())

            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"STT model {model_name} is being downloaded. Please wait and try again.",
                    "model_name": model_name,
                    "downloading": True,
                },
            )

        text = await stt_model.transcribe(tmp_path, language, model_name)

        return models.TranscriptionResponse(
            text=text,
            duration=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
