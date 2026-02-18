"""
FastAPI application for voicebox backend.

Handles voice cloning, generation history, and server mode.
"""

import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import asyncio
import uvicorn
import argparse
import torch
import tempfile
import io
from pathlib import Path
import uuid
import signal
import os

from . import database, models, profiles, history, tts, transcribe, config, export_import, channels, stories, __version__
from .database import get_db, Generation as DBGeneration, GenerationJob as DBGenerationJob, VoiceProfile as DBVoiceProfile
from .utils.progress import get_progress_manager
from .utils.tasks import get_task_manager
from .utils.cache import clear_voice_prompt_cache
from .platform_detect import get_backend_type

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the app."""
    await _startup()
    yield
    await _shutdown()


app = FastAPI(
    title="voicebox API",
    description="Production-quality Qwen3-TTS voice cloning API",
    version=__version__,
    lifespan=lifespan,
)

# Lock to serialize TTS model access — MLX models are NOT thread-safe.
# Concurrent generate() calls on the same model instance corrupt state and crash.
_model_lock = asyncio.Lock()

# Event to wake the job worker when a new job is queued
_job_signal = asyncio.Event()
_cancel_requested_jobs: set[str] = set()
_MAX_ACTIVE_JOBS_PER_USER = 3
_QUEUED_JOB_TIMEOUT_MINUTES = 15
_GENERATING_JOB_TIMEOUT_MINUTES = 5


def _expire_old_queued_jobs(db: Session):
    """Expire queued jobs that have sat too long without starting."""
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(minutes=_QUEUED_JOB_TIMEOUT_MINUTES)
    stale_queued = db.query(DBGenerationJob).filter(
        DBGenerationJob.status == "queued",
        DBGenerationJob.created_at < cutoff,
    ).all()
    for job in stale_queued:
        job.status = "timeout"
        job.error = "Queue timeout"
        job.completed_at = datetime.utcnow()
        try:
            get_progress_manager().mark_error(job.id, "Queue timeout")
        except Exception:
            pass
    if stale_queued:
        db.commit()


def _extract_request_ip(request: Request) -> str:
    """Best-effort client IP extraction, including proxy headers."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"

# CORS middleware — allow_credentials=False because we don't use cookies,
# and allow_origins=["*"] is invalid with credentials per the CORS spec.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Health-Model-Loaded", "X-Health-Model-Size", "X-Health-GPU-Type", "X-Health-Backend"],
)


@app.middleware("http")
async def health_piggyback_middleware(request, call_next):
    """Attach lightweight health info as headers on every response."""
    response = await call_next(request)
    try:
        tts_model = tts.get_tts_model()
        loaded = tts_model.is_loaded()
        response.headers["X-Health-Model-Loaded"] = "1" if loaded else "0"
        if loaded:
            size = getattr(tts_model, '_current_model_size', None)
            if size:
                response.headers["X-Health-Model-Size"] = size
        backend_type = get_backend_type()
        response.headers["X-Health-Backend"] = backend_type
        gpu_type = None
        if backend_type == "mlx":
            gpu_type = "Metal"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_type = "MPS"
        elif torch.cuda.is_available():
            gpu_type = "CUDA"
        if gpu_type:
            response.headers["X-Health-GPU-Type"] = gpu_type
    except Exception:
        pass
    return response


# ============================================
# ROOT & HEALTH ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "voicebox API", "version": __version__}


@app.post("/shutdown")
async def shutdown():
    """Gracefully shutdown the server."""
    async def shutdown_async():
        await asyncio.sleep(0.1)  # Give response time to send
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(shutdown_async())
    return {"message": "Shutting down..."}


@app.get("/auth/me")
async def auth_me_stub():
    """Stub for chickenbox OAuth — voicebox backend has no auth."""
    raise HTTPException(status_code=401, detail="Authentication not available")


@app.get("/health", response_model=models.HealthResponse)
async def health():
    """Health check endpoint."""
    from huggingface_hub import constants as hf_constants
    from pathlib import Path

    tts_model = tts.get_tts_model()
    backend_type = get_backend_type()

    # Touch TTS idle timer — health polls indicate an active user session
    if tts_model.is_loaded():
        tts_model._idle_timer.touch()

    # Check for GPU availability (CUDA or MPS)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    gpu_available = has_cuda or has_mps

    gpu_type = None
    if has_cuda:
        gpu_type = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif has_mps:
        gpu_type = "MPS (Apple Silicon)"
    elif backend_type == "mlx":
        gpu_type = "Metal (Apple Silicon via MLX)"

    vram_used = None
    if has_cuda:
        vram_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    # Check if model is loaded - use the same logic as model status endpoint
    model_loaded = False
    model_size = None
    try:
        # Use the same check as model status endpoint
        if tts_model.is_loaded():
            model_loaded = True
            # Get the actual loaded model size
            # Check _current_model_size first (more reliable for actually loaded models)
            model_size = getattr(tts_model, '_current_model_size', None)
            if not model_size:
                # Fallback to model_size attribute (which should be set when model loads)
                model_size = getattr(tts_model, 'model_size', None)
    except Exception:
        # If there's an error checking, assume not loaded
        model_loaded = False
        model_size = None
    
    # Check if default model is downloaded (cached)
    model_downloaded = None
    try:
        # Check if the default model (1.7B) is cached
        # Use different model IDs based on backend
        if backend_type == "mlx":
            default_model_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
        else:
            default_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        
        # Method 1: Try scan_cache_dir if available
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == default_model_id:
                    model_downloaded = True
                    break
        except (ImportError, Exception):
            # Method 2: Check cache directory (using HuggingFace's OS-specific cache location)
            cache_dir = hf_constants.HF_HUB_CACHE
            repo_cache = Path(cache_dir) / ("models--" + default_model_id.replace("/", "--"))
            if repo_cache.exists():
                has_model_files = (
                    any(repo_cache.rglob("*.bin")) or
                    any(repo_cache.rglob("*.safetensors")) or
                    any(repo_cache.rglob("*.pt")) or
                    any(repo_cache.rglob("*.pth")) or
                    any(repo_cache.rglob("*.npz"))  # MLX models may use npz
                )
                model_downloaded = has_model_files
    except Exception:
        pass
    
    return models.HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_downloaded=model_downloaded,
        model_size=model_size,
        gpu_available=gpu_available,
        gpu_type=gpu_type,
        vram_used_mb=vram_used,
        backend_type=backend_type,
    )


# ============================================
# VOICE PROFILE ENDPOINTS
# ============================================

@app.post("/profiles", response_model=models.VoiceProfileResponse)
async def create_profile(
    data: models.VoiceProfileCreate,
    db: Session = Depends(get_db),
):
    """Create a new voice profile."""
    try:
        return await profiles.create_profile(data, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/profiles", response_model=List[models.VoiceProfileResponse])
async def list_profiles(db: Session = Depends(get_db)):
    """List all voice profiles."""
    return await profiles.list_profiles(db)


@app.post("/profiles/import", response_model=models.VoiceProfileResponse)
async def import_profile(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import a voice profile from a ZIP archive."""
    # Validate file size (max 100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Read file content
    content = await file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)}MB"
        )
    
    try:
        profile = await export_import.import_profile_from_zip(content, db)
        return profile
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profiles/{profile_id}", response_model=models.VoiceProfileResponse)
async def get_profile(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Get a voice profile by ID."""
    profile = await profiles.get_profile(profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@app.put("/profiles/{profile_id}", response_model=models.VoiceProfileResponse)
async def update_profile(
    profile_id: str,
    data: models.VoiceProfileCreate,
    db: Session = Depends(get_db),
):
    """Update a voice profile."""
    profile = await profiles.update_profile(profile_id, data, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@app.delete("/profiles/{profile_id}")
async def delete_profile(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Delete a voice profile."""
    success = await profiles.delete_profile(profile_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"message": "Profile deleted successfully"}


@app.post("/profiles/{profile_id}/samples", response_model=models.ProfileSampleResponse)
async def add_profile_sample(
    profile_id: str,
    file: UploadFile = File(...),
    reference_text: str = Form(...),
    db: Session = Depends(get_db),
):
    """Add a sample to a voice profile."""
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        sample = await profiles.add_profile_sample(
            profile_id,
            tmp_path,
            reference_text,
            db,
        )
        return sample
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/profiles/{profile_id}/samples", response_model=List[models.ProfileSampleResponse])
async def get_profile_samples(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Get all samples for a profile."""
    return await profiles.get_profile_samples(profile_id, db)


@app.delete("/profiles/samples/{sample_id}")
async def delete_profile_sample(
    sample_id: str,
    db: Session = Depends(get_db),
):
    """Delete a profile sample."""
    success = await profiles.delete_profile_sample(sample_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"message": "Sample deleted successfully"}


@app.put("/profiles/samples/{sample_id}", response_model=models.ProfileSampleResponse)
async def update_profile_sample(
    sample_id: str,
    data: models.ProfileSampleUpdate,
    db: Session = Depends(get_db),
):
    """Update a profile sample's reference text."""
    sample = await profiles.update_profile_sample(sample_id, data.reference_text, db)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample


@app.post("/profiles/{profile_id}/avatar", response_model=models.VoiceProfileResponse)
async def upload_profile_avatar(
    profile_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload or update avatar image for a profile."""
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        profile = await profiles.upload_avatar(profile_id, tmp_path, db)
        return profile
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/profiles/{profile_id}/avatar")
async def get_profile_avatar(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Get avatar image for a profile."""
    profile = await profiles.get_profile(profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if not profile.avatar_path:
        raise HTTPException(status_code=404, detail="No avatar found for this profile")

    avatar_path = Path(profile.avatar_path)
    if not avatar_path.exists():
        raise HTTPException(status_code=404, detail="Avatar file not found")

    return FileResponse(avatar_path)


@app.delete("/profiles/{profile_id}/avatar")
async def delete_profile_avatar(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Delete avatar image for a profile."""
    success = await profiles.delete_avatar(profile_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found or no avatar to delete")
    return {"message": "Avatar deleted successfully"}


@app.get("/profiles/{profile_id}/export")
async def export_profile(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Export a voice profile as a ZIP archive."""
    try:
        # Get profile to get name for filename
        profile = await profiles.get_profile(profile_id, db)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Export to ZIP
        zip_bytes = export_import.export_profile_to_zip(profile_id, db)
        
        # Create safe filename
        safe_name = "".join(c for c in profile.name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = "profile"
        filename = f"profile-{safe_name}.voicebox.zip"
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# AUDIO CHANNEL ENDPOINTS
# ============================================

@app.get("/channels", response_model=List[models.AudioChannelResponse])
async def list_channels(db: Session = Depends(get_db)):
    """List all audio channels."""
    return await channels.list_channels(db)


@app.post("/channels", response_model=models.AudioChannelResponse)
async def create_channel(
    data: models.AudioChannelCreate,
    db: Session = Depends(get_db),
):
    """Create a new audio channel."""
    try:
        return await channels.create_channel(data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/channels/{channel_id}", response_model=models.AudioChannelResponse)
async def get_channel(
    channel_id: str,
    db: Session = Depends(get_db),
):
    """Get an audio channel by ID."""
    channel = await channels.get_channel(channel_id, db)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return channel


@app.put("/channels/{channel_id}", response_model=models.AudioChannelResponse)
async def update_channel(
    channel_id: str,
    data: models.AudioChannelUpdate,
    db: Session = Depends(get_db),
):
    """Update an audio channel."""
    try:
        channel = await channels.update_channel(channel_id, data, db)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
        return channel
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/channels/{channel_id}")
async def delete_channel(
    channel_id: str,
    db: Session = Depends(get_db),
):
    """Delete an audio channel."""
    try:
        success = await channels.delete_channel(channel_id, db)
        if not success:
            raise HTTPException(status_code=404, detail="Channel not found")
        return {"message": "Channel deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/channels/{channel_id}/voices")
async def get_channel_voices(
    channel_id: str,
    db: Session = Depends(get_db),
):
    """Get list of profile IDs assigned to a channel."""
    try:
        profile_ids = await channels.get_channel_voices(channel_id, db)
        return {"profile_ids": profile_ids}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/channels/{channel_id}/voices")
async def set_channel_voices(
    channel_id: str,
    data: models.ChannelVoiceAssignment,
    db: Session = Depends(get_db),
):
    """Set which voices are assigned to a channel."""
    try:
        await channels.set_channel_voices(channel_id, data, db)
        return {"message": "Channel voices updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/profiles/{profile_id}/channels")
async def get_profile_channels(
    profile_id: str,
    db: Session = Depends(get_db),
):
    """Get list of channel IDs assigned to a profile."""
    try:
        channel_ids = await channels.get_profile_channels(profile_id, db)
        return {"channel_ids": channel_ids}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/profiles/{profile_id}/channels")
async def set_profile_channels(
    profile_id: str,
    data: models.ProfileChannelAssignment,
    db: Session = Depends(get_db),
):
    """Set which channels a profile is assigned to."""
    try:
        await channels.set_profile_channels(profile_id, data, db)
        return {"message": "Profile channels updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================
# GENERATION ENDPOINTS
# ============================================

@app.post("/generate")
async def generate_speech(
    data: models.GenerationRequest,
    request: Request,
    stream: bool = False,
    db: Session = Depends(get_db),
):
    """Generate speech from text using a voice profile.

    With stream=False (default): blocks and returns GenerationResponse.
    With stream=True: returns 202 with job_id, progress via SSE.
    """
    job_id = str(uuid.uuid4())
    request_ip = _extract_request_ip(request)

    if stream:
        _expire_old_queued_jobs(db)

        # Enforce per-user queue cap (queued/generating/cancelling).
        user_id = (data.request_user_id or "").strip() or None
        active_statuses = ["queued", "generating", "cancelling"]
        if user_id:
            active_count = db.query(DBGenerationJob).filter(
                DBGenerationJob.status.in_(active_statuses),
                DBGenerationJob.request_user_id == user_id,
            ).count()
        else:
            active_count = db.query(DBGenerationJob).filter(
                DBGenerationJob.status.in_(active_statuses),
                DBGenerationJob.request_ip == request_ip,
            ).count()

        if active_count >= _MAX_ACTIVE_JOBS_PER_USER:
            raise HTTPException(
                status_code=429,
                detail=f"Queue limit reached ({_MAX_ACTIVE_JOBS_PER_USER} active jobs per user).",
            )

        # --- Async (streaming) mode — create a job row, worker picks it up ---
        job = DBGenerationJob(
            id=job_id,
            profile_id=data.profile_id,
            text=data.text,
            language=data.language,
            seed=data.seed,
            model_size=data.model_size or "1.7B",
            instruct=data.instruct,
            request_user_id=data.request_user_id,
            request_user_first_name=data.request_user_first_name,
            request_ip=request_ip,
            status="queued",
        )
        db.add(job)
        db.commit()

        progress_manager = get_progress_manager()
        # Initialize progress state so SSE endpoint has data immediately
        progress_manager.update_progress(
            model_name=job_id,
            current=0,
            total=100,
            status="queued",
        )

        # Wake the worker
        _job_signal.set()

        logger.info(f"[TTS] Job {job_id} queued for profile {data.profile_id} from {request_ip}")
        return JSONResponse(
            status_code=202,
            content=models.GenerationStartResponse(
                generation_id=job_id,
                status="queued",
            ).model_dump(),
        )

    # --- Synchronous (blocking) mode — existing behavior for CLI/tests ---

    # Check DB for any actively generating job
    active = db.query(DBGenerationJob).filter(
        DBGenerationJob.status == "generating"
    ).first()
    if active or _model_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="A generation is already in progress. Please wait for it to finish.",
        )

    task_manager = get_task_manager()
    try:
        task_manager.start_generation(
            task_id=job_id,
            profile_id=data.profile_id,
            text=data.text,
        )

        profile = await profiles.get_profile(data.profile_id, db)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")

        voice_prompt = await profiles.create_voice_prompt_for_profile(
            data.profile_id, db,
        )

        generation_started_at = datetime.utcnow()
        async with _model_lock:
            tts_model = tts.get_tts_model()
            model_size = data.model_size or "1.7B"

            # Don't silently download — require the model to be cached first
            if not tts_model._is_model_cached(model_size):
                model_name = f"qwen-tts-{model_size}"
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {model_name} is not downloaded. Please download it first from the Models page.",
                )

            await tts_model.load_model_async(model_size)
            save_model_prefs(tts_size=model_size)
            audio, sample_rate = await tts_model.generate(
                data.text, voice_prompt, data.language, data.seed, data.instruct,
            )
        generation_time_seconds = (datetime.utcnow() - generation_started_at).total_seconds()

        duration = len(audio) / sample_rate
        audio_path = config.get_generations_dir() / f"{job_id}.wav"
        from .utils.audio import save_audio
        save_audio(audio, str(audio_path), sample_rate)

        generation = await history.create_generation(
            profile_id=data.profile_id,
            text=data.text,
            language=data.language,
            audio_path=str(audio_path),
            duration=duration,
            seed=data.seed,
            db=db,
            instruct=data.instruct,
            model_size=model_size,
            backend_type=get_backend_type(),
            request_user_id=data.request_user_id,
            request_user_first_name=data.request_user_first_name,
            request_ip=request_ip,
            generation_time_seconds=generation_time_seconds,
        )
        task_manager.complete_generation(job_id)
        return generation

    except ValueError as e:
        task_manager.complete_generation(job_id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        task_manager.complete_generation(job_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/progress/{generation_id}")
async def generation_progress(generation_id: str):
    """Stream generation progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe(generation_id):
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


@app.get("/generate/busy")
async def generation_busy(db: Session = Depends(get_db)):
    """Check if a generation is currently running (status=generating, not queued)."""
    active = db.query(DBGenerationJob).filter(
        DBGenerationJob.status.in_(["generating", "cancelling"])
    ).first()
    return {"busy": active is not None}


@app.get("/jobs/pending", response_model=List[models.GenerationJobResponse])
async def list_pending_jobs(db: Session = Depends(get_db)):
    """Return all queued and generating jobs, oldest first."""
    jobs = db.query(DBGenerationJob, DBVoiceProfile.name).join(
        DBVoiceProfile, DBGenerationJob.profile_id == DBVoiceProfile.id
    ).filter(
        DBGenerationJob.status.in_(["queued", "generating", "cancelling"])
    ).order_by(DBGenerationJob.created_at).all()

    return [
        models.GenerationJobResponse(
            id=job.id,
            profile_id=job.profile_id,
            profile_name=profile_name,
            text=job.text,
            language=job.language,
            model_size=job.model_size,
            backend_type=job.backend_type,
            request_user_id=job.request_user_id,
            request_user_first_name=job.request_user_first_name,
            request_ip=job.request_ip,
            status=job.status,
            progress=job.progress,
            generation_id=job.generation_id,
            instruct=job.instruct,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job, profile_name in jobs
    ]


@app.get("/jobs", response_model=List[models.GenerationJobResponse])
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default="queued,generating,cancelling,complete"),
    db: Session = Depends(get_db),
):
    """List jobs with optional status filter (comma-separated) and pagination. Default excludes deleted jobs."""
    query = db.query(DBGenerationJob, DBVoiceProfile.name, DBGeneration).outerjoin(
        DBVoiceProfile, DBGenerationJob.profile_id == DBVoiceProfile.id
    ).outerjoin(
        DBGeneration, DBGenerationJob.generation_id == DBGeneration.id
    ).filter(
        DBGenerationJob.status != "deleted"
    ).filter(
        (DBGenerationJob.status != "complete") | (DBGeneration.id.isnot(None))
    )

    if status:
        statuses = [s.strip() for s in status.split(",") if s.strip()]
        if statuses:
            query = query.filter(DBGenerationJob.status.in_(statuses))

    rows = query.order_by(DBGenerationJob.created_at.desc()).offset(offset).limit(limit).all()
    return [
        models.GenerationJobResponse(
            id=job.id,
            profile_id=job.profile_id,
            profile_name=profile_name or "Unknown",
            text=job.text,
            language=job.language,
            model_size=job.model_size,
            backend_type=job.backend_type or (generation.backend_type if generation else None),
            request_user_id=job.request_user_id,
            request_user_first_name=job.request_user_first_name,
            request_ip=job.request_ip,
            status=job.status,
            progress=job.progress,
            generation_id=job.generation_id,
            audio_path=generation.audio_path if generation else None,
            duration=generation.duration if generation else None,
            generation_time_seconds=generation.generation_time_seconds if generation else None,
            instruct=job.instruct or (generation.instruct if generation else None),
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job, profile_name, generation in rows
    ]


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a queued/generating job."""
    job = db.query(DBGenerationJob).filter(DBGenerationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    progress_manager = get_progress_manager()
    now = datetime.utcnow()

    if job.status == "queued":
        job.status = "cancelled"
        job.error = "Cancelled by user"
        job.completed_at = now
        db.commit()
        progress_manager.mark_error(job_id, "Cancelled by user")
        return {"status": "cancelled"}

    if job.status in ("generating", "cancelling"):
        _cancel_requested_jobs.add(job_id)
        job.status = "cancelling"
        db.commit()
        return {"status": "cancelling"}

    return {"status": job.status}


@app.post("/jobs/{job_id}/cancel/force")
async def force_cancel_job(job_id: str, db: Session = Depends(get_db)):
    """Force-cancel a job and unload the model backend."""
    _cancel_requested_jobs.add(job_id)

    job = db.query(DBGenerationJob).filter(DBGenerationJob.id == job_id).first()
    if job and job.status in ("queued", "generating", "cancelling"):
        job.status = "cancelled"
        job.error = "Force-cancelled by user"
        job.completed_at = datetime.utcnow()
        db.commit()

    # Best-effort hard stop for current backend.
    try:
        tts.unload_tts_model()
    except Exception:
        pass

    get_progress_manager().mark_error(job_id, "Force-cancelled by user")
    
    # Wake the job worker to process next queued job
    _job_signal.set()
    
    return {"status": "cancelled"}


# ============================================
# HISTORY ENDPOINTS
# ============================================

@app.get("/history", response_model=models.HistoryListResponse)
async def list_history(
    profile_id: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List generation history with optional filters."""
    query = models.HistoryQuery(
        profile_id=profile_id,
        search=search,
        limit=limit,
        offset=offset,
    )
    return await history.list_generations(query, db)


@app.get("/history/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get generation statistics."""
    return await history.get_generation_stats(db)


@app.post("/history/import")
async def import_generation(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import a generation from a ZIP archive."""
    # Validate file size (max 50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Read file content
    content = await file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)}MB"
        )
    
    try:
        result = await export_import.import_generation_from_zip(content, db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{generation_id}", response_model=models.HistoryResponse)
async def get_generation(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Get a generation by ID."""
    # Get generation with profile name
    result = db.query(
        DBGeneration,
        DBVoiceProfile.name.label('profile_name')
    ).join(
        DBVoiceProfile,
        DBGeneration.profile_id == DBVoiceProfile.id
    ).filter(
        DBGeneration.id == generation_id,
        DBGeneration.deleted_at.is_(None)
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    gen, profile_name = result
    return models.HistoryResponse(
        id=gen.id,
        profile_id=gen.profile_id,
        profile_name=profile_name,
        text=gen.text,
        language=gen.language,
        audio_path=gen.audio_path,
        duration=gen.duration,
        generation_time_seconds=gen.generation_time_seconds,
        seed=gen.seed,
        instruct=gen.instruct,
        model_size=gen.model_size,
        backend_type=gen.backend_type,
        request_user_id=gen.request_user_id,
        request_user_first_name=gen.request_user_first_name,
        request_ip=gen.request_ip,
        created_at=gen.created_at,
    )


@app.delete("/history/{generation_id}")
async def delete_generation(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Delete a generation."""
    success = await history.delete_generation(generation_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Generation not found")

    # Mark linked queue/job rows as deleted so /jobs lists do not surface them.
    jobs = db.query(DBGenerationJob).filter(DBGenerationJob.generation_id == generation_id).all()
    now = datetime.utcnow()
    for job in jobs:
        job.status = "deleted"
        if not job.completed_at:
            job.completed_at = now
        if not job.error:
            job.error = "Deleted by user"
    if jobs:
        db.commit()

    return {"message": "Generation deleted successfully"}


@app.get("/history/{generation_id}/export")
async def export_generation(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Export a generation as a ZIP archive."""
    try:
        # Get generation to create filename
        generation = db.query(DBGeneration).filter(
            DBGeneration.id == generation_id,
            DBGeneration.deleted_at.is_(None),
        ).first()
        if not generation:
            raise HTTPException(status_code=404, detail="Generation not found")
        
        # Export to ZIP
        zip_bytes = export_import.export_generation_to_zip(generation_id, db)
        
        # Create safe filename from text
        safe_text = "".join(c for c in generation.text[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_text:
            safe_text = "generation"
        filename = f"generation-{safe_text}.voicebox.zip"
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{generation_id}/export-audio")
async def export_generation_audio(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Export only the audio file from a generation."""
    generation = db.query(DBGeneration).filter(
        DBGeneration.id == generation_id,
        DBGeneration.deleted_at.is_(None),
    ).first()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    audio_path = Path(generation.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Create safe filename from text
    safe_text = "".join(c for c in generation.text[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_text:
        safe_text = "generation"
    filename = f"{safe_text}.wav"
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# ============================================
# TRANSCRIPTION ENDPOINTS
# ============================================

@app.post("/transcribe", response_model=models.TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Transcribe audio file to text."""
    logger.debug(f"[Transcribe] Received file: {file.filename}, content_type: {file.content_type}")
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    logger.debug(f"[Transcribe] Saved to temp file: {tmp_path}, size: {len(content)} bytes")
    
    try:
        # Get audio duration
        from .utils.audio import load_audio
        logger.debug(f"[Transcribe] Loading audio from {tmp_path}")
        audio, sr = load_audio(tmp_path)
        duration = len(audio) / sr
        logger.debug(f"[Transcribe] Audio loaded: {len(audio)} samples, {sr}Hz, {duration:.2f}s")
        
        # Transcribe
        whisper_model = transcribe.get_whisper_model()

        # Check if Whisper model is downloaded
        model_size = whisper_model.model_size
        logger.debug(f"[Transcribe] Using whisper model size: {model_size}")
        
        # Get the correct model path based on backend type
        backend_type = get_backend_type()
        if backend_type == "mlx":
            from .backends.mlx_backend import MLXSTTBackend
            mlx_whisper_map = MLXSTTBackend.get_mlx_whisper_model_map()
            model_repo_id = mlx_whisper_map.get(model_size, f"mlx-community/whisper-{model_size}-mlx")
        else:
            model_repo_id = f"openai/whisper-{model_size}"

        logger.debug(f"[Transcribe] Model repo ID: {model_repo_id}")

        # Check if model is cached
        from huggingface_hub import constants as hf_constants
        repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + model_repo_id.replace("/", "--"))
        logger.debug(f"[Transcribe] Checking cache at: {repo_cache}, exists: {repo_cache.exists()}")
        
        if not repo_cache.exists():
            # Start download in background
            progress_model_name = f"whisper-{model_size}"

            async def download_whisper_background():
                try:
                    await whisper_model.load_model_async(model_size)
                except Exception as e:
                    logger.exception(f"[Transcribe] Background download error: {e}")
                    get_task_manager().error_download(progress_model_name, str(e))

            get_task_manager().start_download(progress_model_name)
            asyncio.create_task(download_whisper_background())

            # Return 202 Accepted
            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"Whisper model {model_size} is being downloaded. Please wait and try again.",
                    "model_name": progress_model_name,
                    "downloading": True
                }
            )

        logger.debug("[Transcribe] Starting transcription...")
        text = await whisper_model.transcribe(tmp_path, language)
        save_model_prefs(stt_size=model_size)
        logger.debug(f"[Transcribe] Transcription complete: {text[:100] if text else '(empty)'}...")
        
        return models.TranscriptionResponse(
            text=text,
            duration=duration,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[Transcribe] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


# ============================================
# STORY ENDPOINTS
# ============================================

@app.get("/stories", response_model=List[models.StoryResponse])
async def list_stories(db: Session = Depends(get_db)):
    """List all stories."""
    return await stories.list_stories(db)


@app.post("/stories", response_model=models.StoryResponse)
async def create_story(
    data: models.StoryCreate,
    db: Session = Depends(get_db),
):
    """Create a new story."""
    try:
        return await stories.create_story(data, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stories/{story_id}", response_model=models.StoryDetailResponse)
async def get_story(
    story_id: str,
    db: Session = Depends(get_db),
):
    """Get a story with all its items."""
    story = await stories.get_story(story_id, db)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    return story


@app.put("/stories/{story_id}", response_model=models.StoryResponse)
async def update_story(
    story_id: str,
    data: models.StoryCreate,
    db: Session = Depends(get_db),
):
    """Update a story."""
    story = await stories.update_story(story_id, data, db)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    return story


@app.delete("/stories/{story_id}")
async def delete_story(
    story_id: str,
    db: Session = Depends(get_db),
):
    """Delete a story."""
    success = await stories.delete_story(story_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Story not found")
    return {"message": "Story deleted successfully"}


@app.post("/stories/{story_id}/items", response_model=models.StoryItemDetail)
async def add_story_item(
    story_id: str,
    data: models.StoryItemCreate,
    db: Session = Depends(get_db),
):
    """Add a generation to a story."""
    item = await stories.add_item_to_story(story_id, data, db)
    if not item:
        raise HTTPException(status_code=404, detail="Story or generation not found")
    return item


@app.delete("/stories/{story_id}/items/{item_id}")
async def remove_story_item(
    story_id: str,
    item_id: str,
    db: Session = Depends(get_db),
):
    """Remove a story item from a story."""
    success = await stories.remove_item_from_story(story_id, item_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Story item not found")
    return {"message": "Item removed successfully"}


@app.put("/stories/{story_id}/items/times")
async def update_story_item_times(
    story_id: str,
    data: models.StoryItemBatchUpdate,
    db: Session = Depends(get_db),
):
    """Update story item timecodes."""
    success = await stories.update_story_item_times(story_id, data, db)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid timecode update request")
    return {"message": "Item timecodes updated successfully"}


@app.put("/stories/{story_id}/items/reorder", response_model=List[models.StoryItemDetail])
async def reorder_story_items(
    story_id: str,
    data: models.StoryItemReorder,
    db: Session = Depends(get_db),
):
    """Reorder story items and recalculate timecodes."""
    items = await stories.reorder_story_items(story_id, data.generation_ids, db)
    if items is None:
        raise HTTPException(status_code=400, detail="Invalid reorder request - ensure all generation IDs belong to this story")
    return items


@app.put("/stories/{story_id}/items/{item_id}/move", response_model=models.StoryItemDetail)
async def move_story_item(
    story_id: str,
    item_id: str,
    data: models.StoryItemMove,
    db: Session = Depends(get_db),
):
    """Move a story item (update position and/or track)."""
    item = await stories.move_story_item(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found")
    return item


@app.put("/stories/{story_id}/items/{item_id}/trim", response_model=models.StoryItemDetail)
async def trim_story_item(
    story_id: str,
    item_id: str,
    data: models.StoryItemTrim,
    db: Session = Depends(get_db),
):
    """Trim a story item (update trim_start_ms and trim_end_ms)."""
    item = await stories.trim_story_item(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found or invalid trim values")
    return item


@app.post("/stories/{story_id}/items/{item_id}/split", response_model=List[models.StoryItemDetail])
async def split_story_item(
    story_id: str,
    item_id: str,
    data: models.StoryItemSplit,
    db: Session = Depends(get_db),
):
    """Split a story item at a given time, creating two clips."""
    items = await stories.split_story_item(story_id, item_id, data, db)
    if items is None:
        raise HTTPException(status_code=404, detail="Story item not found or invalid split point")
    return items


@app.post("/stories/{story_id}/items/{item_id}/duplicate", response_model=models.StoryItemDetail)
async def duplicate_story_item(
    story_id: str,
    item_id: str,
    db: Session = Depends(get_db),
):
    """Duplicate a story item, creating a copy with all properties."""
    item = await stories.duplicate_story_item(story_id, item_id, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found")
    return item


@app.get("/stories/{story_id}/export-audio")
async def export_story_audio(
    story_id: str,
    db: Session = Depends(get_db),
):
    """Export story as single mixed audio file with timecode-based mixing."""
    try:
        # Get story to create filename
        story = db.query(database.Story).filter_by(id=story_id).first()
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        
        # Export audio
        audio_bytes = await stories.export_story_audio(story_id, db)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Story has no audio items")
        
        # Create safe filename
        safe_name = "".join(c for c in story.name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = "story"
        filename = f"{safe_name}.wav"
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# FILE SERVING
# ============================================

@app.get("/audio/{generation_id}")
async def get_audio(generation_id: str, db: Session = Depends(get_db)):
    """Serve generated audio file."""
    generation = await history.get_generation(generation_id, db)
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    audio_path = Path(generation.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"generation_{generation_id}.wav",
    )


@app.get("/samples/{sample_id}")
async def get_sample_audio(sample_id: str, db: Session = Depends(get_db)):
    """Serve profile sample audio file."""
    from .database import ProfileSample as DBProfileSample
    
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    audio_path = Path(sample.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"sample_{sample_id}.wav",
    )


# ============================================
# MODEL MANAGEMENT
# ============================================

@app.post("/models/load")
async def load_model(model_size: str = "1.7B"):
    """Manually load TTS model."""
    try:
        tts_model = tts.get_tts_model()
        await tts_model.load_model_async(model_size)
        save_model_prefs(tts_size=model_size)
        return {"message": f"Model {model_size} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload")
async def unload_model():
    """Unload TTS model to free memory."""
    try:
        tts.unload_tts_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/progress/{model_name}")
async def get_model_progress(model_name: str):
    """Get model download progress via Server-Sent Events."""
    from fastapi.responses import StreamingResponse

    progress_manager = get_progress_manager()

    async def event_generator():
        """Generate SSE events for progress updates."""
        async for event in progress_manager.subscribe(model_name):
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


@app.get("/models/progress-snapshot/{model_name}")
async def get_model_progress_snapshot(model_name: str):
    """Get current model download progress as a single JSON snapshot (for polling)."""
    progress_manager = get_progress_manager()
    progress = progress_manager.get_progress(model_name)
    if progress is None:
        return {"model_name": model_name, "status": "idle", "current": 0, "total": 0, "progress": 0, "filename": None}
    return progress


@app.get("/models/status", response_model=models.ModelStatusListResponse)
async def get_model_status():
    """Get status of all available models."""
    from huggingface_hub import constants as hf_constants
    from pathlib import Path
    
    backend_type = get_backend_type()
    task_manager = get_task_manager()
    
    # Get set of currently downloading model names
    active_download_names = {task.model_name for task in task_manager.get_active_downloads()}
    
    # Try to import scan_cache_dir (might not be available in older versions)
    try:
        from huggingface_hub import scan_cache_dir
        use_scan_cache = True
    except ImportError:
        use_scan_cache = False
    
    def check_tts_loaded(model_size: str):
        """Check if TTS model is loaded with specific size."""
        try:
            tts_model = tts.get_tts_model()
            return tts_model.is_loaded() and getattr(tts_model, 'model_size', None) == model_size
        except Exception:
            return False
    
    def check_whisper_loaded(model_size: str):
        """Check if Whisper model is loaded with specific size."""
        try:
            whisper_model = transcribe.get_whisper_model()
            return whisper_model.is_loaded() and getattr(whisper_model, 'model_size', None) == model_size
        except Exception:
            return False
    
    # Use backend-specific model IDs
    if backend_type == "mlx":
        from .backends.mlx_backend import MLXTTSBackend, MLXSTTBackend
        _mlx_tts = MLXTTSBackend()
        tts_1_7b_id = _mlx_tts._get_model_path("1.7B")
        tts_0_6b_id = _mlx_tts._get_model_path("0.6B")
        # MLX backend uses mlx-community Whisper models
        mlx_whisper_map = MLXSTTBackend.get_mlx_whisper_model_map()
        whisper_base_id = mlx_whisper_map["base"]
        whisper_small_id = mlx_whisper_map["small"]
        whisper_medium_id = mlx_whisper_map["medium"]
        whisper_large_id = mlx_whisper_map["large"]
        whisper_large_v3_turbo_id = mlx_whisper_map["large-v3-turbo"]
    else:
        tts_1_7b_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        tts_0_6b_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        whisper_base_id = "openai/whisper-base"
        whisper_small_id = "openai/whisper-small"
        whisper_medium_id = "openai/whisper-medium"
        whisper_large_id = "openai/whisper-large"
        whisper_large_v3_turbo_id = "openai/whisper-large-v3-turbo"
    
    model_configs = [
        {
            "model_name": "qwen-tts-1.7B",
            "display_name": "Qwen TTS 1.7B",
            "hf_repo_id": tts_1_7b_id,
            "model_size": "1.7B",
            "check_loaded": lambda: check_tts_loaded("1.7B"),
        },
        {
            "model_name": "qwen-tts-0.6B",
            "display_name": "Qwen TTS 0.6B",
            "hf_repo_id": tts_0_6b_id,
            "model_size": "0.6B",
            "check_loaded": lambda: check_tts_loaded("0.6B"),
        },
        {
            "model_name": "whisper-base",
            "display_name": "Whisper Base",
            "hf_repo_id": whisper_base_id,
            "model_size": "base",
            "check_loaded": lambda: check_whisper_loaded("base"),
        },
        {
            "model_name": "whisper-small",
            "display_name": "Whisper Small",
            "hf_repo_id": whisper_small_id,
            "model_size": "small",
            "check_loaded": lambda: check_whisper_loaded("small"),
        },
        {
            "model_name": "whisper-medium",
            "display_name": "Whisper Medium",
            "hf_repo_id": whisper_medium_id,
            "model_size": "medium",
            "check_loaded": lambda: check_whisper_loaded("medium"),
        },
        {
            "model_name": "whisper-large",
            "display_name": "Whisper Large",
            "hf_repo_id": whisper_large_id,
            "model_size": "large",
            "check_loaded": lambda: check_whisper_loaded("large"),
        },
        {
            "model_name": "whisper-large-v3-turbo",
            "display_name": "Whisper Large V3 Turbo",
            "hf_repo_id": whisper_large_v3_turbo_id,
            "model_size": "large-v3-turbo",
            "check_loaded": lambda: check_whisper_loaded("large-v3-turbo"),
        },
    ]
    
    # Build a mapping of model_name -> hf_repo_id so we can check if shared repos are downloading
    model_to_repo = {cfg["model_name"]: cfg["hf_repo_id"] for cfg in model_configs}
    
    # Get the set of hf_repo_ids that are currently being downloaded
    # This handles the case where multiple models share the same repo (e.g., 0.6B and 1.7B on MLX)
    active_download_repos = {model_to_repo.get(name) for name in active_download_names if name in model_to_repo}
    
    # Get HuggingFace cache info (if available)
    cache_info = None
    if use_scan_cache:
        try:
            cache_info = scan_cache_dir()
        except Exception:
            # Function failed, continue without it
            pass
    
    statuses = []
    
    for model_config in model_configs:
        try:
            downloaded = False
            size_mb = None
            loaded = False
            
            # Method 1: Try using scan_cache_dir if available
            if cache_info:
                repo_id = model_config["hf_repo_id"]
                for repo in cache_info.repos:
                    if repo.repo_id == repo_id:
                        # Check if actual model weight files exist (not just config files)
                        # scan_cache_dir only shows completed files, so check if any are model weights
                        has_model_weights = False
                        for rev in repo.revisions:
                            for f in rev.files:
                                fname = f.file_name.lower()
                                if fname.endswith(('.safetensors', '.bin', '.pt', '.pth', '.npz')):
                                    has_model_weights = True
                                    break
                            if has_model_weights:
                                break
                        
                        # Also check for .incomplete files in blobs directory (downloads in progress)
                        has_incomplete = False
                        try:
                            cache_dir = hf_constants.HF_HUB_CACHE
                            blobs_dir = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "blobs"
                            if blobs_dir.exists():
                                has_incomplete = any(blobs_dir.glob("*.incomplete"))
                        except Exception:
                            pass
                        
                        # Only mark as downloaded if we have model weights AND no incomplete files
                        if has_model_weights and not has_incomplete:
                            downloaded = True
                            # Calculate size from cache info
                            try:
                                total_size = sum(revision.size_on_disk for revision in repo.revisions)
                                size_mb = total_size / (1024 * 1024)
                            except Exception:
                                pass
                        break
            
            # Method 2: Fallback to checking cache directory directly (using HuggingFace's OS-specific cache location)
            if not downloaded:
                try:
                    cache_dir = hf_constants.HF_HUB_CACHE
                    repo_cache = Path(cache_dir) / ("models--" + model_config["hf_repo_id"].replace("/", "--"))
                    
                    if repo_cache.exists():
                        # Check for .incomplete files - if any exist, download is still in progress
                        blobs_dir = repo_cache / "blobs"
                        has_incomplete = blobs_dir.exists() and any(blobs_dir.glob("*.incomplete"))
                        
                        if not has_incomplete:
                            # Check for actual model weight files (not just index files)
                            # in the snapshots directory (symlinks to completed blobs)
                            snapshots_dir = repo_cache / "snapshots"
                            has_model_files = False
                            if snapshots_dir.exists():
                                has_model_files = (
                                    any(snapshots_dir.rglob("*.bin")) or
                                    any(snapshots_dir.rglob("*.safetensors")) or
                                    any(snapshots_dir.rglob("*.pt")) or
                                    any(snapshots_dir.rglob("*.pth")) or
                                    any(snapshots_dir.rglob("*.npz"))
                                )
                            
                            if has_model_files:
                                downloaded = True
                                # Calculate size (exclude .incomplete files)
                                try:
                                    total_size = sum(
                                        f.stat().st_size for f in repo_cache.rglob("*") 
                                        if f.is_file() and not f.name.endswith('.incomplete')
                                    )
                                    size_mb = total_size / (1024 * 1024)
                                except Exception:
                                    pass
                except Exception:
                    pass
            
            # Method 3 removed - checking for config.json is too lenient
            # Methods 1 and 2 properly verify that model weight files exist
            
            # Check if loaded in memory
            try:
                loaded = model_config["check_loaded"]()
            except Exception:
                loaded = False

            # Check if this model (or its shared repo) is currently being downloaded
            is_downloading = model_config["hf_repo_id"] in active_download_repos

            # If downloading, don't report as downloaded (partial files exist)
            if is_downloading:
                downloaded = False
                size_mb = None  # Don't show partial size during download

            # If no size from disk, query HuggingFace API
            if size_mb is None and not is_downloading:
                from .utils.hf_sizes import get_repo_size_mb
                size_mb = await get_repo_size_mb(model_config["hf_repo_id"])

            statuses.append(models.ModelStatus(
                model_name=model_config["model_name"],
                display_name=model_config["display_name"],
                downloaded=downloaded,
                downloading=is_downloading,
                size_mb=size_mb,
                loaded=loaded,
            ))
        except Exception:
            # If check fails, try to at least check if loaded
            try:
                loaded = model_config["check_loaded"]()
            except Exception:
                loaded = False

            # Check if this model (or its shared repo) is currently being downloaded
            is_downloading = model_config["hf_repo_id"] in active_download_repos

            # If not downloading, try to get size from HuggingFace API
            size_mb = None
            if not is_downloading:
                from .utils.hf_sizes import get_repo_size_mb
                size_mb = await get_repo_size_mb(model_config["hf_repo_id"])

            statuses.append(models.ModelStatus(
                model_name=model_config["model_name"],
                display_name=model_config["display_name"],
                downloaded=False,  # Assume not downloaded if check failed
                downloading=is_downloading,
                size_mb=size_mb,
                loaded=loaded,
            ))
    
    return models.ModelStatusListResponse(models=statuses)


@app.post("/models/download")
async def trigger_model_download(request: models.ModelDownloadRequest):
    """Trigger download of a specific model."""
    import asyncio
    
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()
    
    model_configs = {
        "qwen-tts-1.7B": {
            "model_size": "1.7B",
            "load_func": lambda: tts.get_tts_model().load_model("1.7B"),
        },
        "qwen-tts-0.6B": {
            "model_size": "0.6B",
            "load_func": lambda: tts.get_tts_model().load_model("0.6B"),
        },
        "whisper-base": {
            "model_size": "base",
            "load_func": lambda: transcribe.get_whisper_model().load_model("base"),
        },
        "whisper-small": {
            "model_size": "small",
            "load_func": lambda: transcribe.get_whisper_model().load_model("small"),
        },
        "whisper-medium": {
            "model_size": "medium",
            "load_func": lambda: transcribe.get_whisper_model().load_model("medium"),
        },
        "whisper-large": {
            "model_size": "large",
            "load_func": lambda: transcribe.get_whisper_model().load_model("large"),
        },
        "whisper-large-v3-turbo": {
            "model_size": "large-v3-turbo",
            "load_func": lambda: transcribe.get_whisper_model().load_model("large-v3-turbo"),
        },
    }
    
    if request.model_name not in model_configs:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")
    
    config = model_configs[request.model_name]
    
    async def download_in_background():
        """Download model in background without blocking the HTTP request."""
        try:
            # Call the load function (which may be async)
            result = config["load_func"]()
            # If it's a coroutine, await it
            if asyncio.iscoroutine(result):
                await result
            # Mark progress as complete - this notifies SSE listeners
            # This is needed because _load_model_sync only marks complete if
            # the model wasn't already cached, but we always init progress here
            progress_manager.mark_complete(request.model_name)
            task_manager.complete_download(request.model_name)
        except Exception as e:
            progress_manager.mark_error(request.model_name, str(e))
            task_manager.error_download(request.model_name, str(e))

    # Start tracking download
    task_manager.start_download(request.model_name)

    # Initialize progress state so SSE endpoint has initial data to send.
    # This fixes a race condition where the frontend connects to SSE before
    # any progress callbacks have fired (especially for large models like Qwen
    # where huggingface_hub takes time to fetch metadata for all files).
    progress_manager.update_progress(
        model_name=request.model_name,
        current=0,
        total=0,  # Will be updated once actual total is known
        filename="Connecting to HuggingFace...",
        status="downloading",
    )

    # Start download in background task and store reference for cancellation
    bg_task = asyncio.create_task(download_in_background())
    task_manager.set_download_task(request.model_name, bg_task)

    # Return immediately - frontend should poll progress endpoint
    return {"message": f"Model {request.model_name} download started"}


@app.post("/models/cancel/{model_name}")
async def cancel_model_download(model_name: str):
    """Cancel an in-progress model download."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    if not task_manager.is_download_active(model_name):
        raise HTTPException(status_code=404, detail=f"No active download for {model_name}")

    cancelled = task_manager.cancel_download(model_name)
    if cancelled:
        progress_manager.mark_error(model_name, "Download cancelled")
    return {"message": f"Download of {model_name} cancelled"}


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model from the HuggingFace cache."""
    import shutil
    from huggingface_hub import constants as hf_constants
    
    # Map model names to HuggingFace repo IDs
    model_configs = {
        "qwen-tts-1.7B": {
            "hf_repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "model_size": "1.7B",
            "model_type": "tts",
        },
        "qwen-tts-0.6B": {
            "hf_repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "model_size": "0.6B",
            "model_type": "tts",
        },
        "whisper-base": {
            "hf_repo_id": "openai/whisper-base",
            "model_size": "base",
            "model_type": "whisper",
        },
        "whisper-small": {
            "hf_repo_id": "openai/whisper-small",
            "model_size": "small",
            "model_type": "whisper",
        },
        "whisper-medium": {
            "hf_repo_id": "openai/whisper-medium",
            "model_size": "medium",
            "model_type": "whisper",
        },
        "whisper-large": {
            "hf_repo_id": "openai/whisper-large",
            "model_size": "large",
            "model_type": "whisper",
        },
    }
    
    if model_name not in model_configs:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    config = model_configs[model_name]
    hf_repo_id = config["hf_repo_id"]
    
    try:
        # Check if model is loaded and unload it first
        if config["model_type"] == "tts":
            tts_model = tts.get_tts_model()
            if tts_model.is_loaded() and tts_model.model_size == config["model_size"]:
                tts.unload_tts_model()
        elif config["model_type"] == "whisper":
            whisper_model = transcribe.get_whisper_model()
            if whisper_model.is_loaded() and whisper_model.model_size == config["model_size"]:
                transcribe.unload_whisper_model()
        
        # Find and delete the cache directory (using HuggingFace's OS-specific cache location)
        cache_dir = hf_constants.HF_HUB_CACHE
        repo_cache_dir = Path(cache_dir) / ("models--" + hf_repo_id.replace("/", "--"))
        
        # Check if the cache directory exists
        if not repo_cache_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in cache")
        
        # Delete the entire cache directory for this model
        try:
            shutil.rmtree(repo_cache_dir)
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete model cache directory: {str(e)}"
            )
        
        return {"message": f"Model {model_name} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@app.post("/cache/clear")
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


# ============================================
# TASK MANAGEMENT
# ============================================

@app.get("/tasks/active", response_model=models.ActiveTasksResponse)
async def get_active_tasks():
    """Return all currently active downloads and generations."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()
    
    # Get active downloads from both task manager and progress manager
    # Task manager tracks which downloads are active
    # Progress manager has the actual progress data
    active_downloads = []
    task_manager_downloads = task_manager.get_active_downloads()
    progress_active = progress_manager.get_all_active()
    
    # Combine data from both sources
    download_map = {task.model_name: task for task in task_manager_downloads}
    progress_map = {p["model_name"]: p for p in progress_active}
    
    # Create unified list
    all_model_names = set(download_map.keys()) | set(progress_map.keys())
    for model_name in all_model_names:
        task = download_map.get(model_name)
        progress = progress_map.get(model_name)
        
        if task:
            active_downloads.append(models.ActiveDownloadTask(
                model_name=model_name,
                status=task.status,
                started_at=task.started_at,
            ))
        elif progress:
            # Progress exists but no task - create from progress data
            timestamp_str = progress.get("timestamp")
            if timestamp_str:
                try:
                    started_at = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    started_at = datetime.utcnow()
            else:
                started_at = datetime.utcnow()
            
            active_downloads.append(models.ActiveDownloadTask(
                model_name=model_name,
                status=progress.get("status", "downloading"),
                started_at=started_at,
            ))
    
    # Get active generations
    active_generations = []
    for gen_task in task_manager.get_active_generations():
        active_generations.append(models.ActiveGenerationTask(
            task_id=gen_task.task_id,
            profile_id=gen_task.profile_id,
            text_preview=gen_task.text_preview,
            started_at=gen_task.started_at,
        ))
    
    return models.ActiveTasksResponse(
        downloads=active_downloads,
        generations=active_generations,
    )


# ============================================
# STARTUP & SHUTDOWN
# ============================================

def _get_gpu_status() -> str:
    """Get GPU availability status."""
    backend_type = get_backend_type()
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    elif backend_type == "mlx":
        return "Metal (Apple Silicon via MLX)"
    return "None (CPU only)"


def _get_model_prefs_path() -> Path:
    """Get path to model preferences JSON file."""
    return config.get_data_dir() / "model_prefs.json"


def _load_model_prefs() -> dict:
    """Load model preferences from disk."""
    import json
    prefs_path = _get_model_prefs_path()
    if prefs_path.exists():
        try:
            return json.loads(prefs_path.read_text())
        except Exception:
            pass
    return {}


def save_model_prefs(tts_size: str = None, stt_size: str = None):
    """Save model preferences to disk (called after successful model load)."""
    import json
    prefs = _load_model_prefs()
    if tts_size:
        prefs["tts_model_size"] = tts_size
    if stt_size:
        prefs["stt_model_size"] = stt_size
    try:
        _get_model_prefs_path().write_text(json.dumps(prefs, indent=2))
    except Exception as e:
        logger.warning(f"Could not save model preferences: {e}")


def _cleanup_stale_jobs():
    """Mark any leftover queued/generating jobs as timeout on server start."""
    db = database.SessionLocal()
    try:
        stale = db.query(DBGenerationJob).filter(
            DBGenerationJob.status.in_(["queued", "generating", "cancelling"])
        ).all()
        for job in stale:
            job.status = "timeout"
            job.completed_at = datetime.utcnow()
            logger.info(f"[TTS] Marked stale job {job.id} as timeout (server restart)")
        if stale:
            db.commit()
            logger.info(f"[TTS] Cleaned up {len(stale)} stale jobs from previous run")
    finally:
        db.close()


async def _job_worker():
    """Background worker that processes queued generation jobs one at a time."""
    logger.info("[TTS] Job worker started")
    while True:
        try:
            # Wait for signal or poll every 2s
            try:
                await asyncio.wait_for(_job_signal.wait(), timeout=2.0)
                _job_signal.clear()
            except asyncio.TimeoutError:
                pass

            db = database.SessionLocal()
            try:
                # Check for stuck jobs (generating > 5 min)
                from datetime import timedelta
                _expire_old_queued_jobs(db)

                cutoff = datetime.utcnow() - timedelta(minutes=_GENERATING_JOB_TIMEOUT_MINUTES)
                stuck = db.query(DBGenerationJob).filter(
                    DBGenerationJob.status.in_(["generating", "cancelling"]),
                    DBGenerationJob.started_at < cutoff,
                ).all()
                for job in stuck:
                    job.status = "timeout"
                    job.error = "Generation timeout"
                    job.completed_at = datetime.utcnow()
                    try:
                        get_progress_manager().mark_error(job.id, "Generation timeout")
                    except Exception:
                        pass
                    logger.warning(f"[TTS] Job {job.id} timed out (stuck >5 min)")
                if stuck:
                    db.commit()

                # Skip if something is already generating
                active = db.query(DBGenerationJob).filter(
                    DBGenerationJob.status.in_(["generating", "cancelling"])
                ).first()
                if active:
                    continue

                # Pick oldest queued job
                job = db.query(DBGenerationJob).filter(
                    DBGenerationJob.status == "queued"
                ).order_by(DBGenerationJob.created_at).first()
                if not job:
                    continue

                # Mark as generating
                job.status = "generating"
                job.started_at = datetime.utcnow()
                db.commit()

                job_id = job.id
                profile_id = job.profile_id
                text = job.text
                language = job.language
                seed = job.seed
                model_size = job.model_size or "1.7B"
                instruct = job.instruct
                request_user_id = job.request_user_id
                request_user_first_name = job.request_user_first_name
                request_ip = job.request_ip
                backend_type = get_backend_type()
                job.backend_type = backend_type
                db.commit()
            finally:
                db.close()

            logger.info(f"[TTS] Job {job_id} starting generation (ip={request_ip or 'unknown'})")

            progress_manager = get_progress_manager()
            task_manager = get_task_manager()

            task_manager.start_generation(
                task_id=job_id,
                profile_id=profile_id,
                text=text,
            )

            # Update SSE to "generating" status
            progress_manager.update_progress(
                model_name=job_id,
                current=0,
                total=100,
                status="generating",
            )

            gen_db = database.SessionLocal()
            try:
                generation_started_at = datetime.utcnow()
                async with _model_lock:
                    def on_progress(pct):
                        if job_id in _cancel_requested_jobs:
                            raise RuntimeError("Cancelled by user")
                        progress_manager.update_progress(
                            model_name=job_id,
                            current=int(pct),
                            total=100,
                            status="generating",
                        )
                        task_manager.update_generation_progress(job_id, pct)
                        # Throttled DB update (~every 5%)
                        if int(pct) % 5 == 0:
                            try:
                                upd_db = database.SessionLocal()
                                upd_job = upd_db.query(DBGenerationJob).get(job_id)
                                if upd_job:
                                    upd_job.progress = pct
                                    upd_db.commit()
                                upd_db.close()
                            except Exception:
                                pass

                    profile = await profiles.get_profile(profile_id, gen_db)
                    if not profile:
                        raise ValueError("Profile not found")

                    voice_prompt = await profiles.create_voice_prompt_for_profile(
                        profile_id, gen_db,
                    )

                    tts_model = tts.get_tts_model()

                    # Don't silently download — require the model to be cached first
                    if not tts_model._is_model_cached(model_size):
                        model_name = f"qwen-tts-{model_size}"
                        raise ValueError(f"Model {model_name} is not downloaded. Please download it first from the Models page.")

                    await tts_model.load_model_async(model_size)
                    save_model_prefs(tts_size=model_size)

                    audio, sample_rate = await tts_model.generate(
                        text, voice_prompt, language, seed, instruct,
                        progress_callback=on_progress,
                    )

                duration = len(audio) / sample_rate
                generation_time_seconds = (datetime.utcnow() - generation_started_at).total_seconds()
                audio_path = config.get_generations_dir() / f"{job_id}.wav"
                from .utils.audio import save_audio
                save_audio(audio, str(audio_path), sample_rate)

                generation = await history.create_generation(
                    profile_id=profile_id,
                    text=text,
                    language=language,
                    audio_path=str(audio_path),
                    duration=duration,
                    seed=seed,
                    db=gen_db,
                    instruct=instruct,
                    model_size=model_size,
                    backend_type=backend_type,
                    request_user_id=request_user_id,
                    request_user_first_name=request_user_first_name,
                    request_ip=request_ip,
                    generation_time_seconds=generation_time_seconds,
                )

                # Mark job complete
                job_row = gen_db.query(DBGenerationJob).get(job_id)
                if job_row:
                    job_row.status = "complete"
                    job_row.progress = 100.0
                    job_row.generation_id = generation.id if hasattr(generation, 'id') else None
                    job_row.completed_at = datetime.utcnow()
                    gen_db.commit()

                progress_manager.mark_complete(job_id)
                task_manager.complete_generation(job_id)
                _cancel_requested_jobs.discard(job_id)
                logger.info(f"[TTS] Job {job_id} complete ({duration:.1f}s audio)")

            except Exception as e:
                logger.exception(f"[TTS] Job {job_id} failed: {e}")
                is_cancelled = job_id in _cancel_requested_jobs or "cancel" in str(e).lower()
                progress_manager.mark_error(job_id, "Cancelled by user" if is_cancelled else str(e))
                task_manager.complete_generation(job_id)

                err_db = database.SessionLocal()
                try:
                    err_job = err_db.query(DBGenerationJob).get(job_id)
                    if err_job:
                        err_job.status = "cancelled" if is_cancelled else "error"
                        err_job.error = ("Cancelled by user" if is_cancelled else str(e))[:1000]
                        err_job.completed_at = datetime.utcnow()
                        err_db.commit()
                finally:
                    err_db.close()
                _cancel_requested_jobs.discard(job_id)
            finally:
                gen_db.close()

            # Immediately check for more queued jobs
            _job_signal.set()

        except asyncio.CancelledError:
            logger.info("[TTS] Job worker shutting down")
            break
        except Exception as e:
            logger.exception(f"[TTS] Job worker error: {e}")
            await asyncio.sleep(2)


async def _startup():
    """Run on application startup."""
    _log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=getattr(logging, _log_level, logging.INFO),
    )
    logger.info("voicebox API starting up...")
    database.init_db()
    logger.info(f"Database initialized at {database._db_path}")
    backend_type = get_backend_type()
    logger.info(f"Backend: {backend_type.upper()}")
    logger.info(f"GPU available: {_get_gpu_status()}")

    # Initialize progress manager with main event loop for thread-safe operations
    try:
        progress_manager = get_progress_manager()
        progress_manager._set_main_loop(asyncio.get_running_loop())
        logger.info("Progress manager initialized with event loop")
    except Exception as e:
        logger.warning(f"Could not initialize progress manager event loop: {e}")

    # HuggingFace setup
    try:
        from huggingface_hub import constants as hf_constants

        # Check for HF_TOKEN (huggingface_hub reads it automatically)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            logger.info(f"HuggingFace token: {'*' * 4}{hf_token[-4:]}")
        else:
            logger.info("HuggingFace token: not set (set HF_TOKEN env var for gated models)")

        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"HuggingFace cache directory: {cache_dir}")
    except Exception as e:
        logger.warning(f"Could not set up HuggingFace: {e}")
        logger.warning("Model downloads may fail. Please ensure the directory exists and has write permissions.")

    # Set event loop on idle timers so they can schedule unloads
    loop = asyncio.get_running_loop()
    try:
        tts.get_tts_model()._idle_timer.set_loop(loop)
    except Exception:
        pass
    try:
        transcribe.get_whisper_model()._idle_timer.set_loop(loop)
    except Exception:
        pass

    # Clear stale jobs from previous server run
    _cleanup_stale_jobs()

    # Preload models in the background based on last-used preferences
    asyncio.create_task(_preload_models())

    # Start the job worker
    asyncio.create_task(_job_worker())


async def _preload_models():
    """Preload TTS and STT models based on saved preferences."""
    prefs = _load_model_prefs()
    tts_size = prefs.get("tts_model_size", "1.7B")

    # Preload TTS model
    try:
        tts_backend = tts.get_tts_model()
        if tts_backend._is_model_cached(tts_size):
            logger.info(f"Preloading TTS model ({tts_size})...")
            await tts_backend.load_model_async(tts_size)
            logger.info(f"TTS model ({tts_size}) preloaded")
        else:
            logger.info(f"TTS model ({tts_size}) not cached, skipping preload")
    except Exception as e:
        logger.warning(f"TTS preload failed: {e}", exc_info=True)

    # STT model is NOT preloaded — it loads on first /transcribe call.
    # This saves memory when the user doesn't use Create Voice.


async def _shutdown():
    """Run on application shutdown."""
    logger.info("voicebox API shutting down...")
    # Unload models to free memory
    tts.unload_tts_model()
    transcribe.unload_whisper_model()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="voicebox backend server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (use 0.0.0.0 for remote access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for database, profiles, and generated audio",
    )
    args = parser.parse_args()

    # Set data directory if provided
    if args.data_dir:
        config.set_data_dir(args.data_dir)

    # Initialize database after data directory is set
    database.init_db()

    _log_level = os.environ.get("LOG_LEVEL", "info").lower()
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,  # Disable reload in production
        log_level=_log_level,
    )
