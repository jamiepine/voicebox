"""
FastAPI application for voicebox backend.

Handles voice cloning, generation history, and server mode.
"""

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
import asyncio
import signal
import os

from . import database, models, profiles, history, tts, transcribe, config, export_import, channels, stories, __version__
from .database import get_db, Generation as DBGeneration, VoiceProfile as DBVoiceProfile
from .utils.progress import get_progress_manager
from .utils.tasks import get_task_manager
from .utils.cache import clear_voice_prompt_cache
from .platform_detect import get_backend_type

app = FastAPI(
    title="voicebox API",
    description="Production-quality Qwen3-TTS voice cloning API",
    version=__version__,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health", response_model=models.HealthResponse)
async def health():
    """Health check endpoint."""
    from huggingface_hub import hf_hub_download, constants as hf_constants
    from pathlib import Path
    import os

    tts_model = tts.get_tts_model()
    backend_type = get_backend_type()

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
            default_model_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
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

@app.post("/generate", response_model=models.GenerationResponse)
async def generate_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech from text using a voice profile."""
    task_manager = get_task_manager()
    generation_id = str(uuid.uuid4())
    
    try:
        # Start tracking generation
        task_manager.start_generation(
            task_id=generation_id,
            profile_id=data.profile_id,
            text=data.text,
        )
        
        # Get profile
        profile = await profiles.get_profile(data.profile_id, db)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Create voice prompt from profile
        voice_prompt = await profiles.create_voice_prompt_for_profile(
            data.profile_id,
            db,
        )
        
        # Generate audio
        tts_model = tts.get_tts_model()
        # Load the requested model size if different from current (async to not block)
        model_size = data.model_size or "1.7B"

        # Check if model needs to be downloaded first
        model_path = tts_model._get_model_path(model_size)
        if model_path.startswith("Qwen/"):
            # Model not cached - check if it exists remotely or needs download
            from huggingface_hub import constants as hf_constants
            repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + model_path.replace("/", "--"))
            if not repo_cache.exists():
                # Start download in background
                model_name = f"qwen-tts-{model_size}"

                async def download_model_background():
                    try:
                        await tts_model.load_model_async(model_size)
                    except Exception as e:
                        task_manager.error_download(model_name, str(e))

                task_manager.start_download(model_name)
                asyncio.create_task(download_model_background())

                # Return 202 Accepted with download info
                raise HTTPException(
                    status_code=202,
                    detail={
                        "message": f"Model {model_size} is being downloaded. Please wait and try again.",
                        "model_name": model_name,
                        "downloading": True
                    }
                )

        await tts_model.load_model_async(model_size)
        audio, sample_rate = await tts_model.generate(
            data.text,
            voice_prompt,
            data.language,
            data.seed,
            data.instruct,
        )

        # Calculate duration
        duration = len(audio) / sample_rate

        # Save audio
        audio_path = config.get_generations_dir() / f"{generation_id}.wav"

        from .utils.audio import save_audio
        save_audio(audio, str(audio_path), sample_rate)

        # Create history entry
        generation = await history.create_generation(
            profile_id=data.profile_id,
            text=data.text,
            language=data.language,
            audio_path=str(audio_path),
            duration=duration,
            seed=data.seed,
            db=db,
            instruct=data.instruct,
        )
        
        # Mark generation as complete
        task_manager.complete_generation(generation_id)
        
        return generation
        
    except ValueError as e:
        task_manager.complete_generation(generation_id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        task_manager.complete_generation(generation_id)
        raise HTTPException(status_code=500, detail=str(e))


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
        DBGeneration.id == generation_id
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
        seed=gen.seed,
        instruct=gen.instruct,
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
    return {"message": "Generation deleted successfully"}


@app.get("/history/{generation_id}/export")
async def export_generation(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Export a generation as a ZIP archive."""
    try:
        # Get generation to create filename
        generation = db.query(DBGeneration).filter_by(id=generation_id).first()
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
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
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
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Get audio duration
        from .utils.audio import load_audio
        audio, sr = load_audio(tmp_path)
        duration = len(audio) / sr
        
        # Transcribe
        whisper_model = transcribe.get_whisper_model()

        # Check if Whisper model is downloaded (uses default size "base")
        model_size = whisper_model.model_size
        model_name = f"openai/whisper-{model_size}"

        # Check if model is cached
        from huggingface_hub import constants as hf_constants
        repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + model_name.replace("/", "--"))
        if not repo_cache.exists():
            # Start download in background
            progress_model_name = f"whisper-{model_size}"

            async def download_whisper_background():
                try:
                    await whisper_model.load_model_async(model_size)
                except Exception as e:
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

        text = await whisper_model.transcribe(tmp_path, language)
        
        return models.TranscriptionResponse(
            text=text,
            duration=duration,
        )
        
    except Exception as e:
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
        tts_1_7b_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        tts_0_6b_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"  # Fallback to 1.7B
        # MLX backend uses openai/whisper-* models, not mlx-community
        whisper_base_id = "openai/whisper-base"
        whisper_small_id = "openai/whisper-small"
        whisper_medium_id = "openai/whisper-medium"
        whisper_large_id = "openai/whisper-large"
    else:
        tts_1_7b_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        tts_0_6b_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        whisper_base_id = "openai/whisper-base"
        whisper_small_id = "openai/whisper-small"
        whisper_medium_id = "openai/whisper-medium"
        whisper_large_id = "openai/whisper-large"
    
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
    
    for config in model_configs:
        try:
            downloaded = False
            size_mb = None
            loaded = False
            
            # Method 1: Try using scan_cache_dir if available
            if cache_info:
                repo_id = config["hf_repo_id"]
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
                    repo_cache = Path(cache_dir) / ("models--" + config["hf_repo_id"].replace("/", "--"))
                    
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
                loaded = config["check_loaded"]()
            except Exception:
                loaded = False
            
            # Check if this model (or its shared repo) is currently being downloaded
            is_downloading = config["hf_repo_id"] in active_download_repos
            
            # If downloading, don't report as downloaded (partial files exist)
            if is_downloading:
                downloaded = False
                size_mb = None  # Don't show partial size during download
            
            statuses.append(models.ModelStatus(
                model_name=config["model_name"],
                display_name=config["display_name"],
                downloaded=downloaded,
                downloading=is_downloading,
                size_mb=size_mb,
                loaded=loaded,
            ))
        except Exception as e:
            # If check fails, try to at least check if loaded
            try:
                loaded = config["check_loaded"]()
            except Exception:
                loaded = False
            
            # Check if this model (or its shared repo) is currently being downloaded
            is_downloading = config["hf_repo_id"] in active_download_repos
            
            statuses.append(models.ModelStatus(
                model_name=config["model_name"],
                display_name=config["display_name"],
                downloaded=False,  # Assume not downloaded if check failed
                downloading=is_downloading,
                size_mb=None,
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
            task_manager.complete_download(request.model_name)
        except Exception as e:
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

    # Start download in background task (don't await)
    asyncio.create_task(download_in_background())

    # Return immediately - frontend should poll progress endpoint
    return {"message": f"Model {request.model_name} download started"}


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model from the HuggingFace cache."""
    import shutil
    import os
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
            "message": f"Voice prompt cache cleared successfully",
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


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    print("voicebox API starting up...")
    database.init_db()
    print(f"Database initialized at {database._db_path}")
    backend_type = get_backend_type()
    print(f"Backend: {backend_type.upper()}")
    print(f"GPU available: {_get_gpu_status()}")

    # Initialize progress manager with main event loop for thread-safe operations
    try:
        progress_manager = get_progress_manager()
        progress_manager._set_main_loop(asyncio.get_running_loop())
        print("Progress manager initialized with event loop")
    except Exception as e:
        print(f"Warning: Could not initialize progress manager event loop: {e}")

    # Ensure HuggingFace cache directory exists
    try:
        from huggingface_hub import constants as hf_constants
        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"HuggingFace cache directory: {cache_dir}")
    except Exception as e:
        print(f"Warning: Could not create HuggingFace cache directory: {e}")
        print("Model downloads may fail. Please ensure the directory exists and has write permissions.")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    print("voicebox API shutting down...")
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

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,  # Disable reload in production
    )
