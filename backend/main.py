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
import tempfile
import io
from pathlib import Path
import uuid
import asyncio
import signal
import os

# Set HSA_OVERRIDE_GFX_VERSION for AMD GPUs that aren't officially listed in ROCm
# (e.g., RX 6600 is gfx1032 which maps to gfx1030 target)
# This must be set BEFORE any torch.cuda calls
if not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Suppress noisy MIOpen workspace warnings on AMD GPUs
if not os.environ.get("MIOPEN_LOG_LEVEL"):
    os.environ["MIOPEN_LOG_LEVEL"] = "4"

import torch
from urllib.parse import quote


def _safe_content_disposition(disposition_type: str, filename: str) -> str:
    """Build a Content-Disposition header that is safe for non-ASCII filenames.

    Uses RFC 5987 ``filename*`` parameter so that browsers can decode
    UTF-8 filenames while the ``filename`` fallback stays ASCII-only.
    """
    ascii_name = "".join(
        c for c in filename if c.isascii() and (c.isalnum() or c in " -_.")
    ).strip() or "download"
    utf8_name = quote(filename, safe="")
    return (
        f'{disposition_type}; filename="{ascii_name}"; '
        f"filename*=UTF-8''{utf8_name}"
    )


from . import database, models, profiles, history, tts, transcribe, config, export_import, channels, stories, __version__
from .database import get_db, Generation as DBGeneration, VoiceProfile as DBVoiceProfile
from .utils.progress import get_progress_manager
from .utils.tasks import get_task_manager
from .utils.cache import clear_voice_prompt_cache
from .platform_detect import get_backend_type

# Keep references to fire-and-forget background tasks to prevent GC
_background_tasks: set = set()

# Generation queue — serializes TTS inference to avoid GPU contention
_generation_queue: asyncio.Queue = None  # type: ignore  # initialized at startup


def _create_background_task(coro) -> asyncio.Task:
    """Create a background task and prevent it from being garbage collected."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _generation_worker():
    """Worker that processes generation tasks one at a time."""
    while True:
        coro = await _generation_queue.get()
        try:
            await coro
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            _generation_queue.task_done()


def _enqueue_generation(coro):
    """Add a generation coroutine to the serial queue."""
    _generation_queue.put_nowait(coro)


app = FastAPI(
    title="voicebox API",
    description="Production-quality Qwen3-TTS voice cloning API",
    version=__version__,
)

# CORS middleware - restrict to known local origins by default.
# Set VOICEBOX_CORS_ORIGINS env var to a comma-separated list of origins
# to allow additional origins (e.g. for remote server mode).
_default_origins = [
    "http://localhost:5173",     # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:17493",
    "http://127.0.0.1:17493",
    "tauri://localhost",         # Tauri webview (macOS)
    "https://tauri.localhost",   # Tauri webview (Windows/Linux)
]
_env_origins = os.environ.get("VOICEBOX_CORS_ORIGINS", "")
_cors_origins = _default_origins + [o.strip() for o in _env_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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

    # Check for GPU availability (CUDA, MPS, Intel Arc XPU, or DirectML)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # Intel Arc / Intel Xe via intel-extension-for-pytorch (IPEX)
    has_xpu = False
    xpu_name = None
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            has_xpu = True
            try:
                xpu_name = torch.xpu.get_device_name(0)
            except Exception:
                xpu_name = "Intel GPU"
    except ImportError:
        pass

    # DirectML backend (torch-directml) for any Windows GPU
    has_directml = False
    directml_name = None
    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            has_directml = True
            try:
                directml_name = torch_directml.device_name(0)
            except Exception:
                directml_name = "DirectML GPU"
    except ImportError:
        pass

    gpu_available = has_cuda or has_mps or has_xpu or has_directml or backend_type == "mlx"

    gpu_type = None
    if has_cuda:
        gpu_type = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif has_mps:
        gpu_type = "MPS (Apple Silicon)"
    elif backend_type == "mlx":
        gpu_type = "Metal (Apple Silicon via MLX)"
    elif has_xpu:
        gpu_type = f"XPU ({xpu_name})"
    elif has_directml:
        gpu_type = f"DirectML ({directml_name})"

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
        backend_variant=os.environ.get("VOICEBOX_BACKEND_VARIANT", "cuda" if torch.cuda.is_available() else "cpu"),
    )


@app.get("/health/filesystem", response_model=models.FilesystemHealthResponse)
async def filesystem_health():
    """Check filesystem health: directory existence, write permissions, and disk space."""
    import shutil

    dirs_to_check = {
        "generations": config.get_generations_dir(),
        "profiles": config.get_profiles_dir(),
        "data": config.get_data_dir(),
    }

    checks: list[models.DirectoryCheck] = []
    all_ok = True

    for _label, dir_path in dirs_to_check.items():
        exists = dir_path.exists()
        writable = False
        error = None
        if exists:
            # Probe writability with a temp file
            probe = dir_path / ".voicebox_probe"
            try:
                probe.write_text("ok")
                probe.unlink()
                writable = True
            except PermissionError:
                error = "Permission denied"
            except OSError as e:
                error = str(e)
            finally:
                try:
                    probe.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            error = "Directory does not exist"

        if not exists or not writable:
            all_ok = False

        checks.append(
            models.DirectoryCheck(
                path=str(dir_path),
                exists=exists,
                writable=writable,
                error=error,
            )
        )

    # Disk space for the data directory
    disk_free_mb = None
    disk_total_mb = None
    try:
        usage = shutil.disk_usage(str(config.get_data_dir()))
        disk_free_mb = round(usage.free / (1024 * 1024), 1)
        disk_total_mb = round(usage.total / (1024 * 1024), 1)
        if disk_free_mb < 500:
            all_ok = False
    except OSError:
        all_ok = False

    return models.FilesystemHealthResponse(
        healthy=all_ok,
        disk_free_mb=disk_free_mb,
        disk_total_mb=disk_total_mb,
        directories=checks,
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Fallback for unexpected errors
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
    try:
        profile = await profiles.update_profile(profile_id, data, db)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    # Preserve the uploaded file's extension so librosa can detect format correctly.
    # Defaulting to .wav was causing soundfile to reject MP3/WebM content as invalid WAV.
    _allowed_audio_exts = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.webm', '.opus'}
    _uploaded_ext = Path(file.filename or '').suffix.lower()
    file_suffix = _uploaded_ext if _uploaded_ext in _allowed_audio_exts else '.wav'

    with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {str(e)}")
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
                "Content-Disposition": _safe_content_disposition("attachment", filename)
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
    """Generate speech from text using a voice profile.
    
    Creates a history entry immediately with status='generating' and kicks off
    TTS in the background. The frontend can poll or use SSE to detect completion.
    """
    task_manager = get_task_manager()
    generation_id = str(uuid.uuid4())

    # Validate profile exists before creating the record
    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    from .backends import get_tts_backend_for_engine
    engine = data.engine or "qwen"
    tts_model = get_tts_backend_for_engine(engine)
    model_size = data.model_size or "1.7B"

    # Create the history entry immediately with status="generating"
    generation = await history.create_generation(
        profile_id=data.profile_id,
        text=data.text,
        language=data.language,
        audio_path="",
        duration=0,
        seed=data.seed,
        db=db,
        instruct=data.instruct,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=model_size if engine == "qwen" else None,
    )

    # Track in task manager
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=data.profile_id,
        text=data.text,
    )

    # Resolve effects chain: explicit request > profile default > none
    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    else:
        # Check profile default
        import json as _json
        profile_obj = db.query(DBVoiceProfile).filter_by(id=data.profile_id).first()
        if profile_obj and profile_obj.effects_chain:
            try:
                effects_chain_config = _json.loads(profile_obj.effects_chain)
            except Exception:
                pass

    # Kick off TTS in background
    async def _run_generation():
        bg_db = next(get_db())
        try:
            # Load model
            if engine == "qwen":
                await tts_model.load_model_async(model_size)
            else:
                await tts_model.load_model()

            # Create voice prompt
            voice_prompt = await profiles.create_voice_prompt_for_profile(
                data.profile_id,
                bg_db,
                use_cache=True,
                engine=engine,
            )

            from .utils.chunked_tts import generate_chunked

            trim_fn = None
            if engine in ("chatterbox", "chatterbox_turbo"):
                from .utils.audio import trim_tts_output
                trim_fn = trim_tts_output

            audio, sample_rate = await generate_chunked(
                tts_model,
                data.text,
                voice_prompt,
                language=data.language,
                seed=data.seed,
                instruct=data.instruct,
                max_chunk_chars=data.max_chunk_chars,
                crossfade_ms=data.crossfade_ms,
                trim_fn=trim_fn,
            )

            if data.normalize:
                from .utils.audio import normalize_audio
                audio = normalize_audio(audio)

            duration = len(audio) / sample_rate

            # Always save clean version first
            clean_audio_path = config.get_generations_dir() / f"{generation_id}.wav"
            from .utils.audio import save_audio
            save_audio(audio, str(clean_audio_path), sample_rate)

            from . import versions as versions_mod

            has_effects = effects_chain_config and any(
                e.get("enabled", True) for e in effects_chain_config
            )

            # Create clean version entry
            versions_mod.create_version(
                generation_id=generation_id,
                label="original",
                audio_path=str(clean_audio_path),
                db=bg_db,
                effects_chain=None,
                is_default=not has_effects,
            )

            # Apply effects and create processed version if configured
            final_audio_path = str(clean_audio_path)
            if has_effects:
                from .utils.effects import apply_effects, validate_effects_chain
                error_msg = validate_effects_chain(effects_chain_config)
                if error_msg:
                    print(f"Warning: invalid effects chain, skipping: {error_msg}")
                else:
                    processed_audio = apply_effects(audio, sample_rate, effects_chain_config)
                    processed_path = config.get_generations_dir() / f"{generation_id}_processed.wav"
                    save_audio(processed_audio, str(processed_path), sample_rate)
                    final_audio_path = str(processed_path)
                    versions_mod.create_version(
                        generation_id=generation_id,
                        label="version-2",
                        audio_path=str(processed_path),
                        db=bg_db,
                        effects_chain=effects_chain_config,
                        is_default=True,
                    )

            # Update the record to completed
            await history.update_generation_status(
                generation_id=generation_id,
                status="completed",
                db=bg_db,
                audio_path=final_audio_path,
                duration=duration,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=bg_db,
                error=str(e),
            )
        finally:
            task_manager.complete_generation(generation_id)
            bg_db.close()

    _enqueue_generation(_run_generation())

    return generation


@app.post("/generate/{generation_id}/retry", response_model=models.GenerationResponse)
async def retry_generation(generation_id: str, db: Session = Depends(get_db)):
    """Retry a failed generation using the same parameters."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") != "failed":
        raise HTTPException(status_code=400, detail="Only failed generations can be retried")

    # Reset the record to generating
    gen.status = "generating"
    gen.error = None
    gen.audio_path = ""
    gen.duration = 0
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    # Resolve engine/model from stored values
    retry_engine = gen.engine or "qwen"
    retry_model_size = gen.model_size or "1.7B"

    from .backends import get_tts_backend_for_engine
    tts_model = get_tts_backend_for_engine(retry_engine)

    async def _run_retry():
        bg_db = next(get_db())
        try:
            if retry_engine == "qwen":
                await tts_model.load_model_async(retry_model_size)
            else:
                await tts_model.load_model()

            voice_prompt = await profiles.create_voice_prompt_for_profile(
                gen.profile_id,
                bg_db,
                use_cache=True,
                engine=retry_engine,
            )

            from .utils.chunked_tts import generate_chunked

            trim_fn = None
            if retry_engine in ("chatterbox", "chatterbox_turbo"):
                from .utils.audio import trim_tts_output
                trim_fn = trim_tts_output

            audio, sample_rate = await generate_chunked(
                tts_model,
                gen.text,
                voice_prompt,
                language=gen.language,
                seed=gen.seed,
                instruct=gen.instruct,
                trim_fn=trim_fn,
            )

            duration = len(audio) / sample_rate
            audio_path = config.get_generations_dir() / f"{generation_id}.wav"

            from .utils.audio import save_audio
            save_audio(audio, str(audio_path), sample_rate)

            await history.update_generation_status(
                generation_id=generation_id,
                status="completed",
                db=bg_db,
                audio_path=str(audio_path),
                duration=duration,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=bg_db,
                error=str(e),
            )
        finally:
            task_manager.complete_generation(generation_id)
            bg_db.close()

    _enqueue_generation(_run_retry())

    return models.GenerationResponse.model_validate(gen)


@app.post(
    "/generate/{generation_id}/regenerate",
    response_model=models.GenerationResponse,
)
async def regenerate_generation(generation_id: str, db: Session = Depends(get_db)):
    """Re-run TTS with the same parameters and save the result as a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation must be completed to regenerate")

    from .backends import get_tts_backend_for_engine
    from . import versions as versions_mod

    regen_engine = gen.engine or "qwen"
    regen_model_size = gen.model_size or "1.7B"
    tts_model = get_tts_backend_for_engine(regen_engine)

    # Set to generating so the UI shows the loader and SSE picks it up
    gen.status = "generating"
    gen.error = None
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    version_id = str(uuid.uuid4())

    async def _run_regenerate():
        bg_db = next(get_db())
        try:
            if regen_engine == "qwen":
                await tts_model.load_model_async(regen_model_size)
            else:
                await tts_model.load_model()

            voice_prompt = await profiles.create_voice_prompt_for_profile(
                gen.profile_id,
                bg_db,
                use_cache=True,
                engine=regen_engine,
            )

            from .utils.chunked_tts import generate_chunked

            trim_fn = None
            if regen_engine in ("chatterbox", "chatterbox_turbo"):
                from .utils.audio import trim_tts_output
                trim_fn = trim_tts_output

            audio, sample_rate = await generate_chunked(
                tts_model,
                gen.text,
                voice_prompt,
                language=gen.language,
                seed=None,  # New seed for variation
                instruct=gen.instruct,
                trim_fn=trim_fn,
            )

            from .utils.audio import normalize_audio, save_audio
            audio = normalize_audio(audio)

            duration = len(audio) / sample_rate
            audio_path = config.get_generations_dir() / f"{generation_id}_{version_id[:8]}.wav"

            save_audio(audio, str(audio_path), sample_rate)

            # Count existing versions to auto-label
            existing = versions_mod.list_versions(generation_id, bg_db)
            label = f"take-{len(existing) + 1}"

            versions_mod.create_version(
                generation_id=generation_id,
                label=label,
                audio_path=str(audio_path),
                db=bg_db,
                effects_chain=None,
                is_default=True,
            )

            await history.update_generation_status(
                generation_id=generation_id,
                status="completed",
                db=bg_db,
                audio_path=str(audio_path),
                duration=duration,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=bg_db,
                error=str(e),
            )
        finally:
            task_manager.complete_generation(generation_id)
            bg_db.close()

    _enqueue_generation(_run_regenerate())

    return models.GenerationResponse.model_validate(gen)


@app.get("/generate/{generation_id}/status")
async def get_generation_status(generation_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that streams generation status updates.
    
    Polls the DB every second and yields the current status. Closes when
    the generation reaches 'completed' or 'failed'.
    """
    import json

    async def event_stream():
        while True:
            db.expire_all()
            gen = db.query(DBGeneration).filter_by(id=generation_id).first()
            if not gen:
                yield f"data: {json.dumps({'status': 'not_found', 'id': generation_id})}\n\n"
                return

            payload = {
                "id": gen.id,
                "status": gen.status or "completed",
                "duration": gen.duration,
                "error": gen.error,
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if (gen.status or "completed") in ("completed", "failed"):
                return

            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate/stream")
async def stream_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """
    Generate speech and stream the WAV audio directly without saving to disk.

    Returns raw WAV bytes via a StreamingResponse so the client can start
    playing audio before the entire file has been received.  This endpoint
    does NOT create a history entry — use /generate for that.
    """
    from .backends import get_tts_backend_for_engine

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    engine = data.engine or "qwen"
    tts_model = get_tts_backend_for_engine(engine)
    model_size = data.model_size or "1.7B"

    if engine == "qwen":
        if not tts_model._is_model_cached(model_size):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_size} is not downloaded yet. Use /generate to trigger a download.",
            )
        await tts_model.load_model_async(model_size)
    elif engine == "luxtts":
        if not tts_model._is_model_cached():
            raise HTTPException(
                status_code=400,
                detail="LuxTTS model is not downloaded yet. Use /generate to trigger a download.",
            )
        await tts_model.load_model()
    elif engine == "chatterbox":
        if not tts_model._is_model_cached():
            raise HTTPException(
                status_code=400,
                detail="Chatterbox model is not downloaded yet. Use /generate to trigger a download.",
            )
        await tts_model.load_model()
    elif engine == "chatterbox_turbo":
        if not tts_model._is_model_cached():
            raise HTTPException(
                status_code=400,
                detail="Chatterbox Turbo model is not downloaded yet. Use /generate to trigger a download.",
            )
        await tts_model.load_model()

    voice_prompt = await profiles.create_voice_prompt_for_profile(
        data.profile_id, db, engine=engine,
    )

    from .utils.chunked_tts import generate_chunked

    trim_fn = None
    if engine in ("chatterbox", "chatterbox_turbo"):
        from .utils.audio import trim_tts_output
        trim_fn = trim_tts_output

    audio, sample_rate = await generate_chunked(
        tts_model,
        data.text,
        voice_prompt,
        language=data.language,
        seed=data.seed,
        instruct=data.instruct,
        max_chunk_chars=data.max_chunk_chars,
        crossfade_ms=data.crossfade_ms,
        trim_fn=trim_fn,
    )

    if data.normalize:
        from .utils.audio import normalize_audio
        audio = normalize_audio(audio)

    wav_bytes = tts.audio_to_wav_bytes(audio, sample_rate)

    async def _wav_stream():
        # Yield in chunks so large responses don't block the event loop
        chunk_size = 64 * 1024  # 64 KB
        for i in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[i : i + chunk_size]

    return StreamingResponse(
        _wav_stream(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


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


@app.post("/history/{generation_id}/favorite")
async def toggle_favorite(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Toggle the favorite status of a generation."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    gen.is_favorited = not gen.is_favorited
    db.commit()
    return {"is_favorited": gen.is_favorited}


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
                "Content-Disposition": _safe_content_disposition("attachment", filename)
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
            "Content-Disposition": _safe_content_disposition("attachment", filename)
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
        audio, sr = await asyncio.to_thread(load_audio, tmp_path)
        duration = len(audio) / sr
        
        # Transcribe
        whisper_model = transcribe.get_whisper_model()

        # Check if Whisper model is downloaded
        model_size = whisper_model.model_size
        # Map model sizes to HF repo IDs (some need special suffixes)
        whisper_hf_repos = {
            "large": "openai/whisper-large-v3",
            "turbo": "openai/whisper-large-v3-turbo",
        }
        model_name = whisper_hf_repos.get(model_size, f"openai/whisper-{model_size}")

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
            _create_background_task(download_whisper_background())

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


@app.put("/stories/{story_id}/items/{item_id}/version", response_model=models.StoryItemDetail)
async def set_story_item_version(
    story_id: str,
    item_id: str,
    data: models.StoryItemVersionUpdate,
    db: Session = Depends(get_db),
):
    """Pin a story item to a specific generation version."""
    item = await stories.set_story_item_version(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item or version not found")
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
                "Content-Disposition": _safe_content_disposition("attachment", filename)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# EFFECTS & VERSIONS
# ============================================

@app.post("/effects/preview/{generation_id}")
async def preview_effects(
    generation_id: str,
    data: models.ApplyEffectsRequest,
    db: Session = Depends(get_db),
):
    """Apply effects to a generation's clean audio and stream back the result without saving.

    Used for ephemeral preview/auditioning of effects chains.
    """
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation is not completed")

    from . import versions as versions_mod
    from .utils.effects import apply_effects, validate_effects_chain
    from .utils.audio import load_audio

    # Validate chain
    chain_dicts = [e.model_dump() for e in data.effects_chain]
    error = validate_effects_chain(chain_dicts)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Find the original unprocessed version (no effects applied)
    all_versions = versions_mod.list_versions(generation_id, db)
    clean_version = next((v for v in all_versions if v.effects_chain is None), None)
    source_path = clean_version.audio_path if clean_version else gen.audio_path
    if not source_path or not Path(source_path).exists():
        raise HTTPException(status_code=404, detail="Source audio file not found")

    # Process in memory (off the event loop)
    audio, sample_rate = await asyncio.to_thread(load_audio, source_path)
    processed = await asyncio.to_thread(apply_effects, audio, sample_rate, chain_dicts)

    # Write to in-memory buffer
    import soundfile as sf
    buf = io.BytesIO()
    await asyncio.to_thread(lambda: sf.write(buf, processed, sample_rate, format="WAV"))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="preview_{generation_id}.wav"',
            "Cache-Control": "no-cache, no-store",
        },
    )


@app.get("/effects/available", response_model=models.AvailableEffectsResponse)
async def get_available_effects():
    """List all available effect types with parameter definitions."""
    from .utils.effects import get_available_effects as _get_effects
    return models.AvailableEffectsResponse(effects=[
        models.AvailableEffect(**e) for e in _get_effects()
    ])


@app.get("/effects/presets", response_model=List[models.EffectPresetResponse])
async def list_effect_presets(db: Session = Depends(get_db)):
    """List all effect presets (built-in + user-created)."""
    from . import effects as effects_mod
    return effects_mod.list_presets(db)


@app.get("/effects/presets/{preset_id}", response_model=models.EffectPresetResponse)
async def get_effect_preset(preset_id: str, db: Session = Depends(get_db)):
    """Get a specific effect preset."""
    from . import effects as effects_mod
    preset = effects_mod.get_preset(preset_id, db)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset


@app.post("/effects/presets", response_model=models.EffectPresetResponse)
async def create_effect_preset(
    data: models.EffectPresetCreate,
    db: Session = Depends(get_db),
):
    """Create a new effect preset."""
    from . import effects as effects_mod
    try:
        return effects_mod.create_preset(data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/effects/presets/{preset_id}", response_model=models.EffectPresetResponse)
async def update_effect_preset(
    preset_id: str,
    data: models.EffectPresetUpdate,
    db: Session = Depends(get_db),
):
    """Update an effect preset."""
    from . import effects as effects_mod
    try:
        result = effects_mod.update_preset(preset_id, data, db)
        if not result:
            raise HTTPException(status_code=404, detail="Preset not found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/effects/presets/{preset_id}")
async def delete_effect_preset(preset_id: str, db: Session = Depends(get_db)):
    """Delete a user effect preset."""
    from . import effects as effects_mod
    try:
        if not effects_mod.delete_preset(preset_id, db):
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/generations/{generation_id}/versions",
    response_model=List[models.GenerationVersionResponse],
)
async def list_generation_versions(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """List all versions for a generation."""
    gen = await history.get_generation(generation_id, db)
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    from . import versions as versions_mod
    return versions_mod.list_versions(generation_id, db)


@app.post(
    "/generations/{generation_id}/versions/apply-effects",
    response_model=models.GenerationVersionResponse,
)
async def apply_effects_to_generation(
    generation_id: str,
    data: models.ApplyEffectsRequest,
    db: Session = Depends(get_db),
):
    """Apply an effects chain to an existing generation, creating a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation is not completed")

    from . import versions as versions_mod
    from .utils.effects import apply_effects, validate_effects_chain
    from .utils.audio import load_audio, save_audio

    # Validate effects chain
    chain_dicts = [e.model_dump() for e in data.effects_chain]
    error = validate_effects_chain(chain_dicts)
    if error:
        raise HTTPException(status_code=400, detail=error)

    # Determine source audio: use specified version, or fall back to clean/original
    all_versions = versions_mod.list_versions(generation_id, db)
    source_version_id = data.source_version_id
    if source_version_id:
        source_version = next(
            (v for v in all_versions if v.id == source_version_id), None
        )
        if not source_version:
            raise HTTPException(status_code=404, detail="Source version not found")
        source_path = source_version.audio_path
    else:
        clean_version = next(
            (v for v in all_versions if v.effects_chain is None), None
        )
        if not clean_version:
            source_path = gen.audio_path
        else:
            source_path = clean_version.audio_path
            source_version_id = clean_version.id

    if not source_path or not Path(source_path).exists():
        raise HTTPException(status_code=404, detail="Source audio file not found")

    # Load, process, save (off the event loop)
    audio, sample_rate = await asyncio.to_thread(load_audio, source_path)
    processed_audio = await asyncio.to_thread(apply_effects, audio, sample_rate, chain_dicts)

    # Generate a unique filename
    version_id = str(uuid.uuid4())
    processed_path = config.get_generations_dir() / f"{generation_id}_{version_id[:8]}.wav"
    await asyncio.to_thread(save_audio, processed_audio, str(processed_path), sample_rate)

    # Auto-label
    label = data.label or f"version-{len(all_versions) + 1}"

    version = versions_mod.create_version(
        generation_id=generation_id,
        label=label,
        audio_path=str(processed_path),
        db=db,
        effects_chain=chain_dicts,
        is_default=data.set_as_default,
        source_version_id=source_version_id,
    )

    return version


@app.put(
    "/generations/{generation_id}/versions/{version_id}/set-default",
    response_model=models.GenerationVersionResponse,
)
async def set_default_version(
    generation_id: str,
    version_id: str,
    db: Session = Depends(get_db),
):
    """Set a specific version as the default for a generation."""
    from . import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version or version.generation_id != generation_id:
        raise HTTPException(status_code=404, detail="Version not found")

    result = versions_mod.set_default_version(version_id, db)
    if not result:
        raise HTTPException(status_code=404, detail="Version not found")
    return result


@app.delete("/generations/{generation_id}/versions/{version_id}")
async def delete_generation_version(
    generation_id: str,
    version_id: str,
    db: Session = Depends(get_db),
):
    """Delete a version. Cannot delete the last remaining version."""
    from . import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version or version.generation_id != generation_id:
        raise HTTPException(status_code=404, detail="Version not found")

    if not versions_mod.delete_version(version_id, db):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the last remaining version",
        )
    return {"status": "deleted"}


@app.get("/audio/version/{version_id}")
async def get_version_audio(version_id: str, db: Session = Depends(get_db)):
    """Serve audio for a specific version."""
    from . import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    audio_path = Path(version.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"generation_{version.generation_id}_{version.label}.wav",
    )


@app.put("/profiles/{profile_id}/effects", response_model=models.VoiceProfileResponse)
async def update_profile_effects(
    profile_id: str,
    data: models.ProfileEffectsUpdate,
    db: Session = Depends(get_db),
):
    """Set or clear the default effects chain for a voice profile."""
    import json as _json

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if data.effects_chain is not None:
        from .utils.effects import validate_effects_chain
        chain_dicts = [e.model_dump() for e in data.effects_chain]
        error = validate_effects_chain(chain_dicts)
        if error:
            raise HTTPException(status_code=400, detail=error)
        profile.effects_chain = _json.dumps(chain_dicts)
    else:
        profile.effects_chain = None

    profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


def _profile_to_response(profile) -> models.VoiceProfileResponse:
    """Convert a DB profile to a VoiceProfileResponse with parsed effects_chain."""
    import json as _json
    import logging

    effects_chain = None
    if profile.effects_chain:
        try:
            raw = _json.loads(profile.effects_chain)
            effects_chain = [models.EffectConfig(**e) for e in raw]
        except Exception as e:
            logging.warning(f"Failed to parse effects_chain for profile {profile.id}: {e}")

    return models.VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        language=profile.language,
        avatar_path=profile.avatar_path,
        effects_chain=effects_chain,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


# ============================================
# FILE SERVING
# ============================================

@app.get("/audio/{generation_id}")
async def get_audio(generation_id: str, db: Session = Depends(get_db)):
    """Serve generated audio file (serves the default version)."""
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
    """Unload the default Qwen TTS model to free memory."""
    try:
        tts.unload_tts_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/unload")
async def unload_model_by_name(model_name: str):
    """Unload a specific model from memory without deleting it from disk."""
    # Map of model_name -> (model_type, model_size)
    model_types = {
        "qwen-tts-1.7B": ("tts", "1.7B"),
        "qwen-tts-0.6B": ("tts", "0.6B"),
        "luxtts": ("luxtts", "default"),
        "chatterbox-tts": ("chatterbox", "default"),
        "chatterbox-turbo": ("chatterbox_turbo", "default"),
        "whisper-base": ("whisper", "base"),
        "whisper-small": ("whisper", "small"),
        "whisper-medium": ("whisper", "medium"),
        "whisper-large": ("whisper", "large"),
        "whisper-turbo": ("whisper", "turbo"),
    }

    if model_name not in model_types:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    model_type, model_size = model_types[model_name]

    try:
        if model_type == "tts":
            tts_model = tts.get_tts_model()
            loaded_size = getattr(
                tts_model, "_current_model_size", None
            ) or getattr(tts_model, "model_size", None)
            if tts_model.is_loaded() and loaded_size == model_size:
                tts.unload_tts_model()
            else:
                return {"message": f"Model {model_name} is not loaded"}
        elif model_type == "luxtts":
            from .backends import get_tts_backend_for_engine
            backend = get_tts_backend_for_engine("luxtts")
            if backend.is_loaded():
                backend.unload_model()
            else:
                return {"message": f"Model {model_name} is not loaded"}
        elif model_type == "chatterbox":
            from .backends import get_tts_backend_for_engine
            backend = get_tts_backend_for_engine("chatterbox")
            if backend.is_loaded():
                backend.unload_model()
            else:
                return {"message": f"Model {model_name} is not loaded"}
        elif model_type == "chatterbox_turbo":
            from .backends import get_tts_backend_for_engine
            backend = get_tts_backend_for_engine("chatterbox_turbo")
            if backend.is_loaded():
                backend.unload_model()
            else:
                return {"message": f"Model {model_name} is not loaded"}
        elif model_type == "whisper":
            whisper_model = transcribe.get_whisper_model()
            if whisper_model.is_loaded() and whisper_model.model_size == model_size:
                transcribe.unload_whisper_model()
            else:
                return {"message": f"Model {model_name} is not loaded"}

        return {"message": f"Model {model_name} unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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


@app.get("/models/cache-dir")
async def get_models_cache_dir():
    """Get the path to the HuggingFace model cache directory."""
    from huggingface_hub import constants as hf_constants
    return {"path": str(Path(hf_constants.HF_HUB_CACHE))}


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _copy_with_progress(src: Path, dst: Path, progress_manager, copied_so_far: int, total_bytes: int) -> int:
    """Copy a directory tree with byte-level progress tracking."""
    import shutil
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_dir():
            copied_so_far = _copy_with_progress(item, dest_item, progress_manager, copied_so_far, total_bytes)
        else:
            size = item.stat().st_size
            shutil.copy2(str(item), str(dest_item))
            copied_so_far += size
            progress_manager.update_progress(
                "migration", copied_so_far, total_bytes,
                filename=item.name, status="downloading",
            )
    return copied_so_far


@app.post("/models/migrate")
async def migrate_models(request: models.ModelMigrateRequest):
    """Move all downloaded models to a new directory with byte-level progress via SSE."""
    import shutil
    from huggingface_hub import constants as hf_constants

    source = Path(hf_constants.HF_HUB_CACHE)
    destination = Path(request.destination)

    if not source.exists():
        raise HTTPException(status_code=404, detail="Current model cache directory not found")

    model_dirs = [d for d in source.iterdir() if d.name.startswith("models--") and d.is_dir()]
    if not model_dirs:
        return {"moved": 0, "errors": [], "source": str(source), "destination": str(destination)}

    destination.mkdir(parents=True, exist_ok=True)

    progress_manager = get_progress_manager()

    # Check if source and destination are on the same filesystem (rename is instant)
    same_fs = False
    try:
        same_fs = source.stat().st_dev == destination.stat().st_dev
    except OSError:
        pass

    async def migrate_background():
        moved = 0
        errors = []
        try:
            if same_fs:
                # Same filesystem: rename is instant, just track model count
                total = len(model_dirs)
                for i, item in enumerate(model_dirs):
                    dest_item = destination / item.name
                    try:
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        shutil.move(str(item), str(dest_item))
                        moved += 1
                        progress_manager.update_progress(
                            "migration", i + 1, total,
                            filename=item.name, status="downloading",
                        )
                    except Exception as e:
                        errors.append(f"{item.name}: {str(e)}")
            else:
                # Cross-filesystem: copy with byte-level progress, then delete source
                total_bytes = sum(_get_dir_size(d) for d in model_dirs)
                progress_manager.update_progress("migration", 0, total_bytes, filename="Calculating...", status="downloading")

                copied = 0
                for item in model_dirs:
                    dest_item = destination / item.name
                    try:
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        copied = await asyncio.to_thread(
                            _copy_with_progress, item, dest_item, progress_manager, copied, total_bytes
                        )
                        # Remove source after successful copy
                        await asyncio.to_thread(shutil.rmtree, str(item))
                        moved += 1
                    except Exception as e:
                        errors.append(f"{item.name}: {str(e)}")

            progress_manager.update_progress("migration", 1, 1, status="complete")
            progress_manager.mark_complete("migration")
        except Exception as e:
            progress_manager.update_progress("migration", 0, 0, status="error")
            progress_manager.mark_error("migration", str(e))

    _create_background_task(migrate_background())

    return {"source": str(source), "destination": str(destination)}


@app.get("/models/migrate/progress")
async def get_migration_progress():
    """Get model migration progress via Server-Sent Events."""
    from fastapi.responses import StreamingResponse

    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe("migration"):
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
            loaded_size = getattr(
                tts_model, "_current_model_size", None
            ) or getattr(tts_model, "model_size", None)
            return tts_model.is_loaded() and loaded_size == model_size
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
        whisper_large_id = "openai/whisper-large-v3"
    else:
        tts_1_7b_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        tts_0_6b_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        whisper_base_id = "openai/whisper-base"
        whisper_small_id = "openai/whisper-small"
        whisper_medium_id = "openai/whisper-medium"
        whisper_large_id = "openai/whisper-large-v3"
    
    # Check if LuxTTS backend is loaded
    def check_luxtts_loaded():
        try:
            from .backends import get_tts_backend_for_engine
            backend = get_tts_backend_for_engine("luxtts")
            return backend.is_loaded()
        except Exception:
            return False

    # Check if Chatterbox backend is loaded
    def check_chatterbox_loaded():
        try:
            from .backends import get_tts_backend_for_engine
            backend = get_tts_backend_for_engine("chatterbox")
            return backend.is_loaded()
        except Exception:
            return False

    # Check if Chatterbox Turbo backend is loaded
    def check_chatterbox_turbo_loaded():
        try:
            from .backends import get_tts_backend_for_engine
            backend = get_tts_backend_for_engine("chatterbox_turbo")
            return backend.is_loaded()
        except Exception:
            return False

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
            "model_name": "luxtts",
            "display_name": "LuxTTS (Fast, CPU-friendly)",
            "hf_repo_id": "YatharthS/LuxTTS",
            "model_size": "default",
            "check_loaded": check_luxtts_loaded,
        },
        {
            "model_name": "chatterbox-tts",
            "display_name": "Chatterbox TTS (Multilingual)",
            "hf_repo_id": "ResembleAI/chatterbox",
            "model_size": "default",
            "check_loaded": check_chatterbox_loaded,
        },
        {
            "model_name": "chatterbox-turbo",
            "display_name": "Chatterbox Turbo (English, Tags)",
            "hf_repo_id": "ResembleAI/chatterbox-turbo",
            "model_size": "default",
            "check_loaded": check_chatterbox_turbo_loaded,
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
            "model_name": "whisper-turbo",
            "display_name": "Whisper Turbo",
            "hf_repo_id": "openai/whisper-large-v3-turbo",
            "model_size": "turbo",
            "check_loaded": lambda: check_whisper_loaded("turbo"),
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
                hf_repo_id=config["hf_repo_id"],
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
                hf_repo_id=config["hf_repo_id"],
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
    from .backends import get_tts_backend_for_engine
    
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
        "luxtts": {
            "model_size": "default",
            "load_func": lambda: get_tts_backend_for_engine("luxtts").load_model(),
        },
        "chatterbox-tts": {
            "model_size": "default",
            "load_func": lambda: get_tts_backend_for_engine("chatterbox").load_model(),
        },
        "chatterbox-turbo": {
            "model_size": "default",
            "load_func": lambda: get_tts_backend_for_engine("chatterbox_turbo").load_model(),
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
        "whisper-turbo": {
            "model_size": "turbo",
            "load_func": lambda: transcribe.get_whisper_model().load_model("turbo"),
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
    _create_background_task(download_in_background())

    # Return immediately - frontend should poll progress endpoint
    return {"message": f"Model {request.model_name} download started"}


@app.post("/models/download/cancel")
async def cancel_model_download(request: models.ModelDownloadRequest):
    """Cancel or dismiss an errored/stale download task."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    removed = task_manager.cancel_download(request.model_name)

    # Also clear progress state so the model doesn't show as downloading
    progress_removed = False
    with progress_manager._lock:
        if request.model_name in progress_manager._progress:
            del progress_manager._progress[request.model_name]
            progress_removed = True

    if removed or progress_removed:
        return {"message": f"Download task for {request.model_name} cancelled"}
    return {"message": f"No active task found for {request.model_name}"}


@app.post("/tasks/clear")
async def clear_all_tasks():
    """Clear all download tasks and progress state. Does not delete downloaded files."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    task_manager.clear_all()

    with progress_manager._lock:
        progress_manager._progress.clear()
        progress_manager._last_notify_time.clear()
        progress_manager._last_notify_progress.clear()

    return {"message": "All task state cleared"}


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
        "luxtts": {
            "hf_repo_id": "YatharthS/LuxTTS",
            "model_size": "default",
            "model_type": "luxtts",
        },
        "chatterbox-tts": {
            "hf_repo_id": "ResembleAI/chatterbox",
            "model_size": "default",
            "model_type": "chatterbox",
        },
        "chatterbox-turbo": {
            "hf_repo_id": "ResembleAI/chatterbox-turbo",
            "model_size": "default",
            "model_type": "chatterbox_turbo",
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
            "hf_repo_id": "openai/whisper-large-v3",
            "model_size": "large",
            "model_type": "whisper",
        },
        "whisper-turbo": {
            "hf_repo_id": "openai/whisper-large-v3-turbo",
            "model_size": "turbo",
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
            loaded_size = getattr(
                tts_model, "_current_model_size", None
            ) or getattr(tts_model, "model_size", None)
            if tts_model.is_loaded() and loaded_size == config["model_size"]:
                tts.unload_tts_model()
        elif config["model_type"] == "luxtts":
            from .backends import get_tts_backend_for_engine
            luxtts = get_tts_backend_for_engine("luxtts")
            if luxtts.is_loaded():
                luxtts.unload_model()
        elif config["model_type"] == "chatterbox":
            from .backends import get_tts_backend_for_engine
            chatterbox = get_tts_backend_for_engine("chatterbox")
            if chatterbox.is_loaded():
                chatterbox.unload_model()
        elif config["model_type"] == "chatterbox_turbo":
            from .backends import get_tts_backend_for_engine
            turbo = get_tts_backend_for_engine("chatterbox_turbo")
            if turbo.is_loaded():
                turbo.unload_model()
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
            # Prefer task error, fall back to progress manager error
            error = task.error
            if not error:
                with progress_manager._lock:
                    pm_data = progress_manager._progress.get(model_name)
                    if pm_data:
                        error = pm_data.get("error")
            # Include progress data if available
            prog = progress or {}
            if not prog:
                with progress_manager._lock:
                    pm_data = progress_manager._progress.get(model_name)
                    if pm_data:
                        prog = pm_data
            active_downloads.append(models.ActiveDownloadTask(
                model_name=model_name,
                status=task.status,
                started_at=task.started_at,
                error=error,
                progress=prog.get("progress"),
                current=prog.get("current"),
                total=prog.get("total"),
                filename=prog.get("filename"),
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
                error=progress.get("error"),
                progress=progress.get("progress"),
                current=progress.get("current"),
                total=progress.get("total"),
                filename=progress.get("filename"),
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
# CUDA BACKEND MANAGEMENT
# ============================================

@app.get("/backend/cuda-status")
async def get_cuda_status():
    """Get CUDA backend download/availability status."""
    from . import cuda_download
    return cuda_download.get_cuda_status()


@app.post("/backend/download-cuda")
async def download_cuda_backend():
    """Download the CUDA backend binary. Returns immediately; track progress via SSE."""
    from . import cuda_download

    # Check if already downloaded
    if cuda_download.get_cuda_binary_path() is not None:
        raise HTTPException(status_code=409, detail="CUDA backend already downloaded")

    async def _download():
        try:
            await cuda_download.download_cuda_binary()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"CUDA download failed: {e}")

    _create_background_task(_download())
    return {"message": "CUDA backend download started", "progress_key": "cuda-backend"}


@app.delete("/backend/cuda")
async def delete_cuda_backend():
    """Delete the downloaded CUDA backend binary."""
    from . import cuda_download

    if cuda_download.is_cuda_active():
        raise HTTPException(
            status_code=409,
            detail="Cannot delete CUDA backend while it is active. Switch to CPU first.",
        )

    deleted = await cuda_download.delete_cuda_binary()
    if not deleted:
        raise HTTPException(status_code=404, detail="No CUDA backend found to delete")

    return {"message": "CUDA backend deleted"}


@app.get("/backend/cuda-progress")
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


# ============================================
# STARTUP & SHUTDOWN
# ============================================

def _get_gpu_status() -> str:
    """Get GPU availability status."""
    backend_type = get_backend_type()
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        # Check if this is ROCm (AMD) or CUDA (NVIDIA)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_rocm:
            return f"ROCm ({device_name})"
        return f"CUDA ({device_name})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    elif backend_type == "mlx":
        return "Metal (Apple Silicon via MLX)"
    return "None (CPU only)"


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    global _generation_queue
    print("voicebox API starting up...")
    database.init_db()
    print(f"Database initialized at {database._db_path}")

    # Start the serial generation worker
    _generation_queue = asyncio.Queue()
    _create_background_task(_generation_worker())

    # Mark any stale "generating" records as failed — these are leftovers
    # from a previous process that was killed mid-generation
    try:
        from sqlalchemy import text as sa_text
        db = next(get_db())
        result = db.execute(
            sa_text("UPDATE generations SET status = 'failed', error = 'Server was shut down during generation' WHERE status = 'generating'")
        )
        if result.rowcount > 0:
            print(f"Marked {result.rowcount} stale generation(s) as failed")
        db.commit()
        db.close()
    except Exception as e:
        print(f"Warning: Could not clean up stale generations: {e}")
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
