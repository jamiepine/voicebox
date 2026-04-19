"""Health and infrastructure endpoints."""

import asyncio
import os
import signal
from pathlib import Path

import torch
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..services import tts
from ..database import get_db
from ..utils.platform_detect import get_backend_type

router = APIRouter()

# Frontend build directory — present in Docker, absent in dev/API-only mode
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"


@router.get("/")
async def root():
    """Root endpoint — serves SPA index.html in Docker, JSON otherwise."""
    from .. import __version__

    index = _frontend_dir / "index.html"
    if index.is_file():
        return FileResponse(index, media_type="text/html")
    return {"message": "voicebox API", "version": __version__}


@router.post("/shutdown")
async def shutdown():
    """Gracefully shutdown the server."""

    async def shutdown_async():
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(shutdown_async())
    return {"message": "Shutting down..."}


@router.post("/watchdog/disable")
async def watchdog_disable():
    """Disable the parent process watchdog so the server keeps running."""
    from backend.server import disable_watchdog

    disable_watchdog()
    return {"message": "Watchdog disabled"}


@router.get("/health", response_model=models.HealthResponse)
async def health():
    """Health check endpoint."""
    from huggingface_hub import constants as hf_constants
    from pathlib import Path

    tts_model = tts.get_tts_model()
    backend_type = get_backend_type()

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    has_xpu = False
    xpu_name = None
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401 -- side-effect import enables XPU

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            has_xpu = True
            try:
                xpu_name = torch.xpu.get_device_name(0)
            except Exception:
                xpu_name = "Intel GPU"
    except ImportError:
        pass

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

    gpu_compat_warning = None
    if has_cuda:
        from ..backends.base import check_cuda_compatibility

        _compatible, gpu_compat_warning = check_cuda_compatibility()

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
        vram_used = torch.cuda.memory_allocated() / 1024 / 1024
    elif has_xpu:
        try:
            vram_used = torch.xpu.memory_allocated() / 1024 / 1024
        except Exception:
            pass  # memory_allocated() may not be available on all IPEX versions

    model_loaded = False
    model_size = None
    try:
        if tts_model.is_loaded():
            model_loaded = True
            model_size = getattr(tts_model, "_current_model_size", None)
            if not model_size:
                model_size = getattr(tts_model, "model_size", None)
    except Exception:
        model_loaded = False
        model_size = None

    model_downloaded = None
    try:
        from ..backends import get_model_config

        default_config = get_model_config("qwen-tts-1.7B")
        default_model_id = default_config.hf_repo_id if default_config else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == default_model_id:
                    model_downloaded = True
                    break
        except (ImportError, Exception):
            cache_dir = hf_constants.HF_HUB_CACHE
            repo_cache = Path(cache_dir) / ("models--" + default_model_id.replace("/", "--"))
            if repo_cache.exists():
                has_model_files = (
                    any(repo_cache.rglob("*.bin"))
                    or any(repo_cache.rglob("*.safetensors"))
                    or any(repo_cache.rglob("*.pt"))
                    or any(repo_cache.rglob("*.pth"))
                    or any(repo_cache.rglob("*.npz"))
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
        backend_variant=os.environ.get(
            "VOICEBOX_BACKEND_VARIANT",
            "cuda" if torch.cuda.is_available() else ("xpu" if has_xpu else "cpu"),
        ),
        gpu_compatibility_warning=gpu_compat_warning,
    )


@router.get("/health/filesystem", response_model=models.FilesystemHealthResponse)
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
                path=str(dir_path.resolve()),
                exists=exists,
                writable=writable,
                error=error,
            )
        )

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
