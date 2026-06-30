"""Model management endpoints."""

import asyncio
import shutil
from contextlib import suppress
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import config as backend_config, models
from ..services.task_queue import create_background_task
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager

router = APIRouter()
MODEL_CACHE_DIR_PREFIX = "models--"


def _resolve_missing_leaf(path: Path) -> Path:
    """Resolve a path whose final component may not exist yet."""
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        try:
            return path.parent.resolve(strict=True) / path.name
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail="Destination parent directory does not exist",
            ) from exc


def _resolve_migration_destination(raw_destination: str, source: Path) -> Path:
    """Resolve and constrain a model migration destination.

    The API is unauthenticated, so migration is only allowed into Voicebox's
    app-owned model storage root or one direct child of that root.
    """
    if not raw_destination.strip():
        raise HTTPException(status_code=400, detail="Destination is required")

    owned_root_path = backend_config.get_models_dir()
    if owned_root_path.is_symlink():
        raise HTTPException(status_code=400, detail="Voicebox models directory cannot be a symlink")

    owned_root = owned_root_path.resolve()
    requested = Path(raw_destination).expanduser()
    candidate = requested if requested.is_absolute() else owned_root / requested
    if candidate.is_symlink():
        raise HTTPException(status_code=400, detail="Destination cannot be a symlink")

    destination = _resolve_missing_leaf(candidate)

    try:
        relative_destination = destination.relative_to(owned_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Destination must be inside the Voicebox-owned models directory",
        ) from exc

    if len(relative_destination.parts) > 1:
        raise HTTPException(
            status_code=400,
            detail="Destination must be the Voicebox models directory or one direct child",
        )

    source = source.resolve()
    if source == destination:
        raise HTTPException(status_code=400, detail="Source and destination are the same directory")

    if destination.is_relative_to(source):
        raise HTTPException(status_code=400, detail="Destination cannot be inside the current cache directory")

    return destination


def _model_cache_dirs(source: Path) -> list[Path]:
    """Return top-level HuggingFace model cache directories eligible to migrate."""
    return [
        item
        for item in source.iterdir()
        if item.name.startswith(MODEL_CACHE_DIR_PREFIX) and item.is_dir() and not item.is_symlink()
    ]


def _validate_no_destination_collisions(model_dirs: list[Path], destination: Path) -> None:
    """Prevent migration from deleting or overwriting destination model dirs."""
    collisions = [
        item.name for item in model_dirs if (destination / item.name).exists() or (destination / item.name).is_symlink()
    ]
    if collisions:
        names = ", ".join(sorted(collisions))
        raise HTTPException(
            status_code=409,
            detail=f"Destination already contains model cache directories: {names}",
        )


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _copy_with_progress(src: Path, dst: Path, progress_manager, copied_so_far: int, total_bytes: int) -> int:
    """Copy a directory tree with byte-level progress tracking."""
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
                "migration",
                copied_so_far,
                total_bytes,
                filename=item.name,
                status="downloading",
            )
    return copied_so_far


@router.post("/models/load")
async def load_model(model_size: str = "1.7B"):
    """Manually load TTS model."""
    from ..services import tts

    try:
        tts_model = tts.get_tts_model()
        await tts_model.load_model_async(model_size)
        return {"message": f"Model {model_size} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unload")
async def unload_model():
    """Unload the default Qwen TTS model to free memory."""
    from ..services import tts

    try:
        tts.unload_tts_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/unload")
async def unload_model_by_name(model_name: str):
    """Unload a specific model from memory without deleting it from disk."""
    from ..backends import get_model_config, unload_model_by_config

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        was_loaded = unload_model_by_config(config)
        if not was_loaded:
            return {"message": f"Model {model_name} is not loaded"}
        return {"message": f"Model {model_name} unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/progress/{model_name}")
async def get_model_progress(model_name: str):
    """Get model download progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
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


@router.get("/models/cache-dir")
async def get_models_cache_dir():
    """Get the path to the HuggingFace model cache directory."""
    from huggingface_hub import constants as hf_constants

    return {"path": str(Path(hf_constants.HF_HUB_CACHE))}


@router.post("/models/migrate")
async def migrate_models(request: models.ModelMigrateRequest):
    """Move all downloaded models to a new directory with byte-level progress via SSE."""
    from huggingface_hub import constants as hf_constants

    source = Path(hf_constants.HF_HUB_CACHE).resolve()

    if not source.exists():
        raise HTTPException(status_code=404, detail="Current model cache directory not found")

    destination = _resolve_migration_destination(request.destination, source)

    progress_manager = get_progress_manager()
    model_dirs = _model_cache_dirs(source)
    if not model_dirs:
        progress_manager.update_progress("migration", 1, 1, status="complete")
        progress_manager.mark_complete("migration")
        return {"moved": 0, "errors": [], "source": str(source), "destination": str(destination)}

    _validate_no_destination_collisions(model_dirs, destination)
    destination.mkdir(parents=True, exist_ok=True)

    same_fs = False
    with suppress(OSError):
        same_fs = source.stat().st_dev == destination.stat().st_dev

    async def migrate_background():
        moved = 0
        errors = []
        try:
            if same_fs:
                total = len(model_dirs)
                for i, item in enumerate(model_dirs):
                    dest_item = destination / item.name
                    try:
                        shutil.move(str(item), str(dest_item))
                        moved += 1
                        progress_manager.update_progress(
                            "migration",
                            i + 1,
                            total,
                            filename=item.name,
                            status="downloading",
                        )
                    except Exception as e:
                        errors.append(f"{item.name}: {e!s}")
            else:
                total_bytes = sum(_get_dir_size(d) for d in model_dirs)
                progress_manager.update_progress(
                    "migration", 0, total_bytes, filename="Calculating...", status="downloading"
                )

                copied = 0
                for item in model_dirs:
                    dest_item = destination / item.name
                    try:
                        copied = await asyncio.to_thread(
                            _copy_with_progress, item, dest_item, progress_manager, copied, total_bytes
                        )
                        await asyncio.to_thread(shutil.rmtree, str(item))
                        moved += 1
                    except Exception as e:
                        errors.append(f"{item.name}: {e!s}")

            progress_manager.update_progress("migration", 1, 1, status="complete")
            progress_manager.mark_complete("migration")
        except Exception as e:
            progress_manager.update_progress("migration", 0, 0, status="error")
            progress_manager.mark_error("migration", str(e))

    create_background_task(migrate_background())

    return {"source": str(source), "destination": str(destination)}


@router.get("/models/migrate/progress")
async def get_migration_progress():
    """Get model migration progress via Server-Sent Events."""
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


@router.get("/models/status", response_model=models.ModelStatusListResponse)
async def get_model_status():
    """Get status of all available models."""
    from huggingface_hub import constants as hf_constants

    task_manager = get_task_manager()

    active_download_names = {task.model_name for task in task_manager.get_active_downloads()}

    try:
        from huggingface_hub import scan_cache_dir

        use_scan_cache = True
    except ImportError:
        use_scan_cache = False

    from ..backends import check_model_loaded, get_all_model_configs

    registry_configs = get_all_model_configs()
    model_configs = [
        {
            "model_name": cfg.model_name,
            "display_name": cfg.display_name,
            "hf_repo_id": cfg.hf_repo_id,
            "model_size": cfg.model_size,
            "check_loaded": lambda c=cfg: check_model_loaded(c),
        }
        for cfg in registry_configs
    ]

    model_to_repo = {cfg["model_name"]: cfg["hf_repo_id"] for cfg in model_configs}
    active_download_repos = {model_to_repo.get(name) for name in active_download_names if name in model_to_repo}

    cache_info = None
    if use_scan_cache:
        try:
            cache_info = scan_cache_dir()
        except Exception:
            pass

    statuses = []

    for config in model_configs:
        try:
            downloaded = False
            size_mb = None
            loaded = False

            if cache_info:
                repo_id = config["hf_repo_id"]
                for repo in cache_info.repos:
                    if repo.repo_id == repo_id:
                        has_model_weights = False
                        for rev in repo.revisions:
                            for f in rev.files:
                                fname = f.file_name.lower()
                                if fname.endswith((".safetensors", ".bin", ".pt", ".pth", ".npz")):
                                    has_model_weights = True
                                    break
                            if has_model_weights:
                                break

                        has_incomplete = False
                        try:
                            cache_dir = hf_constants.HF_HUB_CACHE
                            blobs_dir = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "blobs"
                            if blobs_dir.exists():
                                has_incomplete = any(blobs_dir.glob("*.incomplete"))
                        except Exception:
                            pass

                        if has_model_weights and not has_incomplete:
                            downloaded = True
                            try:
                                total_size = sum(revision.size_on_disk for revision in repo.revisions)
                                size_mb = total_size / (1024 * 1024)
                            except Exception:
                                pass
                        break

            if not downloaded:
                try:
                    cache_dir = hf_constants.HF_HUB_CACHE
                    repo_cache = Path(cache_dir) / ("models--" + config["hf_repo_id"].replace("/", "--"))

                    if repo_cache.exists():
                        blobs_dir = repo_cache / "blobs"
                        has_incomplete = blobs_dir.exists() and any(blobs_dir.glob("*.incomplete"))

                        if not has_incomplete:
                            snapshots_dir = repo_cache / "snapshots"
                            has_model_files = False
                            if snapshots_dir.exists():
                                has_model_files = (
                                    any(snapshots_dir.rglob("*.bin"))
                                    or any(snapshots_dir.rglob("*.safetensors"))
                                    or any(snapshots_dir.rglob("*.pt"))
                                    or any(snapshots_dir.rglob("*.pth"))
                                    or any(snapshots_dir.rglob("*.npz"))
                                )

                            if has_model_files:
                                downloaded = True
                                try:
                                    total_size = sum(
                                        f.stat().st_size
                                        for f in repo_cache.rglob("*")
                                        if f.is_file() and not f.name.endswith(".incomplete")
                                    )
                                    size_mb = total_size / (1024 * 1024)
                                except Exception:
                                    pass
                except Exception:
                    pass

            try:
                loaded = config["check_loaded"]()
            except Exception:
                loaded = False

            is_downloading = config["hf_repo_id"] in active_download_repos

            if is_downloading:
                downloaded = False
                size_mb = None

            statuses.append(
                models.ModelStatus(
                    model_name=config["model_name"],
                    display_name=config["display_name"],
                    hf_repo_id=config["hf_repo_id"],
                    downloaded=downloaded,
                    downloading=is_downloading,
                    size_mb=size_mb,
                    loaded=loaded,
                )
            )
        except Exception:
            try:
                loaded = config["check_loaded"]()
            except Exception:
                loaded = False

            is_downloading = config["hf_repo_id"] in active_download_repos

            statuses.append(
                models.ModelStatus(
                    model_name=config["model_name"],
                    display_name=config["display_name"],
                    hf_repo_id=config["hf_repo_id"],
                    downloaded=False,
                    downloading=is_downloading,
                    size_mb=None,
                    loaded=loaded,
                )
            )

    return models.ModelStatusListResponse(models=statuses)


@router.post("/models/download")
async def trigger_model_download(request: models.ModelDownloadRequest):
    """Trigger download of a specific model."""
    from ..backends import get_model_config, get_model_load_func

    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    config = get_model_config(request.model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")

    load_func = get_model_load_func(config)

    async def download_in_background():
        try:
            result = load_func()
            if asyncio.iscoroutine(result):
                await result
            task_manager.complete_download(request.model_name)
        except Exception as e:
            task_manager.error_download(request.model_name, str(e))

    task_manager.start_download(request.model_name)

    progress_manager.update_progress(
        model_name=request.model_name,
        current=0,
        total=0,
        filename="Connecting to HuggingFace...",
        status="downloading",
    )

    create_background_task(download_in_background())

    return {"message": f"Model {request.model_name} download started"}


@router.post("/models/download/cancel")
async def cancel_model_download(request: models.ModelDownloadRequest):
    """Cancel or dismiss an errored/stale download task."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    removed = task_manager.cancel_download(request.model_name)

    progress_removed = False
    with progress_manager._lock:
        if request.model_name in progress_manager._progress:
            del progress_manager._progress[request.model_name]
            progress_removed = True

    if removed or progress_removed:
        return {"message": f"Download task for {request.model_name} cancelled"}
    return {"message": f"No active task found for {request.model_name}"}


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model from the HuggingFace cache."""
    from huggingface_hub import constants as hf_constants

    from ..backends import get_model_config, unload_model_by_config

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    hf_repo_id = config.hf_repo_id

    try:
        unload_model_by_config(config)

        cache_dir = hf_constants.HF_HUB_CACHE
        repo_cache_dir = Path(cache_dir) / ("models--" + hf_repo_id.replace("/", "--"))

        if not repo_cache_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in cache")

        try:
            shutil.rmtree(repo_cache_dir)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete model cache directory: {str(e)}")

        return {"message": f"Model {model_name} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")
