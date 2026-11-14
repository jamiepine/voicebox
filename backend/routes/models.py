"""Model management endpoints."""

import asyncio
import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import models
from ..utils.platform_detect import get_backend_type
from ..services.task_queue import create_background_task
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_dir_size(path: Path) -> int:
    """Get the on-disk size of a directory in bytes.

    Symlinks are skipped: in a HuggingFace cache every ``snapshots/<rev>/<file>``
    is a link to ``blobs/<hash>``, so following them counts each model twice.
    """
    total = 0
    for f in path.rglob("*"):
        if not f.is_symlink() and f.is_file():
            total += f.stat().st_size
    return total


def _internal_link_target(link: Path, root: Path) -> str | None:
    """Return *link*'s target relative to its own directory if it stays inside *root*.

    HuggingFace writes relative links (``../../blobs/<hash>``), but resolve
    absolute ones too so a cache that was linked absolutely still relocates.
    Returns ``None`` for links that point outside the tree being copied.
    """
    raw = os.readlink(link)
    # Windows may hand back extended-length paths for absolute links.
    if raw.startswith("\\\\?\\"):
        raw = raw[4:]
    target = Path(raw)
    absolute = target if target.is_absolute() else link.parent / target
    root_abs = Path(os.path.abspath(root))
    absolute = Path(os.path.abspath(absolute))
    if not absolute.is_relative_to(root_abs):
        return None
    return os.path.relpath(absolute, os.path.abspath(link.parent))


def _recreate_link(src_link: Path, dest_link: Path, rel_target: str) -> None:
    """Recreate a cache-internal symlink at the destination.

    Order of preference:
      1. A symlink with the same relative target (keeps the HF layout intact).
      2. A hard link to the already-copied target. Creating symlinks on Windows
         needs Developer Mode or admin rights and is often refused on network
         shares, but NTFS hard links need neither and cost no extra space.
      3. A plain copy of the target's contents (what huggingface_hub itself
         does when symlinks are unavailable).
    """
    try:
        os.symlink(rel_target, dest_link)
        return
    except OSError:
        pass

    dest_target = Path(os.path.normpath(dest_link.parent / rel_target))
    if dest_target.is_file():
        try:
            os.link(dest_target, dest_link)
            return
        except OSError:
            pass
        shutil.copy2(dest_target, dest_link)
    else:
        shutil.copy2(src_link, dest_link)


def _copy_with_progress(src: Path, dst: Path, progress_manager, copied_so_far: int, total_bytes: int) -> int:
    """Copy a model cache tree with byte-level progress tracking.

    Regular files are copied first; symlinks that point inside the tree (the
    HuggingFace ``snapshots`` -> ``blobs`` links) are recreated afterwards
    instead of being dereferenced, so a migrated model takes the same space it
    did before rather than double.
    """
    root = src
    pending_links: list[tuple[Path, Path, str]] = []

    def _walk(cur_src: Path, cur_dst: Path) -> None:
        nonlocal copied_so_far
        cur_dst.mkdir(parents=True, exist_ok=True)
        for item in cur_src.iterdir():
            dest_item = cur_dst / item.name
            if item.is_symlink():
                rel_target = _internal_link_target(item, root)
                if rel_target is not None:
                    pending_links.append((item, dest_item, rel_target))
                    continue
                # Link out of the tree: fall through and copy its contents.
            if item.is_dir():
                _walk(item, dest_item)
                continue
            size = item.stat().st_size
            shutil.copy2(str(item), str(dest_item))
            copied_so_far += size
            progress_manager.update_progress(
                "migration",
                min(copied_so_far, total_bytes) if total_bytes else copied_so_far,
                total_bytes,
                filename=item.name,
                status="downloading",
            )

    _walk(src, dst)
    for src_link, dest_link, rel_target in pending_links:
        _recreate_link(src_link, dest_link, rel_target)
    return copied_so_far


DEFAULT_QWEN_MODEL_SIZE = "1.7B"


def _resolve_model_config(model_name: str):
    """Look up a model config by name or raise a 400 listing the valid ids."""
    from ..backends import get_all_model_configs, get_model_config

    config = get_model_config(model_name)
    if config is None:
        known = sorted(cfg.model_name for cfg in get_all_model_configs())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name!r}. Available: {known}",
        )
    return config


@router.post("/models/load")
async def load_model(
    request: models.ModelLoadRequest | None = None,
    model_size: str | None = None,
):
    """Load a model into memory.

    Pass ``{"model_name": "kokoro"}`` to target any registered engine. Only an
    absent ``model_name`` falls back to the default Qwen TTS backend, selected
    by the ``model_size`` query parameter, which is what callers did before
    ``model_name`` existed. A supplied-but-empty name is a bad request, not a
    request for the default — silently loading Qwen there is the very bug this
    endpoint had.
    """
    from ..backends import get_model_load_func
    from ..services import tts

    requested_name = request.model_name if request else None
    requested_size = (request.model_size if request else None) or model_size

    if requested_name is not None:
        config = _resolve_model_config(requested_name)
        try:
            result = get_model_load_func(config)()
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {"message": f"Model {config.model_name} loaded successfully"}

    size = requested_size or DEFAULT_QWEN_MODEL_SIZE
    try:
        tts_model = tts.get_tts_model()
        await tts_model.load_model_async(size)
        return {"message": f"Model {size} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/unload")
async def unload_model(request: models.ModelLoadRequest | None = None):
    """Unload a model from memory.

    Pass ``{"model_name": "chatterbox-tts"}`` to target a specific engine —
    same behaviour as ``POST /models/{model_name}/unload``. Only an absent
    ``model_name`` unloads the default Qwen TTS model, as it always has; a
    supplied-but-empty name is a bad request.
    """
    from ..backends import unload_model_by_config
    from ..services import tts

    requested_name = request.model_name if request else None

    if requested_name is not None:
        config = _resolve_model_config(requested_name)
        try:
            was_loaded = unload_model_by_config(config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        if not was_loaded:
            return {"message": f"Model {config.model_name} is not loaded"}
        return {"message": f"Model {config.model_name} unloaded successfully"}

    try:
        tts.unload_tts_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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

    source = Path(hf_constants.HF_HUB_CACHE)
    destination = Path(request.destination)

    if not source.exists():
        raise HTTPException(status_code=404, detail="Current model cache directory not found")

    if source.resolve() == destination.resolve():
        raise HTTPException(status_code=400, detail="Source and destination are the same directory")

    if destination.resolve().is_relative_to(source.resolve()):
        raise HTTPException(status_code=400, detail="Destination cannot be inside the current cache directory")

    progress_manager = get_progress_manager()
    model_dirs = [d for d in source.iterdir() if d.name.startswith("models--") and d.is_dir()]
    if not model_dirs:
        progress_manager.update_progress("migration", 1, 1, status="complete")
        progress_manager.mark_complete("migration")
        return {"moved": 0, "errors": [], "source": str(source), "destination": str(destination)}

    destination.mkdir(parents=True, exist_ok=True)

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
                total = len(model_dirs)
                for i, item in enumerate(model_dirs):
                    dest_item = destination / item.name
                    try:
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
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
                        errors.append(f"{item.name}: {str(e)}")
            else:
                total_bytes = sum(_get_dir_size(d) for d in model_dirs)
                progress_manager.update_progress(
                    "migration", 0, total_bytes, filename="Calculating...", status="downloading"
                )

                copied = 0
                for item in model_dirs:
                    dest_item = destination / item.name
                    try:
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        copied = await asyncio.to_thread(
                            _copy_with_progress, item, dest_item, progress_manager, copied, total_bytes
                        )
                        await asyncio.to_thread(shutil.rmtree, str(item))
                        moved += 1
                    except Exception as e:
                        errors.append(f"{item.name}: {str(e)}")

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

    backend_type = get_backend_type()
    task_manager = get_task_manager()

    # Pending only — an errored task stays in the active list for the
    # error/retry UI, but reporting it as "downloading" here would mask
    # the model's real cache state until the app restarts (issue #925).
    active_download_names = {task.model_name for task in task_manager.get_pending_downloads()}

    try:
        from huggingface_hub import scan_cache_dir

        use_scan_cache = True
    except ImportError:
        use_scan_cache = False

    from ..backends import get_all_model_configs, check_model_loaded
    from ..backends.base import has_in_progress_download, is_model_cached

    registry_configs = get_all_model_configs()
    model_configs = [
        {
            "model_name": cfg.model_name,
            "display_name": cfg.display_name,
            "hf_repo_id": cfg.hf_repo_id,
            "aux_hf_repo_ids": cfg.aux_hf_repo_ids,
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
                            has_incomplete = has_in_progress_download(blobs_dir)
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
                        has_incomplete = has_in_progress_download(blobs_dir)

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

            # A model whose backend also pulls auxiliary repos at load time
            # (e.g. Chatterbox MLX's S3 tokenizer) is only downloaded once
            # those are present too, matching the backend's own cache check.
            if downloaded and not all(is_model_cached(repo) for repo in config["aux_hf_repo_ids"]):
                downloaded = False

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
    from ..backends import check_model_loaded, get_model_config, get_model_load_func, unload_model_by_config

    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    config = get_model_config(request.model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")

    load_func = get_model_load_func(config)

    try:
        was_loaded = check_model_loaded(config)
    except Exception:
        was_loaded = False

    async def download_in_background():
        try:
            result = load_func()
            if asyncio.iscoroutine(result):
                await result
            # Downloading works by loading the model, which leaves it resident
            # on the GPU. Queue a few downloads and VRAM fills up until later
            # ones fail with "CUDA out of memory". A download should leave
            # memory as it found it, so drop models it loaded only to fetch
            # their weights -- unless a generation may be using one by now.
            if not was_loaded and not task_manager.get_active_generations():
                try:
                    unload_model_by_config(config)
                except Exception as unload_error:
                    logger.warning(
                        "Could not unload %s after download: %s", request.model_name, unload_error
                    )
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
            for aux_repo_id in config.aux_hf_repo_ids:
                aux_cache_dir = Path(cache_dir) / ("models--" + aux_repo_id.replace("/", "--"))
                if aux_cache_dir.exists():
                    shutil.rmtree(aux_cache_dir)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete model cache directory: {str(e)}")

        return {"message": f"Model {model_name} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")
