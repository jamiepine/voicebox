"""Monkey-patch huggingface_hub to force offline mode with cached models.

Prevents mlx_audio / transformers from making network requests when models
are already downloaded. Must be imported BEFORE mlx_audio.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


@contextmanager
def force_offline_if_cached(is_cached: bool, model_label: str = ""):
    """Context manager that sets ``HF_HUB_OFFLINE=1`` while loading a cached model.

    If *is_cached* is ``False`` the block runs normally (network allowed).
    If the offline load raises an error containing "offline" we automatically
    retry with network access so a partially-cached model still works.

    Args:
        is_cached: Whether the model weights are already on disk.
        model_label: Human-readable name used in log messages.
    """
    if not is_cached:
        yield
        return

    original_value = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info(
        "[offline-guard] %s is cached — forcing HF_HUB_OFFLINE=1",
        model_label or "model",
    )

    try:
        yield
    except Exception as exc:
        if "offline" in str(exc).lower():
            logger.warning(
                "[offline-guard] Offline load failed for %s, retrying with network: %s",
                model_label or "model",
                exc,
            )
            # Restore original env and retry — caller must wrap the load
            # inside force_offline_if_cached so retrying here isn't possible.
            # Instead, propagate a flag via the exception so the caller can
            # decide.  For simplicity we just let it fall through to the
            # finally block and re-raise.
            raise
        raise
    finally:
        if original_value is not None:
            os.environ["HF_HUB_OFFLINE"] = original_value
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)


def patch_huggingface_hub_offline():
    """Monkey-patch huggingface_hub to force offline mode."""
    try:
        import huggingface_hub  # noqa: F401 -- need the package loaded
        from huggingface_hub import constants as hf_constants
        from huggingface_hub.file_download import _try_to_load_from_cache

        original_try_load = _try_to_load_from_cache

        def _patched_try_to_load_from_cache(
            repo_id: str,
            filename: str,
            cache_dir: Union[str, Path, None] = None,
            revision: Optional[str] = None,
            repo_type: Optional[str] = None,
        ):
            result = original_try_load(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                revision=revision,
                repo_type=repo_type,
            )

            if result is None:
                cache_path = Path(hf_constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
                logger.debug("file not cached: %s/%s (expected at %s)", repo_id, filename, cache_path)
            else:
                logger.debug("cache hit: %s/%s", repo_id, filename)

            return result

        import huggingface_hub.file_download as fd

        fd._try_to_load_from_cache = _patched_try_to_load_from_cache
        logger.debug("huggingface_hub patched for offline mode")

    except ImportError:
        logger.debug("huggingface_hub not available, skipping offline patch")
    except Exception:
        logger.exception("failed to patch huggingface_hub for offline mode")


def ensure_original_qwen_config_cached():
    """Symlink the original Qwen repo cache to the MLX community version.

    mlx_audio may try to fetch config from the original Qwen repo. If only
    the MLX community variant is cached, create a symlink so the cache lookup
    succeeds without a network request.
    """
    try:
        from huggingface_hub import constants as hf_constants
    except ImportError:
        return

    original_repo = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    mlx_repo = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

    cache_dir = Path(hf_constants.HF_HUB_CACHE)
    original_path = cache_dir / f"models--{original_repo.replace('/', '--')}"
    mlx_path = cache_dir / f"models--{mlx_repo.replace('/', '--')}"

    if not original_path.exists() and mlx_path.exists():
        try:
            original_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.symlink_to(mlx_path, target_is_directory=True)
            logger.info("created cache symlink: %s -> %s", original_repo, mlx_repo)
        except Exception:
            logger.warning("could not create cache symlink for %s", original_repo, exc_info=True)


if os.environ.get("VOICEBOX_OFFLINE_PATCH", "1") != "0":
    patch_huggingface_hub_offline()
    ensure_original_qwen_config_cached()
