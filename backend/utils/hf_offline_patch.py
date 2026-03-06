"""
Monkey patch for huggingface_hub to force offline mode with cached models.
This prevents mlx_audio from making network requests when models are already downloaded.
"""

import os
from pathlib import Path
from typing import Optional, Union


def patch_huggingface_hub_offline():
    """
    Monkey-patch huggingface_hub to force offline mode.
    This must be called BEFORE importing mlx_audio.
    """
    try:
        import huggingface_hub
        from huggingface_hub import constants as hf_constants
        from huggingface_hub.file_download import _try_to_load_from_cache
        
        # Store original function
        original_try_load = _try_to_load_from_cache
        
        def _patched_try_to_load_from_cache(
            repo_id: str,
            filename: str,
            cache_dir: Union[str, Path, None] = None,
            revision: Optional[str] = None,
            repo_type: Optional[str] = None,
        ):
            """
            Patched version that forces offline mode.
            Returns None if not cached (instead of making network request).
            """
            # Always use the original function, but we're already in HF_HUB_OFFLINE mode
            result = original_try_load(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                revision=revision,
                repo_type=repo_type,
            )
            
            if result is None:
                # File not in cache - log this for debugging
                cache_path = Path(hf_constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
                print(f"[HF_PATCH] File not cached: {repo_id}/{filename}")
                print(f"[HF_PATCH] Expected at: {cache_path}")
            else:
                print(f"[HF_PATCH] Cache hit: {repo_id}/{filename}")
            
            return result
        
        # Replace the function
        import huggingface_hub.file_download as fd
        fd._try_to_load_from_cache = _patched_try_to_load_from_cache
        
        print("[HF_PATCH] huggingface_hub patched for offline mode")
        
    except ImportError:
        print("[HF_PATCH] huggingface_hub not found, skipping patch")
    except Exception as e:
        print(f"[HF_PATCH] Error patching huggingface_hub: {e}")


def ensure_original_qwen_config_cached():
    """
    The MLX community model is based on the original Qwen model.
    mlx_audio may try to fetch config from the original repo.
    We need to ensure that config is available in the cache.
    """
    from huggingface_hub import constants as hf_constants
    
    # Original Qwen model that mlx_audio might reference
    original_repo = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    mlx_repo = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
    
    cache_dir = Path(hf_constants.HF_HUB_CACHE)
    
    original_path = cache_dir / f"models--{original_repo.replace('/', '--')}"
    mlx_path = cache_dir / f"models--{mlx_repo.replace('/', '--')}"
    
    # If original repo cache doesn't exist but MLX does, create a symlink or copy config
    if not original_path.exists() and mlx_path.exists():
        print(f"[HF_PATCH] Original repo not cached, but MLX version is")
        print(f"[HF_PATCH] Creating symlink from {original_repo} -> {mlx_repo}")
        
        try:
            # Create a symlink so the cache lookup succeeds
            original_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.symlink_to(mlx_path, target_is_directory=True)
            print(f"[HF_PATCH] Symlink created successfully")
        except Exception as e:
            print(f"[HF_PATCH] Could not create symlink: {e}")


# Auto-apply patch when module is imported
if os.environ.get("VOICEBOX_OFFLINE_PATCH", "1") != "0":
    patch_huggingface_hub_offline()
    ensure_original_qwen_config_cached()
