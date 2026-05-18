"""
Voice prompt caching utilities.
"""

import hashlib
import logging
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any

from .. import config

logger = logging.getLogger(__name__)


def _get_cache_dir() -> Path:
    """Get cache directory from config."""
    return config.get_cache_dir()


# In-memory cache - can store dict (voice prompt) or tensor (legacy)
_memory_cache: dict[str, Union[torch.Tensor, Dict[str, Any]]] = {}


def _move_prompt_to_cpu(value: Any) -> Any:
    """Return a CPU-only copy of a cached voice prompt structure.

    SRT2Voice deliberately unloads CUDA models between heavy steps. Keeping a
    cached voice prompt tensor on CUDA can pin VRAM and can also make the next
    generation reuse tensors tied to a previous model/device lifetime.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_prompt_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_prompt_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_prompt_to_cpu(item) for item in value)
    return value


def get_cache_key(audio_path: str, reference_text: str) -> str:
    """
    Generate cache key from audio file and reference text.

    Args:
        audio_path: Path to audio file
        reference_text: Reference text

    Returns:
        Cache key (SHA-256 hash)
    """
    digest = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    digest.update(reference_text.encode("utf-8"))
    return digest.hexdigest()


def get_cached_voice_prompt(
    cache_key: str,
) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
    """
    Get cached voice prompt if available.

    Args:
        cache_key: Cache key

    Returns:
        Cached voice prompt (dict or tensor) or None
    """
    # Check in-memory cache
    if cache_key in _memory_cache:
        return _memory_cache[cache_key]

    # Check disk cache
    cache_file = _get_cache_dir() / f"{cache_key}.prompt"
    if cache_file.exists():
        try:
            prompt = torch.load(cache_file, weights_only=True, map_location="cpu")
            prompt = _move_prompt_to_cpu(prompt)
            _memory_cache[cache_key] = prompt
            return prompt
        except Exception:
            # Cache file corrupted, delete it
            cache_file.unlink()

    return None


def cache_voice_prompt(
    cache_key: str,
    voice_prompt: Union[torch.Tensor, Dict[str, Any]],
) -> None:
    """
    Cache voice prompt to memory and disk.

    Args:
        cache_key: Cache key
        voice_prompt: Voice prompt (dict or tensor)
    """
    voice_prompt_cpu = _move_prompt_to_cpu(voice_prompt)

    # Store in memory
    _memory_cache[cache_key] = voice_prompt_cpu

    # Store on disk (torch.save can handle both dicts and tensors)
    cache_file = _get_cache_dir() / f"{cache_key}.prompt"
    torch.save(voice_prompt_cpu, cache_file)


def clear_voice_prompt_memory_cache() -> int:
    """Clear only in-process voice prompt cache.

    Disk prompt files stay available, so the next generation can reload the
    prompt without recomputing it while still allowing CUDA memory to be
    released after unload-heavy workflows such as SRT2Voice.
    """
    count = len(_memory_cache)
    _memory_cache.clear()
    return count


def clear_voice_prompt_cache() -> int:
    """
    Clear all voice prompt caches (memory and disk).

    Returns:
        Number of cache files deleted
    """
    # Clear memory cache
    _memory_cache.clear()

    # Clear disk cache
    cache_dir = _get_cache_dir()
    deleted_count = 0

    if cache_dir.exists():
        # Delete prompt cache files
        for cache_file in cache_dir.glob("*.prompt"):
            try:
                cache_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning("Failed to delete cache file %s: %s", cache_file, e)

        # Delete combined audio files
        for audio_file in cache_dir.glob("combined_*.wav"):
            try:
                audio_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning("Failed to delete combined audio file %s: %s", audio_file, e)

    return deleted_count


def clear_profile_cache(profile_id: str) -> int:
    """
    Clear cache files for a specific profile.

    Args:
        profile_id: Profile ID

    Returns:
        Number of cache files deleted
    """
    cache_dir = _get_cache_dir()
    deleted_count = 0

    if cache_dir.exists():
        # Delete combined audio files for this profile
        pattern = f"combined_{profile_id}_*.wav"
        for audio_file in cache_dir.glob(pattern):
            try:
                audio_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning("Failed to delete combined audio file %s: %s", audio_file, e)

    return deleted_count
