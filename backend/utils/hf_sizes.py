"""
Query HuggingFace API for model repository sizes.
Caches results on disk with a 1-day TTL plus in-memory for the server lifetime.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# In-memory cache: repo_id -> size_mb
_size_cache: Dict[str, Optional[float]] = {}

# Disk cache: ~/.cache/voicebox/hf_sizes.json
_CACHE_FILE = Path.home() / ".cache" / "voicebox" / "hf_sizes.json"
_CACHE_TTL = 86400  # 1 day in seconds
_disk_cache_loaded = False


def _load_disk_cache() -> Dict:
    """Load disk cache, returning {repo_id: {size_mb: float|null, ts: epoch}}."""
    global _disk_cache_loaded
    _disk_cache_loaded = True
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_disk_cache(cache: Dict) -> None:
    """Persist cache to disk."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        logger.debug(f"Failed to save HF size cache: {e}")


async def get_repo_size_mb(repo_id: str) -> Optional[float]:
    """
    Get the total size of a HuggingFace model repo in MB.

    Uses a 1-day disk cache to avoid repeated API calls across restarts.
    Returns None if the query fails (network error, repo not found, etc).
    """
    # Check in-memory cache first (includes None for repos with no size info)
    if repo_id in _size_cache:
        return _size_cache[repo_id]

    # Check disk cache (1-day TTL) — only caches successful (non-None) results
    disk_cache = _load_disk_cache()
    entry = disk_cache.get(repo_id)
    if entry and time.time() - entry.get("ts", 0) < _CACHE_TTL:
        size_mb = entry.get("size_mb")
        if size_mb is not None:
            _size_cache[repo_id] = size_mb
            return size_mb

    # Fetch from HuggingFace API
    size_mb = await _fetch_repo_size(repo_id)

    # Always cache in memory (avoids repeated API calls within same session)
    _size_cache[repo_id] = size_mb

    # Only persist non-None results to disk (don't cache failures for a day)
    if size_mb is not None:
        disk_cache[repo_id] = {"size_mb": size_mb, "ts": time.time()}
        _save_disk_cache(disk_cache)

    return size_mb


async def _fetch_repo_size(repo_id: str) -> Optional[float]:
    """Fetch model repo size from the HuggingFace API."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"https://huggingface.co/api/models/{repo_id}")
            if resp.status_code != 200:
                logger.warning(f"HF API returned {resp.status_code} for {repo_id}")
                return None

            data = resp.json()

            # Method 1: Use safetensors metadata if available
            safetensors = data.get("safetensors")
            if safetensors and "total" in safetensors:
                return safetensors["total"] / (1024 * 1024)

            # Method 2: Sum sibling file sizes
            siblings = data.get("siblings", [])
            total_bytes = sum(s.get("size", 0) for s in siblings if s.get("size"))
            if total_bytes > 0:
                return total_bytes / (1024 * 1024)

            return None

    except Exception as e:
        logger.warning(f"Failed to query HF for {repo_id}: {e}")
        return None
