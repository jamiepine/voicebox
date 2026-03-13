"""
CUDA backend binary download, assembly, and verification.

Downloads split parts of the CUDA-enabled voicebox-server binary from
GitHub Releases, reassembles them, verifies integrity via SHA-256,
and places the binary in the app's data directory for use on next
backend restart.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .config import get_data_dir
from .utils.progress import get_progress_manager
from . import __version__

logger = logging.getLogger(__name__)

GITHUB_RELEASES_URL = "https://github.com/jamiepine/voicebox/releases/download"

PROGRESS_KEY = "cuda-backend"


def get_backends_dir() -> Path:
    """Directory where downloaded backend binaries are stored."""
    d = get_data_dir() / "backends"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cuda_binary_name() -> str:
    """Platform-specific CUDA binary filename."""
    if sys.platform == "win32":
        return "voicebox-server-cuda.exe"
    return "voicebox-server-cuda"


def get_cuda_binary_path() -> Optional[Path]:
    """Return path to CUDA binary if it exists."""
    p = get_backends_dir() / get_cuda_binary_name()
    if p.exists():
        return p
    return None


def is_cuda_active() -> bool:
    """Check if the current process is the CUDA binary.

    The CUDA binary sets this env var on startup (see server.py).
    """
    return os.environ.get("VOICEBOX_BACKEND_VARIANT") == "cuda"


def get_cuda_status() -> dict:
    """Get current CUDA backend status for the API."""
    progress_manager = get_progress_manager()
    cuda_path = get_cuda_binary_path()
    progress = progress_manager.get_progress(PROGRESS_KEY)

    return {
        "available": cuda_path is not None,
        "active": is_cuda_active(),
        "binary_path": str(cuda_path) if cuda_path else None,
        "downloading": progress is not None and progress.get("status") == "downloading",
        "download_progress": progress,
    }


async def download_cuda_binary(version: Optional[str] = None):
    """Download the CUDA backend binary from GitHub Releases.

    Downloads split parts listed in a manifest file, concatenates them,
    and verifies the SHA-256 checksum for integrity. Atomic write
    (temp file -> rename).

    Args:
        version: Version tag (e.g. "v0.2.0"). Defaults to current app version.
    """
    import httpx

    if version is None:
        version = f"v{__version__}"

    progress = get_progress_manager()
    binary_name = get_cuda_binary_name()
    dest_dir = get_backends_dir()
    final_path = dest_dir / binary_name
    temp_path = dest_dir / f"{binary_name}.download"

    # Clean up any leftover partial download
    if temp_path.exists():
        temp_path.unlink()

    logger.info(f"Starting CUDA backend download for {version}")
    progress.update_progress(
        PROGRESS_KEY, current=0, total=0,
        filename="Fetching manifest...", status="downloading",
    )

    base_url = f"{GITHUB_RELEASES_URL}/{version}"
    stem = Path(binary_name).stem  # voicebox-server-cuda

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            # Fetch the manifest (list of split part filenames)
            manifest_url = f"{base_url}/{stem}.manifest"
            manifest_resp = await client.get(manifest_url)
            manifest_resp.raise_for_status()
            parts = [p.strip() for p in manifest_resp.text.strip().splitlines() if p.strip()]

            if not parts:
                raise ValueError("Empty manifest — no split parts found")

            logger.info(f"Found {len(parts)} split parts to download")

            # Fetch expected checksum (optional — for integrity verification)
            expected_sha = None
            try:
                sha_url = f"{base_url}/{stem}.sha256"
                sha_resp = await client.get(sha_url)
                if sha_resp.status_code == 200:
                    # Format: "sha256hex  filename\n"
                    expected_sha = sha_resp.text.strip().split()[0]
                    logger.info(f"Expected SHA-256: {expected_sha[:16]}...")
            except Exception as e:
                logger.warning(f"Could not fetch checksum file — skipping verification: {e}")

            # Download and concatenate parts
            total_downloaded = 0
            with open(temp_path, "wb") as f:
                for i, part_name in enumerate(parts):
                    part_url = f"{base_url}/{part_name}"
                    logger.info(f"Downloading part {i + 1}/{len(parts)}: {part_name}")

                    async with client.stream("GET", part_url) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                            total_downloaded += len(chunk)
                            progress.update_progress(
                                PROGRESS_KEY, current=total_downloaded, total=0,
                                filename=f"Part {i + 1}/{len(parts)}",
                                status="downloading",
                            )

        # Verify integrity if checksum was available
        if expected_sha:
            progress.update_progress(
                PROGRESS_KEY, current=total_downloaded, total=total_downloaded,
                filename="Verifying integrity...", status="downloading",
            )
            sha256 = hashlib.sha256()
            with open(temp_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    sha256.update(chunk)

            actual = sha256.hexdigest()
            if actual != expected_sha:
                raise ValueError(
                    f"Integrity check failed: expected {expected_sha[:16]}..., "
                    f"got {actual[:16]}..."
                )
            logger.info(f"Integrity verified: {actual[:16]}...")

        # Atomic move into place (replace handles existing target on all platforms)
        temp_path.replace(final_path)

        # Make executable on Unix
        if sys.platform != "win32":
            final_path.chmod(0o755)

        logger.info(f"CUDA backend downloaded to {final_path}")
        progress.mark_complete(PROGRESS_KEY)

    except Exception as e:
        # Clean up on failure
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"CUDA backend download failed: {e}")
        progress.mark_error(PROGRESS_KEY, str(e))
        raise


async def delete_cuda_binary() -> bool:
    """Delete the downloaded CUDA binary. Returns True if deleted."""
    path = get_cuda_binary_path()
    if path and path.exists():
        path.unlink()
        logger.info(f"Deleted CUDA binary: {path}")
        return True
    return False
