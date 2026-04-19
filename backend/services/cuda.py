"""
CUDA backend download, assembly, and verification.

Downloads two archives from GitHub Releases:
  1. Server core (voicebox-server-cuda.tar.gz) — the exe + non-NVIDIA deps,
     versioned with the app.
  2. CUDA libs (cuda-libs-{version}.tar.gz) — NVIDIA runtime libraries,
     versioned independently (only redownloaded on CUDA toolkit bump).

Both archives are extracted into {data_dir}/backends/cuda/ which forms the
complete PyInstaller --onedir directory structure that torch expects.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import tarfile
from pathlib import Path
from typing import Optional

from ..config import get_data_dir
from ..utils.progress import get_progress_manager
from .. import __version__

logger = logging.getLogger(__name__)

GITHUB_RELEASES_URL = "https://github.com/jamiepine/voicebox/releases/download"

PROGRESS_KEY = "cuda-backend"

# The current expected CUDA libs version.  Bump this when we change the
# CUDA toolkit version or torch's CUDA dependency changes (e.g. cu126 -> cu128).
CUDA_LIBS_VERSION = "cu128-v1"

# Prevents concurrent download_cuda_binary() calls from racing on the same
# temp file.  The auto-update background task and the manual HTTP endpoint
# can both invoke download_cuda_binary(); without this lock the progress-
# manager status check is a TOCTOU race.
_download_lock = asyncio.Lock()


def get_backends_dir() -> Path:
    """Directory where downloaded backend binaries are stored."""
    d = get_data_dir() / "backends"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cuda_dir() -> Path:
    """Directory where the CUDA backend (onedir) is extracted."""
    d = get_backends_dir() / "cuda"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cuda_exe_name() -> str:
    """Platform-specific CUDA executable filename."""
    if sys.platform == "win32":
        return "voicebox-server-cuda.exe"
    return "voicebox-server-cuda"


def get_cuda_binary_path() -> Optional[Path]:
    """Return path to the CUDA executable if it exists inside the onedir."""
    p = get_cuda_dir() / get_cuda_exe_name()
    if p.exists():
        return p
    return None


def get_cuda_libs_manifest_path() -> Path:
    """Path to the cuda-libs.json manifest inside the CUDA dir."""
    return get_cuda_dir() / "cuda-libs.json"


def get_installed_cuda_libs_version() -> Optional[str]:
    """Read the installed CUDA libs version from cuda-libs.json, or None."""
    manifest_path = get_cuda_libs_manifest_path()
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        return data.get("version")
    except Exception as e:
        logger.warning(f"Could not read cuda-libs.json: {e}")
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
    cuda_libs_version = get_installed_cuda_libs_version()

    return {
        "available": cuda_path is not None,
        "active": is_cuda_active(),
        "binary_path": str(cuda_path) if cuda_path else None,
        "cuda_libs_version": cuda_libs_version,
        "downloading": progress is not None and progress.get("status") == "downloading",
        "download_progress": progress,
    }


def _needs_server_download(version: Optional[str] = None) -> bool:
    """Check if the server core archive needs to be (re)downloaded."""
    cuda_path = get_cuda_binary_path()
    if not cuda_path:
        return True
    # Check if the binary version matches the expected app version
    installed = get_cuda_binary_version()
    expected = version or __version__
    if expected.startswith("v"):
        expected = expected[1:]
    return installed != expected


def _needs_cuda_libs_download() -> bool:
    """Check if the CUDA libs archive needs to be (re)downloaded."""
    installed = get_installed_cuda_libs_version()
    if installed is None:
        return True
    return installed != CUDA_LIBS_VERSION


async def _download_and_extract_archive(
    client,
    url: str,
    sha256_url: Optional[str],
    dest_dir: Path,
    label: str,
    progress_offset: int,
    total_size: int,
):
    """Download a .tar.gz archive and extract it into dest_dir.

    Args:
        client: httpx.AsyncClient
        url: URL of the .tar.gz archive
        sha256_url: URL of the .sha256 checksum file (optional)
        dest_dir: Directory to extract into
        label: Human-readable label for progress updates
        progress_offset: Byte offset for progress reporting (when downloading
            multiple archives sequentially)
        total_size: Total bytes across all downloads (for progress bar)
    """
    progress = get_progress_manager()
    temp_path = dest_dir / f".download-{label.replace(' ', '-')}.tmp"

    # Clean up leftover partial download
    if temp_path.exists():
        temp_path.unlink()

    # Fetch expected checksum (fail-fast: never extract an unverified archive)
    expected_sha = None
    if sha256_url:
        try:
            sha_resp = await client.get(sha256_url)
            sha_resp.raise_for_status()
            expected_sha = sha_resp.text.strip().split()[0]
            logger.info(f"{label}: expected SHA-256: {expected_sha[:16]}...")
        except Exception as e:
            raise RuntimeError(f"{label}: failed to fetch checksum from {sha256_url}") from e

    # Stream download, verify, and extract — always clean up temp file
    downloaded = 0
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress.update_progress(
                        PROGRESS_KEY,
                        current=progress_offset + downloaded,
                        total=total_size,
                        filename=f"Downloading {label}",
                        status="downloading",
                    )

        # Verify integrity
        if expected_sha:
            progress.update_progress(
                PROGRESS_KEY,
                current=progress_offset + downloaded,
                total=total_size,
                filename=f"Verifying {label}...",
                status="downloading",
            )
            sha256 = hashlib.sha256()
            with open(temp_path, "rb") as f:
                while True:
                    data = f.read(1024 * 1024)
                    if not data:
                        break
                    sha256.update(data)
            actual = sha256.hexdigest()
            if actual != expected_sha:
                raise ValueError(
                    f"{label} integrity check failed: expected {expected_sha[:16]}..., got {actual[:16]}..."
                )
            logger.info(f"{label}: integrity verified")

        # Extract (use data filter for path traversal protection on Python 3.12+)
        progress.update_progress(
            PROGRESS_KEY,
            current=progress_offset + downloaded,
            total=total_size,
            filename=f"Extracting {label}...",
            status="downloading",
        )
        with tarfile.open(temp_path, "r:gz") as tar:
            if sys.version_info >= (3, 12):
                tar.extractall(path=dest_dir, filter="data")
            else:
                tar.extractall(path=dest_dir)

        logger.info(f"{label}: extracted to {dest_dir}")
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return downloaded


async def download_cuda_binary(version: Optional[str] = None):
    """Download the CUDA backend (server core + CUDA libs if needed).

    Downloads both archives from GitHub Releases, extracts them into
    {data_dir}/backends/cuda/, and writes the cuda-libs.json manifest.

    Only downloads what's needed:
    - Server core: always redownloaded (versioned with app)
    - CUDA libs: only if missing or version mismatch

    Args:
        version: Version tag (e.g. "v0.3.0"). Defaults to current app version.
    """
    if _download_lock.locked():
        logger.info("CUDA download already in progress, skipping duplicate request")
        return
    async with _download_lock:
        await _download_cuda_binary_locked(version)


async def _download_cuda_binary_locked(version: Optional[str] = None):
    """Inner implementation of download_cuda_binary, called under _download_lock."""
    import httpx

    if version is None:
        version = f"v{__version__}"

    progress = get_progress_manager()
    cuda_dir = get_cuda_dir()

    need_server = _needs_server_download(version)
    need_libs = _needs_cuda_libs_download()

    if not need_server and not need_libs:
        logger.info("CUDA backend is up to date, nothing to download")
        return

    logger.info(
        f"Starting CUDA backend download for {version} "
        f"(server={'yes' if need_server else 'cached'}, "
        f"libs={'yes' if need_libs else 'cached'})"
    )
    progress.update_progress(
        PROGRESS_KEY,
        current=0,
        total=0,
        filename="Preparing download...",
        status="downloading",
    )

    base_url = f"{GITHUB_RELEASES_URL}/{version}"
    server_archive = "voicebox-server-cuda.tar.gz"
    libs_archive = f"cuda-libs-{CUDA_LIBS_VERSION}.tar.gz"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            # Estimate total download size
            total_size = 0
            if need_server:
                try:
                    head = await client.head(f"{base_url}/{server_archive}")
                    total_size += int(head.headers.get("content-length", 0))
                except Exception:
                    pass
            if need_libs:
                try:
                    head = await client.head(f"{base_url}/{libs_archive}")
                    total_size += int(head.headers.get("content-length", 0))
                except Exception:
                    pass

            logger.info(f"Total download size: {total_size / 1024 / 1024:.1f} MB")

            offset = 0

            # Download server core
            if need_server:
                server_downloaded = await _download_and_extract_archive(
                    client,
                    url=f"{base_url}/{server_archive}",
                    sha256_url=f"{base_url}/{server_archive}.sha256",
                    dest_dir=cuda_dir,
                    label="CUDA server",
                    progress_offset=offset,
                    total_size=total_size,
                )
                offset += server_downloaded

                # Make executable on Unix
                exe_path = cuda_dir / get_cuda_exe_name()
                if sys.platform != "win32" and exe_path.exists():
                    exe_path.chmod(0o755)

            # Download CUDA libs
            if need_libs:
                await _download_and_extract_archive(
                    client,
                    url=f"{base_url}/{libs_archive}",
                    sha256_url=f"{base_url}/{libs_archive}.sha256",
                    dest_dir=cuda_dir,
                    label="CUDA libraries",
                    progress_offset=offset,
                    total_size=total_size,
                )

                # Write local cuda-libs.json manifest
                manifest = {"version": CUDA_LIBS_VERSION}
                get_cuda_libs_manifest_path().write_text(json.dumps(manifest, indent=2) + "\n")

        logger.info(f"CUDA backend ready at {cuda_dir}")
        progress.mark_complete(PROGRESS_KEY)

    except Exception as e:
        logger.error(f"CUDA backend download failed: {e}")
        progress.mark_error(PROGRESS_KEY, str(e))
        raise


def get_cuda_binary_version() -> Optional[str]:
    """Get the version of the installed CUDA binary, or None if not installed."""
    import subprocess

    cuda_path = get_cuda_binary_path()
    if not cuda_path:
        return None
    try:
        result = subprocess.run(
            [str(cuda_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(cuda_path.parent),  # Run from the onedir directory
        )
        # Output format: "voicebox-server 0.3.0"
        for line in result.stdout.strip().splitlines():
            if "voicebox-server" in line:
                return line.split()[-1]
    except Exception as e:
        logger.warning(f"Could not get CUDA binary version: {e}")
    return None


async def check_and_update_cuda_binary():
    """Check if the CUDA binary is outdated and auto-download if so.

    Called on server startup. Checks both server version and CUDA libs
    version. Downloads only what's needed.
    """
    cuda_path = get_cuda_binary_path()
    if not cuda_path:
        return  # No CUDA binary installed, nothing to update

    need_server = _needs_server_download()
    need_libs = _needs_cuda_libs_download()

    if not need_server and not need_libs:
        logger.info(f"CUDA binary is up to date (server=v{__version__}, libs={get_installed_cuda_libs_version()})")
        return

    reasons = []
    if need_server:
        cuda_version = get_cuda_binary_version()
        reasons.append(f"server v{cuda_version} != v{__version__}")
    if need_libs:
        installed_libs = get_installed_cuda_libs_version()
        reasons.append(f"libs {installed_libs} != {CUDA_LIBS_VERSION}")

    logger.info(f"CUDA backend needs update ({', '.join(reasons)}). Auto-downloading...")

    try:
        await download_cuda_binary()
    except Exception as e:
        logger.error(f"Auto-update of CUDA binary failed: {e}")


async def delete_cuda_binary() -> bool:
    """Delete the downloaded CUDA backend directory. Returns True if deleted."""
    import shutil

    cuda_dir = get_cuda_dir()
    if cuda_dir.exists() and any(cuda_dir.iterdir()):
        shutil.rmtree(cuda_dir)
        logger.info(f"Deleted CUDA backend directory: {cuda_dir}")
        return True
    return False
