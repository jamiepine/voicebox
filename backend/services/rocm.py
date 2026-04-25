"""
ROCm backend download, assembly, and verification.

Downloads two archives from GitHub Releases:
  1. Server core (voicebox-server-rocm.tar.gz) — the exe + non-AMD deps,
     versioned with the app.
  2. ROCm libs (rocm-libs-{version}.tar.gz) — AMD runtime libraries,
     versioned independently (only redownloaded on ROCm toolkit bump).

Both archives are extracted into {data_dir}/backends/rocm/ which forms the
complete PyInstaller --onedir directory structure that torch expects.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional

from ..config import get_data_dir
from ..utils.progress import get_progress_manager
from .. import __version__

logger = logging.getLogger(__name__)

GITHUB_RELEASES_URL = "https://github.com/jamiepine/voicebox/releases/download"

PROGRESS_KEY = "rocm-backend"

# The current expected ROCm libs version.  Bump this when we change the
# ROCm toolkit version or torch's ROCm dependency changes (e.g. rocm7.2 -> rocm7.4).
ROCM_LIBS_VERSION = "rocm7.2-v1"

# Prevents concurrent download_rocm_binary() calls from racing on the same
# temp file.  The auto-update background task and the manual HTTP endpoint
# can both invoke download_rocm_binary(); without this lock the progress-
# manager status check is a TOCTOU race.
_download_lock = asyncio.Lock()


def get_backends_dir() -> Path:
    """Directory where downloaded backend binaries are stored."""
    d = get_data_dir() / "backends"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_rocm_dir() -> Path:
    """Directory where the ROCm backend (onedir) is extracted."""
    d = get_backends_dir() / "rocm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_rocm_exe_name() -> str:
    """Platform-specific ROCm executable filename."""
    if sys.platform == "win32":
        return "voicebox-server-rocm.exe"
    return "voicebox-server-rocm"


def get_rocm_binary_path() -> Optional[Path]:
    """Return path to the ROCm executable if it exists inside the onedir."""
    p = get_rocm_dir() / get_rocm_exe_name()
    if p.exists():
        return p
    return None


def get_rocm_libs_manifest_path() -> Path:
    """Path to the rocm-libs.json manifest inside the ROCm dir."""
    return get_rocm_dir() / "rocm-libs.json"


def get_installed_rocm_libs_version() -> Optional[str]:
    """Read the installed ROCm libs version from rocm-libs.json, or None."""
    manifest_path = get_rocm_libs_manifest_path()
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        return data.get("version")
    except Exception as e:
        logger.warning(f"Could not read rocm-libs.json: {e}")
        return None


def is_rocm_active() -> bool:
    """Check if the current process is the ROCm binary.

    The ROCm binary sets this env var on startup (see server.py).
    """
    return os.environ.get("VOICEBOX_BACKEND_VARIANT") == "rocm"


def get_rocm_status() -> dict:
    """Get current ROCm backend status for the API."""
    progress_manager = get_progress_manager()
    rocm_path = get_rocm_binary_path()
    progress = progress_manager.get_progress(PROGRESS_KEY)
    rocm_libs_version = get_installed_rocm_libs_version()

    return {
        "available": rocm_path is not None,
        "active": is_rocm_active(),
        "binary_path": str(rocm_path) if rocm_path else None,
        "rocm_libs_version": rocm_libs_version,
        "downloading": progress is not None and progress.get("status") == "downloading",
        "download_progress": progress,
    }


def _needs_server_download(version: Optional[str] = None) -> bool:
    """Check if the server core archive needs to be (re)downloaded."""
    rocm_path = get_rocm_binary_path()
    if not rocm_path:
        return True
    # Check if the binary version matches the expected app version
    installed = get_rocm_binary_version()
    expected = version or __version__
    if expected.startswith("v"):
        expected = expected[1:]
    return installed != expected


def _needs_rocm_libs_download() -> bool:
    """Check if the ROCm libs archive needs to be (re)downloaded."""
    installed = get_installed_rocm_libs_version()
    if installed is None:
        return True
    return installed != ROCM_LIBS_VERSION


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
            tar.extractall(path=dest_dir, filter="data")

        logger.info(f"{label}: extracted to {dest_dir}")
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return downloaded


async def download_rocm_binary(version: Optional[str] = None):
    """Download the ROCm backend (server core + ROCm libs if needed).

    Downloads both archives from GitHub Releases, extracts them into
    {data_dir}/backends/rocm/, and writes the rocm-libs.json manifest.

    Only downloads what's needed:
    - Server core: always redownloaded (versioned with app)
    - ROCm libs: only if missing or version mismatch

    Args:
        version: Version tag (e.g. "v0.3.0"). Defaults to current app version.
    """
    if _download_lock.locked():
        logger.info("ROCm download already in progress, skipping duplicate request")
        return
    async with _download_lock:
        await _download_rocm_binary_locked(version)


async def _download_rocm_binary_locked(version: Optional[str] = None):
    """Inner implementation of download_rocm_binary, called under _download_lock."""
    import httpx

    if version is None:
        version = f"v{__version__}"

    progress = get_progress_manager()
    rocm_dir = get_rocm_dir()

    need_server = _needs_server_download(version)
    need_libs = _needs_rocm_libs_download()

    if not need_server and not need_libs:
        logger.info("ROCm backend is up to date, nothing to download")
        return

    logger.info(
        f"Starting ROCm backend download for {version} "
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

    server_base_url = f"{GITHUB_RELEASES_URL}/{version}"
    libs_base_url = f"{GITHUB_RELEASES_URL}/{ROCM_LIBS_VERSION}"
    server_archive = "voicebox-server-rocm.tar.gz"
    libs_archive = f"rocm-libs-{ROCM_LIBS_VERSION}.tar.gz"

    # Always stage when any download is needed, then atomically rename over
    # rocm_dir on success. This prevents a failed mid-extraction from leaving
    # rocm_dir in a partially-installed state that still passes the
    # get_rocm_binary_path() existence check. Existing files are pre-copied
    # into staging so partial updates (e.g. libs-only or server-only) preserve
    # whatever isn't being re-downloaded.
    use_staging = need_server or need_libs
    staging_dir = get_backends_dir() / "rocm-staging"

    if use_staging:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        # Preserve existing files (server or libs) that don't need re-downloading.
        # Extracted archives will overwrite only what we actually download.
        if rocm_dir.exists():
            shutil.copytree(rocm_dir, staging_dir, dirs_exist_ok=True)
        extract_dir = staging_dir
    else:
        extract_dir = rocm_dir

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            # Estimate total download size
            total_size = 0
            if need_server:
                try:
                    head = await client.head(f"{server_base_url}/{server_archive}")
                    total_size += int(head.headers.get("content-length", 0))
                except Exception:
                    pass
            if need_libs:
                try:
                    head = await client.head(f"{libs_base_url}/{libs_archive}")
                    total_size += int(head.headers.get("content-length", 0))
                except Exception:
                    pass

            logger.info(f"Total download size: {total_size / 1024 / 1024:.1f} MB")

            offset = 0

            # Download server core
            if need_server:
                server_downloaded = await _download_and_extract_archive(
                    client,
                    url=f"{server_base_url}/{server_archive}",
                    sha256_url=f"{server_base_url}/{server_archive}.sha256",
                    dest_dir=extract_dir,
                    label="ROCm server",
                    progress_offset=offset,
                    total_size=total_size,
                )
                offset += server_downloaded

                # Make executable on Unix
                exe_path = extract_dir / get_rocm_exe_name()
                if sys.platform != "win32" and exe_path.exists():
                    exe_path.chmod(0o755)

            # Download ROCm libs
            if need_libs:
                await _download_and_extract_archive(
                    client,
                    url=f"{libs_base_url}/{libs_archive}",
                    sha256_url=f"{libs_base_url}/{libs_archive}.sha256",
                    dest_dir=extract_dir,
                    label="ROCm libraries",
                    progress_offset=offset,
                    total_size=total_size,
                )

                # Write local rocm-libs.json manifest
                manifest = {"version": ROCM_LIBS_VERSION}
                (extract_dir / "rocm-libs.json").write_text(json.dumps(manifest, indent=2) + "\n")

        # Atomic swap: replace rocm_dir with the fully-extracted staging dir
        if use_staging:
            backup_dir = get_backends_dir() / "rocm-backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            if rocm_dir.exists():
                rocm_dir.rename(backup_dir)
            try:
                staging_dir.rename(rocm_dir)
            except Exception:
                if backup_dir.exists() and not rocm_dir.exists():
                    backup_dir.rename(rocm_dir)
                raise
            else:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)

        logger.info(f"ROCm backend ready at {rocm_dir}")
        progress.mark_complete(PROGRESS_KEY)

    except Exception as e:
        if use_staging and staging_dir.exists():
            shutil.rmtree(staging_dir)
        logger.error(f"ROCm backend download failed: {e}")
        progress.mark_error(PROGRESS_KEY, str(e))
        raise


def get_rocm_binary_version() -> Optional[str]:
    """Get the version of the installed ROCm binary, or None if not installed."""
    import subprocess

    rocm_path = get_rocm_binary_path()
    if not rocm_path:
        return None
    try:
        result = subprocess.run(
            [str(rocm_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(rocm_path.parent),  # Run from the onedir directory
        )
        # Output format: "voicebox-server 0.3.0"
        for line in result.stdout.strip().splitlines():
            if "voicebox-server" in line:
                return line.split()[-1]
    except Exception as e:
        logger.warning(f"Could not get ROCm binary version: {e}")
    return None


async def check_and_update_rocm_binary():
    """Check if the ROCm binary is outdated and auto-download if so.

    Called on server startup. Checks both server version and ROCm libs
    version. Downloads only what's needed.
    """
    rocm_path = get_rocm_binary_path()
    if not rocm_path:
        return  # No ROCm binary installed, nothing to update

    if is_rocm_active():
        logger.info("ROCm backend is active; skipping auto-update to avoid replacing the running backend")
        return

    need_server = _needs_server_download()
    need_libs = _needs_rocm_libs_download()

    if not need_server and not need_libs:
        logger.info(f"ROCm binary is up to date (server=v{__version__}, libs={get_installed_rocm_libs_version()})")
        return

    reasons = []
    if need_server:
        rocm_version = get_rocm_binary_version()
        reasons.append(f"server v{rocm_version} != v{__version__}")
    if need_libs:
        installed_libs = get_installed_rocm_libs_version()
        reasons.append(f"libs {installed_libs} != {ROCM_LIBS_VERSION}")

    logger.info(f"ROCm backend needs update ({', '.join(reasons)}). Auto-downloading...")

    try:
        await download_rocm_binary()
    except Exception as e:
        logger.error(f"Auto-update of ROCm binary failed: {e}")


async def delete_rocm_binary() -> bool:
    """Delete the downloaded ROCm backend directory. Returns True if deleted."""
    import shutil

    rocm_dir = get_rocm_dir()
    if rocm_dir.exists() and any(rocm_dir.iterdir()):
        shutil.rmtree(rocm_dir)
        logger.info(f"Deleted ROCm backend directory: {rocm_dir}")
        return True
    return False
