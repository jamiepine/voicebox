"""
Provider download and installation manager.
"""

import asyncio
import httpx
import platform
from pathlib import Path
from typing import Optional

from .types import ProviderType
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager


# Provider version (independent of app version)
PROVIDER_VERSION = "1.0.0"

# Base URL for provider downloads (Cloudflare R2)
PROVIDER_DOWNLOAD_BASE_URL = "https://downloads.voicebox.sh/providers"


def _get_providers_dir() -> Path:
    """Get the directory where providers are stored."""
    system = platform.system()
    
    if system == "Windows":
        appdata = Path.home() / "AppData" / "Roaming"
    elif system == "Darwin":
        appdata = Path.home() / "Library" / "Application Support"
    else:  # Linux
        appdata = Path.home() / ".local" / "share"
    
    providers_dir = appdata / "voicebox" / "providers"
    providers_dir.mkdir(parents=True, exist_ok=True)
    return providers_dir


def _get_provider_binary_name(provider_type: str) -> str:
    """Get the local binary filename for a provider type."""
    system = platform.system()
    ext = ".exe" if system == "Windows" else ""
    
    binary_map = {
        "pytorch-cpu": f"tts-provider-pytorch-cpu{ext}",
        "pytorch-cuda": f"tts-provider-pytorch-cuda{ext}",
    }
    
    if provider_type not in binary_map:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return binary_map[provider_type]


def _get_provider_download_name(provider_type: str) -> str:
    """Get the remote download filename for a provider type (includes platform suffix)."""
    system = platform.system()

    if system == "Windows":
        platform_suffix = "windows"
        ext = ".zip"
    elif system == "Linux":
        platform_suffix = "linux"
        ext = ".tar.gz"
    elif system == "Darwin":
        # Detect macOS architecture
        machine = platform.machine()
        if machine == "arm64":
            platform_suffix = "macos-arm64"
        else:
            platform_suffix = "macos-x64"
        ext = ".tar.gz"
    else:
        raise ValueError(f"Provider downloads not supported on {system}")

    return f"tts-provider-{provider_type}-{platform_suffix}{ext}"


def _get_provider_download_url(provider_type: str) -> str:
    """Get the download URL for a provider."""
    download_name = _get_provider_download_name(provider_type)
    return f"{PROVIDER_DOWNLOAD_BASE_URL}/v{PROVIDER_VERSION}/{download_name}"


async def download_provider(provider_type: str) -> Path:
    """
    Download and extract a provider archive from Cloudflare R2.

    Args:
        provider_type: Type of provider to download (e.g., "pytorch-cpu")

    Returns:
        Path to the extracted provider binary

    Raises:
        ValueError: If provider_type is invalid
        httpx.HTTPError: If download fails
    """
    if provider_type not in ["pytorch-cpu", "pytorch-cuda"]:
        raise ValueError(f"Provider type {provider_type} cannot be downloaded")

    progress_manager = get_progress_manager()
    task_manager = get_task_manager()

    archive_name = _get_provider_download_name(provider_type)
    download_url = _get_provider_download_url(provider_type)
    providers_dir = _get_providers_dir()
    archive_path = providers_dir / archive_name

    # Start tracking download
    task_manager.start_download(provider_type)

    # Initialize progress state
    progress_manager.update_progress(
        model_name=provider_type,
        current=0,
        total=0,  # Will be updated once we get Content-Length
        filename=archive_name,
        status="downloading",
    )

    try:
        # Download archive
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", download_url) as response:
                response.raise_for_status()

                # Get total size from Content-Length header
                total_size = int(response.headers.get("Content-Length", 0))

                if total_size > 0:
                    progress_manager.update_progress(
                        model_name=provider_type,
                        current=0,
                        total=total_size,
                        filename=archive_name,
                        status="downloading",
                    )

                # Download with progress tracking
                downloaded = 0
                with open(archive_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Update progress
                        progress_manager.update_progress(
                            model_name=provider_type,
                            current=downloaded,
                            total=total_size if total_size > 0 else downloaded,
                            filename=archive_name,
                            status="downloading",
                        )

        # Extract archive
        progress_manager.update_progress(
            model_name=provider_type,
            current=downloaded,
            total=downloaded,
            filename="Extracting...",
            status="downloading",
        )

        import zipfile
        import tarfile

        if archive_name.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(providers_dir)
        elif archive_name.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(providers_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_name}")

        # Remove archive after extraction
        archive_path.unlink()

        # Get path to extracted binary
        binary_path = get_provider_binary_path(provider_type)
        if not binary_path:
            raise ValueError(f"Provider binary not found after extraction")

        # Make executable on Unix systems
        if platform.system() != "Windows":
            binary_path.chmod(0o755)

        # Mark as complete
        progress_manager.update_progress(
            model_name=provider_type,
            current=downloaded,
            total=downloaded,
            filename=_get_provider_binary_name(provider_type),
            status="complete",
        )
        task_manager.complete_download(provider_type)

        return binary_path

    except Exception as e:
        # Clean up archive if it exists
        if archive_path.exists():
            archive_path.unlink()

        # Mark as error
        progress_manager.update_progress(
            model_name=provider_type,
            current=0,
            total=0,
            filename=archive_name,
            status="error",
        )
        task_manager.error_download(provider_type, str(e))
        raise


def get_provider_binary_path(provider_type: str) -> Optional[Path]:
    """
    Get the path to an installed provider binary.

    Args:
        provider_type: Type of provider

    Returns:
        Path to provider binary, or None if not installed
    """
    providers_dir = _get_providers_dir()
    binary_name = _get_provider_binary_name(provider_type)

    # Check for --onedir structure (directory with binary inside)
    provider_dir = providers_dir / f"tts-provider-{provider_type}"
    if provider_dir.exists() and provider_dir.is_dir():
        binary_path = provider_dir / binary_name
        if binary_path.exists() and binary_path.is_file():
            return binary_path

    # Fallback: check for direct binary (legacy)
    provider_path = providers_dir / binary_name
    if provider_path.exists() and provider_path.is_file():
        return provider_path

    return None


def delete_provider(provider_type: str) -> bool:
    """
    Delete an installed provider binary.
    
    Args:
        provider_type: Type of provider to delete
        
    Returns:
        True if deleted, False if not found
    """
    provider_path = get_provider_binary_path(provider_type)
    
    if provider_path and provider_path.exists():
        provider_path.unlink()
        return True
    
    return False
