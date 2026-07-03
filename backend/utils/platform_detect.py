"""
Platform detection for backend selection.
"""

import os
import platform
import subprocess
from functools import lru_cache
from typing import Literal


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (arm64 macOS).

    Returns:
        True if on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def is_amd_gpu_windows() -> bool:
    """
    Check if the primary GPU on Windows is an AMD Radeon card.

    Uses WMI to query Win32_VideoController, with a fallback to
    torch.cuda.get_device_name(0) if WMI is unavailable.  This is
    useful for deciding whether the ROCm backend is appropriate.

    Result is cached since it shells out to PowerShell and the GPU
    does not change at runtime — safe to call from the health path.

    Returns:
        True if an AMD GPU is detected on Windows, False otherwise.
    """
    if platform.system() != "Windows":
        return False

    # Primary method: WMI query for AMD adapters
    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Where-Object {$_.AdapterCompatibility -like '*AMD*'} | "
                "Measure-Object | Select-Object -ExpandProperty Count",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if int(result.stdout.strip()) > 0:
            return True
    except Exception:
        pass

    # Fallback: torch.cuda.get_device_name(0) (works for ROCm/HIP too)
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if "Radeon" in name or "AMD" in name:
                return True
    except Exception:
        pass

    return False


# Platform tokens for which we actually build and publish downloadable GPU
# server assets. Anything else must not be offered a download (it would 404).
SUPPORTED_GPU_ASSET_PLATFORMS = ("linux-x86_64", "windows-x86_64")


@lru_cache(maxsize=1)
def is_amd_rocm_capable() -> bool:
    """Whether this host can run the downloadable ROCm backend.

    Decides whether to offer the ROCm backend download in the UI, so it must
    work before ROCm torch is installed (can't rely on torch.version.hip).

    Windows: reuse the WMI/torch AMD probe. Linux: /dev/kfd is created by the
    amdgpu kernel driver whenever ROCm compute is available — the same gate the
    Docker entrypoint and setup recipe use. Gated on a platform we publish
    assets for so unsupported arches (e.g. linux-arm64) aren't offered a 404.
    """
    if server_asset_platform() not in SUPPORTED_GPU_ASSET_PLATFORMS:
        return False
    system = platform.system()
    if system == "Windows":
        return is_amd_gpu_windows()
    if system == "Linux":
        return os.path.exists("/dev/kfd")
    return False


def server_asset_platform() -> str:
    """Platform token identifying which downloadable GPU server assets to fetch.

    Windows and Linux ROCm/CUDA builds are distinct binaries published as
    separate release assets, so the download URL must be qualified by platform.
    Only the platforms that ship a downloadable GPU backend are meaningful here
    (macOS uses MLX/Metal and has no CUDA/ROCm server to download).
    """
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    return f"{platform.system().lower()}-{arch}"


def get_backend_type() -> Literal["mlx", "pytorch"]:
    """
    Detect the best backend for the current platform.

    Returns:
        "mlx" on Apple Silicon (if MLX is available and functional), "pytorch" otherwise
    """
    if is_apple_silicon():
        try:
            import mlx.core  # noqa: F401 — triggers native lib loading
            return "mlx"
        except (ImportError, OSError, RuntimeError):
            # MLX not installed, or native libraries failed to load inside a
            # PyInstaller bundle (OSError on missing .dylib / .metallib).
            # Fall through to PyTorch.
            return "pytorch"
    return "pytorch"
