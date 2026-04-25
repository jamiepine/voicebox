"""
Platform detection for backend selection.
"""

import platform
import subprocess
from typing import Literal


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (arm64 macOS).

    Returns:
        True if on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_amd_gpu_windows() -> bool:
    """
    Check if the primary GPU on Windows is an AMD Radeon card.

    Uses WMI to query Win32_VideoController, with a fallback to
    torch.cuda.get_device_name(0) if WMI is unavailable.  This is
    useful for deciding whether the ROCm backend is appropriate.

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
