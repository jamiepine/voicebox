"""
Platform detection for backend selection.
"""

import platform
from typing import Literal


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (arm64 macOS).
    
    Returns:
        True if on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


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
