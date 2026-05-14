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


def get_supported_platforms() -> list[str]:
    """Return which compute platforms the current machine supports.

    Possible values: "cuda", "mps", "xpu", "rocm", "cpu"

    Rules:
    - "cpu" is always included (every machine can run CPU inference).
    - "cuda" is added when PyTorch reports a CUDA device available.
    - "rocm" is added on ROCm builds (torch.version.hip is set).
    - "mps" is added when the Metal Performance Shaders backend is available.
    - "xpu" is added when Intel Extension for PyTorch detects an Arc/XPU device.

    Apple Silicon machines therefore return ["mps", "cpu"], a typical
    CUDA Linux machine returns ["cuda", "cpu"], an Intel Arc machine returns
    ["xpu", "cpu"], and a CPU-only machine returns ["cpu"].
    """
    supported: list[str] = []

    try:
        import torch

        if torch.cuda.is_available():
            # Distinguish ROCm from CUDA — both report via cuda.is_available()
            # on the ROCm PyTorch build, but torch.version.hip is non-None.
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            if is_rocm:
                supported.append("rocm")
            else:
                supported.append("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            supported.append("mps")

        try:
            import intel_extension_for_pytorch  # noqa: F401

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                supported.append("xpu")
        except ImportError:
            pass

    except ImportError:
        pass  # torch not available at all — only CPU

    supported.append("cpu")
    return supported
