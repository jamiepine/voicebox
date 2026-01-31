"""
Shared types for TTS providers.
"""

from typing import Optional, TypedDict
from enum import Enum


class ProviderType(str, Enum):
    """Available provider types."""
    BUNDLED_MLX = "bundled-mlx"
    BUNDLED_PYTORCH = "bundled-pytorch"
    PYTORCH_CPU = "pytorch-cpu"
    PYTORCH_CUDA = "pytorch-cuda"
    REMOTE = "remote"
    OPENAI = "openai"


class ProviderHealth(TypedDict):
    """Provider health status."""
    status: str  # "healthy", "unhealthy", "starting"
    provider: str
    version: Optional[str]
    model: Optional[str]
    device: Optional[str]


class ProviderStatus(TypedDict):
    """Provider model status."""
    model_loaded: bool
    model_size: Optional[str]
    available_sizes: list[str]
    gpu_available: Optional[bool]
    vram_used_mb: Optional[int]
