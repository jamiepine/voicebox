"""
LLM inference module - delegates to backend abstraction layer.
"""

from ..backends import get_llm_backend, LLMBackend


def get_llm_model() -> LLMBackend:
    """Get LLM backend instance (MLX or PyTorch based on platform)."""
    return get_llm_backend()


def unload_llm_model() -> None:
    """Unload LLM model to free memory."""
    get_llm_backend().unload_model()
