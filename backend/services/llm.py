"""
LLM inference module - delegates to backend abstraction layer.
"""

from ..backends import LLMBackend, get_llm_backend, unload_backend


def get_llm_model() -> LLMBackend:
    """Get LLM backend instance (MLX or PyTorch based on platform)."""
    return get_llm_backend()


async def unload_llm_model() -> None:
    """Unload LLM model to free memory, serialized onto the MLX worker."""
    await unload_backend(get_llm_backend())
