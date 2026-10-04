"""
LLM inference module - delegates to backend abstraction layer.
"""

from ..backends import LLMBackend, get_llm_backend


def get_llm_model() -> LLMBackend:
    """Get LLM backend instance (MLX or PyTorch based on platform)."""
    return get_llm_backend()


def resolve_model_size(backend: LLMBackend, requested: str | None) -> str:
    """Model label to record for a call on ``backend``.

    The remote OpenAI-compatible backend ignores Qwen size requests and
    reports its own model name, so the label persisted on captures and
    returned to clients must come from the backend, not the request.
    """
    from ..backends.openai_compat_backend import OpenAICompatLLMBackend

    if isinstance(backend, OpenAICompatLLMBackend):
        return backend.model_size
    return requested or backend.model_size


def unload_llm_model() -> None:
    """Unload LLM model to free memory."""
    get_llm_backend().unload_model()
