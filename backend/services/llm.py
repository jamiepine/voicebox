"""
LLM inference module - delegates to backend abstraction layer.
"""

import threading

from ..backends import (
    LLM_ENGINES,
    LLMBackend,
    get_llm_backend_for_engine,
    get_llm_model_configs,
    get_model_config,
)

# Guards the "unload every other engine, then hand back the target one"
# sequence in get_llm_model so two concurrent callers switching engines can't
# interleave into a state where the engine being evicted and the engine being
# activated are both mid-transition at once.
_engine_switch_lock = threading.Lock()


def get_llm_model(engine: str = "qwen_llm") -> LLMBackend:
    """Get the LLM backend for `engine`, unloading any other loaded LLM engine first.

    Only one LLM engine is meant to hold memory at a time app-wide. Switching
    engines used to be safe "by accident" when qwen_llm was the only engine —
    its own load_model() unloads itself before loading a different size. That
    doesn't extend across two different engines' separate singletons, so this
    unloads every other registered engine explicitly before returning the one
    the caller asked for.

    This has a real side effect (it can evict another engine's loaded
    weights) and must only be called by code that is actually about to
    activate `engine`. Read-only status checks belong on
    `get_llm_backend_for_engine` instead, which never evicts anything.
    """
    with _engine_switch_lock:
        for other_engine in LLM_ENGINES:
            if other_engine != engine:
                other = get_llm_backend_for_engine(other_engine)
                if other.is_loaded():
                    other.unload_model()

        return get_llm_backend_for_engine(engine)


def unload_llm_model(engine: str = "qwen_llm") -> None:
    """Unload the LLM model for `engine` to free memory."""
    get_llm_backend_for_engine(engine).unload_model()


def resolve_backend_and_size(model_name: str | None) -> tuple[LLMBackend, str, str]:
    """Resolve a model identifier to (backend, bare size, resolved model_name).

    `model_name` (e.g. "minicpm5-1b") names a specific engine+size — resolve
    it via the model registry instead of always assuming qwen_llm. `None`
    keeps the default (whichever size is already active on the default
    qwen_llm backend) and reverse-resolves it to its own model_name, since
    callers persist this as capture attribution and a bare size like "1B"
    no longer uniquely identifies a model once more than one engine exists.
    """
    if model_name is None:
        backend = get_llm_model()
        resolved_size = backend.model_size
        config = next(
            c for c in get_llm_model_configs()
            if c.engine == "qwen_llm" and c.model_size == resolved_size
        )
        return backend, resolved_size, config.model_name

    config = get_model_config(model_name)
    backend = get_llm_model(config.engine)
    return backend, config.model_size, config.model_name
