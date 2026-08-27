"""
Regression tests: unloading an MLX-backed model must release MLX's internal
buffer cache, not just drop the Python reference.

MLX keeps freed array buffers in an internal pool for reuse instead of
returning them to the OS (see `mlx.core.clear_cache` / `get_cache_memory`).
Before this fix, `unload_model()` on the MLX TTS/Whisper/LLM backends only
did `del self.model; self.model = None`, so the process's memory footprint
never shrank after "unload" (issue: resources stay held after first use on
Apple Silicon). These tests assert each backend's `unload_model()` calls the
shared `empty_mlx_cache()` helper, without requiring the real `mlx` package
to be installed — `empty_mlx_cache` is monkeypatched, so its own lazy
`import mlx.core` never executes here.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")

from backend.backends import mlx_backend, qwen_llm_backend


def test_mlx_tts_backend_unload_clears_mlx_cache(monkeypatch):
    """Unloading the MLX TTS backend must call empty_mlx_cache()."""
    mock_clear = MagicMock()
    monkeypatch.setattr(mlx_backend, "empty_mlx_cache", mock_clear)

    backend = mlx_backend.MLXTTSBackend()
    backend.model = MagicMock()
    backend._current_model_size = "1.7B"

    backend.unload_model()

    assert backend.model is None
    assert backend._current_model_size is None
    mock_clear.assert_called_once()


def test_mlx_stt_backend_unload_clears_mlx_cache(monkeypatch):
    """Unloading the MLX Whisper backend must call empty_mlx_cache()."""
    mock_clear = MagicMock()
    monkeypatch.setattr(mlx_backend, "empty_mlx_cache", mock_clear)

    backend = mlx_backend.MLXSTTBackend()
    backend.model = MagicMock()

    backend.unload_model()

    assert backend.model is None
    mock_clear.assert_called_once()


def test_mlx_llm_backend_unload_clears_mlx_cache(monkeypatch):
    """Unloading the MLX Qwen3 LLM backend must call empty_mlx_cache()."""
    mock_clear = MagicMock()
    monkeypatch.setattr(qwen_llm_backend, "empty_mlx_cache", mock_clear)

    backend = qwen_llm_backend.MLXQwenLLMBackend()
    backend.model = MagicMock()
    backend.tokenizer = MagicMock()
    backend._current_model_size = "4B"

    backend.unload_model()

    assert backend.model is None
    assert backend.tokenizer is None
    mock_clear.assert_called_once()


def test_mlx_backends_do_not_clear_cache_when_already_unloaded(monkeypatch):
    """Calling unload on an already-unloaded backend is a no-op (no spurious clear)."""
    mock_clear = MagicMock()
    monkeypatch.setattr(mlx_backend, "empty_mlx_cache", mock_clear)

    backend = mlx_backend.MLXTTSBackend()
    assert backend.model is None

    backend.unload_model()

    mock_clear.assert_not_called()
