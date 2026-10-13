"""LuxTTS turns the vocoder's short-input crash into a readable error (#1046)."""

from unittest.mock import AsyncMock

import pytest

from backend.backends.luxtts_backend import LuxTTSBackend


class _FakeModel:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate_speech(self, **kwargs):
        raise self.error


def _backend(error: Exception) -> LuxTTSBackend:
    backend = LuxTTSBackend()
    backend.model = _FakeModel(error)
    backend._device = "cpu"
    backend.load_model = AsyncMock(return_value=None)
    return backend


async def test_short_text_kernel_error_becomes_value_error():
    err = RuntimeError(
        "Calculated padded input size per channel: (6). Kernel size: (7). "
        "Kernel size can't be greater than actual input size"
    )
    with pytest.raises(ValueError, match="too short"):
        await _backend(err).generate("This is a test", voice_prompt={})


async def test_other_runtime_errors_are_not_masked():
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        await _backend(RuntimeError("CUDA out of memory")).generate("hello", voice_prompt={})
