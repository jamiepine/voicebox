"""Unit tests for long-audio chunking in the PyTorch STT backend.

Regression: Whisper's encoder only accepts 30s of audio per pass. Before
chunking, a 6:43 upload silently transcribed just the first 30 seconds.
Also guards the ``return_timestamps=True`` flag, which prevents Whisper
from emitting an early <|endoftext|> after the first sentence of a
window (observed: multi-sentence chunks truncated to one sentence).

These tests mock the HF processor/model, so they run without GPU or
model downloads.
"""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.backends.pytorch_backend import PyTorchSTTBackend


class _FakeInputs(dict):
    """Mimics the transformers BatchFeature mapping (has .to())."""

    def to(self, device=None, dtype=None):
        return self


def _make_backend(chunk_outputs):
    """Build a PyTorchSTTBackend with mocked processor/model."""
    backend = PyTorchSTTBackend.__new__(PyTorchSTTBackend)
    backend.device = "cpu"
    backend.model_size = "small"

    outputs = list(chunk_outputs)

    processor = MagicMock()
    processor.side_effect = lambda audio, sampling_rate, return_tensors: _FakeInputs(
        input_features=np.zeros((1, 80, 3000), dtype=np.float32)
    )
    processor.batch_decode.side_effect = lambda ids, skip_special_tokens: [
        outputs.pop(0)
    ]
    processor.get_decoder_prompt_ids.return_value = [(1, 50257)]
    backend.processor = processor

    model = MagicMock()
    model.dtype = "float32"
    model.generate.side_effect = lambda feats, **kw: np.array([[0]])
    backend.model = model
    return backend


def _transcribe(backend, seconds: float, language="es"):
    pcm = np.zeros(int(seconds * 16000), dtype=np.float32)
    with (
        patch(
            "backend.backends.pytorch_backend.load_audio",
            return_value=(pcm, 16000),
        ),
        patch.object(
            PyTorchSTTBackend, "load_model_async", return_value=None
        ),
        patch.object(PyTorchSTTBackend, "is_loaded", return_value=True),
    ):
        return asyncio.run(backend.transcribe("dummy.wav", language=language))


def test_short_audio_single_chunk():
    backend = _make_backend(["hola mundo"])
    text = _transcribe(backend, seconds=10)
    assert text == "hola mundo"
    assert backend.model.generate.call_count == 1


def test_long_audio_is_chunked_and_joined():
    backend = _make_backend(["primera parte", "segunda parte", "tercera parte"])
    text = _transcribe(backend, seconds=70)  # 3 chunks of 30s
    assert backend.model.generate.call_count == 3
    assert text == "primera parte segunda parte tercera parte"


def test_timestamps_forced_to_prevent_early_eot():
    backend = _make_backend(["a", "b"])
    _transcribe(backend, seconds=45)
    for call in backend.model.generate.call_args_list:
        assert call.kwargs.get("return_timestamps") is True


def test_language_forcing_still_passed():
    backend = _make_backend(["texto"])
    _transcribe(backend, seconds=5, language="es")
    backend.processor.get_decoder_prompt_ids.assert_called_once_with(
        language="es", task="transcribe"
    )
    assert "forced_decoder_ids" in backend.model.generate.call_args.kwargs
