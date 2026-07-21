from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from backend import backends, models
from backend.backends import pytorch_backend
from backend.backends.mlx_backend import MLXSTTBackend
from backend.backends.pytorch_backend import PyTorchSTTBackend


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeProcessor:
    def __call__(self, *_args, **_kwargs):
        return _FakeBatch(input_features=torch.zeros((1, 80, 10)))

    def get_decoder_prompt_ids(self, *, language, task):
        return [(1, language)]

    def batch_decode(self, *_args, **_kwargs):
        return ["  bonjour le monde  "]


def test_transcription_result_contract_exists():
    assert hasattr(backends, "TranscriptionResult")
    assert get_type_hints(backends.STTBackend.transcribe)["return"] is str
    assert get_type_hints(backends.STTBackend.transcribe_with_metadata)["return"] is backends.TranscriptionResult


@pytest.mark.asyncio
async def test_metadata_adapter_preserves_legacy_text_only_backends():
    class LegacyBackend:
        async def transcribe(self, audio_path, language=None, model_size=None):
            assert audio_path == "sample.wav"
            assert model_size == "small"
            return "  hola mundo  "

    result = await backends.transcribe_with_metadata(LegacyBackend(), "sample.wav", language="es", model_size="small")

    assert result == backends.TranscriptionResult(text="hola mundo", language="es")


def test_transcription_response_exposes_detected_language():
    response = models.TranscriptionResponse(
        text="bonjour",
        duration=1.0,
        language="fr",
    )

    assert response.language == "fr"


def test_pytorch_whisper_language_token_maps_to_code():
    generation_config = SimpleNamespace(
        lang_to_id={"<|en|>": 100, "<|zh|>": 200},
    )

    assert pytorch_backend.whisper_language_code_from_token_id(generation_config, 200) == "zh"


@pytest.mark.asyncio
async def test_pytorch_transcribe_returns_auto_detected_language(monkeypatch):
    processor = _FakeProcessor()
    detect_language = MagicMock(return_value=torch.tensor([200]))
    generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    model = SimpleNamespace(
        generation_config=SimpleNamespace(lang_to_id={"<|en|>": 100, "<|fr|>": 200}),
        detect_language=detect_language,
        generate=generate,
    )
    backend = object.__new__(PyTorchSTTBackend)
    backend.model = model
    backend.processor = processor
    backend.model_size = "base"
    backend.device = "cpu"
    backend.load_model_async = AsyncMock()
    monkeypatch.setattr(pytorch_backend, "load_audio", lambda *_args, **_kwargs: ([0.0], 16000))

    result = await backend.transcribe_with_metadata("sample.wav")

    assert result == backends.TranscriptionResult(text="bonjour le monde", language="fr")
    assert "forced_decoder_ids" not in generate.call_args.kwargs
    assert await backend.transcribe("sample.wav") == "bonjour le monde"


@pytest.mark.asyncio
async def test_pytorch_transcribe_forces_only_explicit_language(monkeypatch):
    processor = _FakeProcessor()
    detect_language = MagicMock()
    generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
    backend = object.__new__(PyTorchSTTBackend)
    backend.model = SimpleNamespace(
        generation_config=SimpleNamespace(lang_to_id={"<|en|>": 100}),
        detect_language=detect_language,
        generate=generate,
    )
    backend.processor = processor
    backend.model_size = "base"
    backend.device = "cpu"
    backend.load_model_async = AsyncMock()
    monkeypatch.setattr(
        pytorch_backend, "load_audio", lambda *_args, **_kwargs: ([0.0], 16000)
    )

    result = await backend.transcribe_with_metadata("sample.wav", language="en")

    assert result.language == "en"
    detect_language.assert_not_called()
    assert generate.call_args.kwargs["forced_decoder_ids"] == [(1, "en")]


@pytest.mark.asyncio
async def test_mlx_transcribe_returns_detected_language():
    backend = MLXSTTBackend()
    backend.model = SimpleNamespace(
        generate=lambda *_args, **_kwargs: SimpleNamespace(text="  你好世界  ", language="zh")
    )
    backend.load_model_async = AsyncMock()

    result = await backend.transcribe_with_metadata("sample.wav")

    assert result == backends.TranscriptionResult(text="你好世界", language="zh")
    assert await backend.transcribe("sample.wav") == "你好世界"
