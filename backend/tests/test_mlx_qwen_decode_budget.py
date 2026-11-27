"""
Regression tests: MLX Qwen3-TTS generation must pass a length-scaled
``max_tokens`` to mlx-audio.

Without it a missed codec EOS keeps decoding until mlx-audio's own limit
(4096 tokens per segment on the pinned 0.4.1 ≈ 5.5 min of audio, or the
checkpoint's 8192 on newer releases) before the runaway detector can even
look at the output. The model is a fake with mlx-audio's ``generate()``
signature, so the real ``mlx`` package is not needed.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from backend.backends import mlx_backend
from backend.backends.base import QWEN_MIN_NEW_TOKENS, estimate_max_new_tokens

SAMPLE_RATE = 24000


class FakeQwenModel:
    """Mimics mlx-audio 0.4.1 ``Qwen3TTSModel.generate`` (max_tokens=4096)."""

    def __init__(self, token_count: int | None = None):
        self.calls = []
        self.token_count = token_count

    def generate(
        self,
        text,
        voice=None,
        lang_code="auto",
        ref_audio=None,
        ref_text=None,
        max_tokens: int = 4096,
    ):
        self.calls.append({"text": text, "ref_audio": ref_audio, "max_tokens": max_tokens})
        yield SimpleNamespace(
            audio=np.zeros(SAMPLE_RATE, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            token_count=self.token_count if self.token_count is not None else 10,
        )


def _params(func):
    import inspect

    return inspect.signature(func).parameters


def test_budget_scales_with_text_and_keeps_floor():
    def generate(text, max_tokens: int = 8192):
        pass

    assert mlx_backend._decode_budget("hi", _params(generate)) == QWEN_MIN_NEW_TOKENS
    long_text = "a" * 500
    assert mlx_backend._decode_budget(long_text, _params(generate)) == estimate_max_new_tokens(long_text)


def test_budget_never_raises_a_smaller_library_default():
    def generate(text, max_tokens: int = 4096):
        pass

    # 800 chars would allow 8192 tokens; mlx-audio 0.4.1 already stops at 4096.
    assert mlx_backend._decode_budget("a" * 800, _params(generate)) == 4096


def test_budget_uses_estimate_when_library_default_is_unset():
    def generate(text, max_tokens=None):
        pass

    assert mlx_backend._decode_budget("hello", _params(generate)) == QWEN_MIN_NEW_TOKENS


def test_budget_skipped_when_generate_takes_no_max_tokens():
    def generate(text, lang_code="auto"):
        pass

    assert mlx_backend._decode_budget("hello", _params(generate)) is None


def _backend_with(model):
    backend = mlx_backend.MLXTTSBackend()
    backend.model = model
    backend._current_model_size = backend.model_size
    return backend


async def test_voice_clone_generation_passes_decode_budget(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    model = FakeQwenModel()
    backend = _backend_with(model)

    audio, sample_rate = await backend.generate(
        "Short sentence.", {"ref_audio": str(ref), "ref_text": "Reference text."}
    )

    assert sample_rate == SAMPLE_RATE
    assert len(audio) == SAMPLE_RATE
    assert len(model.calls) == 1
    assert model.calls[0]["ref_audio"] == str(ref)
    assert model.calls[0]["max_tokens"] == QWEN_MIN_NEW_TOKENS


async def test_hitting_the_budget_is_logged(caplog):
    model = FakeQwenModel(token_count=QWEN_MIN_NEW_TOKENS)
    backend = _backend_with(model)

    with caplog.at_level(logging.WARNING, logger=mlx_backend.logger.name):
        await backend.generate("Short sentence.", {})

    assert model.calls[0]["max_tokens"] == QWEN_MIN_NEW_TOKENS
    assert "missed EOS" in caplog.text
