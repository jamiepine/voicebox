"""Ensure TADA voice-prompt encoding disables autograd (#890)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import numpy as np
import pytest
import soundfile as sf
import torch

from backend.backends.hume_backend import HumeTadaBackend


@dataclass
class _FakeEncoderOutput:
    emb: torch.Tensor


class _GradTrackingEncoder:
    """Raises unless called under torch.inference_mode()."""

    def __init__(self) -> None:
        self.called_under_inference_mode = False

    def __call__(self, audio, text=None, sample_rate=None):
        self.called_under_inference_mode = torch.is_inference_mode_enabled()
        if not self.called_under_inference_mode:
            raise AssertionError("encoder forward must run under inference_mode")
        # Touch a requires_grad tensor the way Snake1d alpha would.
        alpha = torch.nn.Parameter(torch.ones(1, device=audio.device))
        _ = audio.mean() * alpha
        return _FakeEncoderOutput(emb=torch.zeros(1, 4, device=audio.device))


@pytest.mark.asyncio
async def test_create_voice_prompt_runs_encoder_under_inference_mode(tmp_path, monkeypatch):
    wav = tmp_path / "ref.wav"
    sf.write(str(wav), np.zeros(24000, dtype=np.float32), 24000)

    backend = HumeTadaBackend()
    backend.model = object()  # mark loaded
    backend.model_size = "1B"
    backend._device = "cpu"
    encoder = _GradTrackingEncoder()
    backend.encoder = encoder

    monkeypatch.setattr(backend, "load_model", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "backend.backends.hume_backend.get_cached_voice_prompt",
        lambda key: None,
    )
    monkeypatch.setattr(
        "backend.backends.hume_backend.cache_voice_prompt",
        lambda key, value: None,
    )

    prompt, from_cache = await backend.create_voice_prompt(
        str(wav),
        reference_text="hello world",
        use_cache=False,
    )

    assert from_cache is False
    assert encoder.called_under_inference_mode is True
    assert isinstance(prompt["emb"], torch.Tensor)
    assert prompt["emb"].device.type == "cpu"
