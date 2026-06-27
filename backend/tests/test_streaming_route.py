"""Unit tests for the /generate/stream route behavior."""

import importlib.util
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("torch") is None:
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.load = lambda *args, **kwargs: None
    torch.save = lambda *args, **kwargs: None
    sys.modules.setdefault("torch", torch)

from backend import backends, models
from backend.routes import generations


class LiveBackend:
    def __init__(self):
        self.generate_called = False
        self.generate_stream_called = False

    async def generate_stream(
        self,
        text,
        voice_prompt,
        language="en",
        seed=None,
        instruct=None,
    ):
        self.generate_stream_called = True
        yield np.array([0.0, 0.5], dtype=np.float32), 24000
        yield np.array([-1.0, 1.0], dtype=np.float32), 24000

    async def generate(self, *args, **kwargs):
        self.generate_called = True
        return np.array([0.25, -0.25], dtype=np.float32), 24000


async def _noop_async(*args, **kwargs):
    return None


async def _read_body(response) -> bytes:
    return b"".join([chunk async for chunk in response.body_iterator])


def _install_route_fakes(monkeypatch, backend, *, effects_chain=None):
    profile = SimpleNamespace(
        id="profile-1",
        effects_chain=effects_chain,
        default_engine="qwen",
        preset_engine=None,
    )

    async def get_profile(profile_id, db):
        return profile

    async def create_voice_prompt_for_profile(*args, **kwargs):
        return {}

    monkeypatch.setattr(generations.profiles, "get_profile", get_profile)
    monkeypatch.setattr(generations.profiles, "create_voice_prompt_for_profile", create_voice_prompt_for_profile)
    monkeypatch.setattr(generations.profiles, "validate_profile_engine", lambda *args, **kwargs: None)
    monkeypatch.setattr(backends, "engine_needs_trim", lambda engine: False)
    monkeypatch.setattr(backends, "ensure_model_cached_or_raise", _noop_async)
    monkeypatch.setattr(backends, "load_engine_model", _noop_async)
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: backend)


@pytest.mark.asyncio
async def test_stream_speech_uses_live_mode_when_request_can_stream(monkeypatch):
    backend = LiveBackend()
    _install_route_fakes(monkeypatch, backend)
    data = models.GenerationRequest(
        profile_id="profile-1",
        text="Hello.",
        engine="qwen",
        normalize=False,
    )

    response = await generations.stream_speech(data, db=object())
    body = await _read_body(response)

    assert response.headers["x-voicebox-stream-mode"] == "live"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert backend.generate_stream_called is True
    assert backend.generate_called is False
    assert body[:4] == b"RIFF"
    assert body[36:40] == b"data"
    assert body[44:] == struct.pack("<hhhh", 0, 16383, -32768, 32767)


@pytest.mark.asyncio
async def test_stream_speech_buffers_when_normalization_requires_whole_file(monkeypatch):
    backend = LiveBackend()
    _install_route_fakes(monkeypatch, backend)
    data = models.GenerationRequest(
        profile_id="profile-1",
        text="Hello.",
        engine="qwen",
        normalize=True,
    )

    response = await generations.stream_speech(data, db=object())
    body = await _read_body(response)

    assert response.headers["x-voicebox-stream-mode"] == "buffered"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert backend.generate_stream_called is False
    assert backend.generate_called is True
    assert body[:4] == b"RIFF"
