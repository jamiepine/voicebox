"""Unit tests for the native VoxCPM2 backend without loading model weights."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.backends.voxcpm_backend import VoxCPMBackend


@pytest.mark.asyncio
async def test_create_voice_prompt_preserves_reference_audio_and_text(tmp_path):
    backend = VoxCPMBackend()
    audio_path = tmp_path / "reference.wav"

    prompt, cached = await backend.create_voice_prompt(str(audio_path), "  hello  ")

    assert prompt == {"ref_audio": str(audio_path), "ref_text": "hello"}
    assert cached is False


@pytest.mark.asyncio
async def test_generate_forwards_ultimate_clone_prompt_and_returns_48khz_audio(tmp_path):
    backend = VoxCPMBackend()
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"reference")
    backend._device = "cpu"
    captured = {}

    class FakeModel:
        tts_model = SimpleNamespace(sample_rate=48000)

        def _generate(self, text, seed=None, **kwargs):
            pass

        def generate(self, **kwargs):
            captured.update(kwargs)
            return np.zeros(32, dtype=np.float32)

    backend.model = FakeModel()

    audio, sample_rate = await backend.generate(
        "hello",
        {"ref_audio": str(audio_path), "ref_text": "reference text"},
        seed=7,
        instruct="warm and clear",
    )

    assert audio.dtype == np.float32
    assert audio.shape == (32,)
    assert sample_rate == 48000
    assert captured == {
        "text": "(warm and clear)hello",
        "seed": 7,
        "retry_badcase": True,
        "reference_wav_path": str(audio_path),
    }


@pytest.mark.asyncio
async def test_generate_uses_transcript_guided_cloning_without_instruction(tmp_path):
    backend = VoxCPMBackend()
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"reference")
    captured = {}

    class LegacyModel:
        tts_model = SimpleNamespace(sample_rate=48000)

        def _generate(self, text, **kwargs):
            pass

        def generate(self, **kwargs):
            captured.update(kwargs)
            return np.ones(16, dtype=np.float32)

    backend.model = LegacyModel()
    backend._device = "cpu"

    await backend.generate("hello", {"ref_audio": str(audio_path), "ref_text": "reference text"}, seed=9)

    assert captured == {
        "text": "hello",
        "retry_badcase": True,
        "reference_wav_path": str(audio_path),
        "prompt_wav_path": str(audio_path),
        "prompt_text": "reference text",
    }


def test_model_registry_contains_voxcpm2():
    from backend.backends import get_model_config, get_tts_backend_for_engine

    config = get_model_config("voxcpm2")
    assert config is not None
    assert config.engine == "voxcpm2"
    assert config.hf_repo_id == "openbmb/VoxCPM2"
    assert isinstance(get_tts_backend_for_engine("voxcpm2"), VoxCPMBackend)


def test_voxcpm2_request_accepts_languages_added_by_the_engine():
    from backend.models import GenerationRequest, VoiceProfileCreate

    request = GenerationRequest(profile_id="profile", text="สวัสดี", language="th", engine="voxcpm2")
    profile = VoiceProfileCreate(name="Thai voice", language="th")

    assert request.language == "th"
    assert profile.language == "th"
