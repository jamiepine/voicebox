"""Regression coverage for TADA 3B multilingual prompt alignment (#1067)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np
import pytest
import soundfile as sf
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.backends.hume_backend import HumeTadaBackend
from backend.database import Base, ProfileSample, VoiceProfile
from backend.services.profiles import create_voice_prompt_for_profile


@dataclass
class _FakeEncoderOutput:
    emb: torch.Tensor
    aligner_language: str


class _FakeEncoder:
    def __init__(self, aligner_language: str) -> None:
        self.aligner_language = aligner_language

    def to(self, device: str):
        return self

    def eval(self) -> None:
        return None

    def __call__(self, audio, text=None, sample_rate=None):
        return _FakeEncoderOutput(
            emb=torch.zeros(1, 4, device=audio.device),
            aligner_language=self.aligner_language,
        )


class _CountingEncoder(_FakeEncoder):
    def __init__(self, aligner_language: str) -> None:
        super().__init__(aligner_language)
        self.call_count = 0

    def __call__(self, audio, text=None, sample_rate=None):
        self.call_count += 1
        time.sleep(0.05)
        return super().__call__(audio, text=text, sample_rate=sample_rate)


def _write_reference_wav(tmp_path):
    wav = tmp_path / "reference.wav"
    sf.write(str(wav), np.zeros(2400, dtype=np.float32), 24000)
    return wav


def _loaded_backend(model_size: str) -> HumeTadaBackend:
    backend = HumeTadaBackend()
    backend.model = object()
    backend.model_size = model_size
    backend._device = "cpu"
    backend.encoder = _FakeEncoder("en")
    return backend


def _install_encoder_factory(monkeypatch):
    from backend.utils.dac_shim import install_dac_shim

    install_dac_shim()

    from tada.modules.encoder import Encoder

    requested_languages: list[str] = []

    def fake_from_pretrained(repo_id, *, subfolder, **kwargs):
        language = kwargs.get("language", "en")
        requested_languages.append(language)
        return _FakeEncoder(language)

    monkeypatch.setattr(Encoder, "from_pretrained", fake_from_pretrained)
    return requested_languages


@pytest.mark.asyncio
async def test_tada_3b_switches_between_polish_and_default_english_aligners(tmp_path, monkeypatch):
    wav = _write_reference_wav(tmp_path)
    backend = _loaded_backend("3B")
    requested_languages = _install_encoder_factory(monkeypatch)

    polish_prompt, _ = await backend.create_voice_prompt(
        str(wav),
        "To jest próbka głosu.",
        use_cache=False,
        language="pl",
    )
    english_prompt, _ = await backend.create_voice_prompt(
        str(wav),
        "This is a voice sample.",
        use_cache=False,
        language="en",
    )

    assert polish_prompt["aligner_language"] == "pl"
    assert english_prompt["aligner_language"] == "en"
    assert requested_languages == ["pl", "en"]


@pytest.mark.asyncio
async def test_tada_3b_maps_voicebox_chinese_to_tada_ch_aligner(tmp_path, monkeypatch):
    wav = _write_reference_wav(tmp_path)
    backend = _loaded_backend("3B")
    requested_languages = _install_encoder_factory(monkeypatch)

    prompt, _ = await backend.create_voice_prompt(
        str(wav),
        "这是语音样本。",
        use_cache=False,
        language="zh",
    )

    assert prompt["aligner_language"] == "ch"
    assert requested_languages == ["ch"]


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["ar", "ch", "de", "es", "fr", "it", "ja", "pl", "pt"])
async def test_tada_3b_accepts_documented_aligner_codes(language, tmp_path, monkeypatch):
    wav = _write_reference_wav(tmp_path)
    backend = _loaded_backend("3B")
    requested_languages = _install_encoder_factory(monkeypatch)

    prompt, _ = await backend.create_voice_prompt(
        str(wav),
        "reference transcript",
        use_cache=False,
        language=language,
    )

    assert prompt["aligner_language"] == language
    assert requested_languages == [language]


@pytest.mark.asyncio
async def test_tada_3b_prompt_cache_is_scoped_by_aligner_language(tmp_path, monkeypatch):
    wav = _write_reference_wav(tmp_path)
    backend = _loaded_backend("3B")
    cache_keys: list[str] = []

    def cached_prompt(key):
        cache_keys.append(key)
        return {"cache_key": key}

    monkeypatch.setattr("backend.backends.hume_backend.get_cached_voice_prompt", cached_prompt)

    polish_prompt, polish_cached = await backend.create_voice_prompt(
        str(wav),
        "shared transcript",
        language="pl",
    )
    english_prompt, english_cached = await backend.create_voice_prompt(
        str(wav),
        "shared transcript",
        language="en",
    )

    assert polish_cached is True
    assert english_cached is True
    assert polish_prompt["cache_key"].startswith("tada_v2_pl_")
    assert english_prompt["cache_key"].startswith("tada_v2_en_")
    assert polish_prompt["cache_key"] != english_prompt["cache_key"]


@pytest.mark.asyncio
async def test_concurrent_tada_prompts_encode_once_and_reuse_the_locked_cache(tmp_path, monkeypatch):
    wav = _write_reference_wav(tmp_path)
    backend = _loaded_backend("3B")
    encoder = _CountingEncoder("pl")
    backend.encoder = encoder
    backend._encoder_language = "pl"
    cached_prompts = {}

    monkeypatch.setattr(
        "backend.backends.hume_backend.get_cached_voice_prompt",
        lambda key: cached_prompts.get(key),
    )
    monkeypatch.setattr(
        "backend.backends.hume_backend.cache_voice_prompt",
        lambda key, prompt: cached_prompts.__setitem__(key, prompt),
    )

    results = await asyncio.gather(
        backend.create_voice_prompt(str(wav), "shared transcript", language="pl"),
        backend.create_voice_prompt(str(wav), "shared transcript", language="pl"),
    )

    assert encoder.call_count == 1
    assert sorted(from_cache for _, from_cache in results) == [False, True]
    assert results[0][0]["aligner_language"] == "pl"
    assert results[1][0]["aligner_language"] == "pl"


@pytest.mark.asyncio
async def test_tada_1b_keeps_default_encoder_and_legacy_cache(tmp_path, monkeypatch):
    wav = _write_reference_wav(tmp_path)
    backend = _loaded_backend("1B")
    requested_languages = _install_encoder_factory(monkeypatch)
    cache_keys: list[str] = []

    def no_cached_prompt(key):
        cache_keys.append(key)

    monkeypatch.setattr("backend.backends.hume_backend.get_cached_voice_prompt", no_cached_prompt)
    monkeypatch.setattr("backend.backends.hume_backend.cache_voice_prompt", lambda key, prompt: None)

    prompt, from_cache = await backend.create_voice_prompt(
        str(wav),
        "To jest próbka głosu.",
        language="pl",
    )

    assert from_cache is False
    assert prompt["aligner_language"] == "en"
    assert requested_languages == []
    assert cache_keys[0].startswith("tada_")
    assert not cache_keys[0].startswith("tada_v2_")


class _PromptBackend:
    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
        language: str | None = None,
    ):
        return {"language": language}, False


class _PromptBackendWithoutLanguage:
    async def create_voice_prompt(self, audio_path: str, reference_text: str, use_cache: bool = True):
        return {"language_argument_received": False}, False


@pytest.fixture
def profile_db(tmp_path):
    db_engine = create_engine(f"sqlite:///{tmp_path / 'profiles.db'}")
    Base.metadata.create_all(bind=db_engine)
    session = sessionmaker(bind=db_engine)()
    try:
        yield session
    finally:
        session.close()
        db_engine.dispose()


def _add_polish_profile(profile_db, wav_path) -> str:
    profile = VoiceProfile(name="Polish voice", language="pl", voice_type="cloned")
    profile_db.add(profile)
    profile_db.flush()
    profile_db.add(
        ProfileSample(
            profile_id=profile.id,
            audio_path=str(wav_path),
            reference_text="To jest próbka głosu.",
        )
    )
    profile_db.commit()
    return profile.id


@pytest.mark.asyncio
async def test_profile_language_reaches_tada_prompt_encoding(tmp_path, profile_db, monkeypatch):
    profile_id = _add_polish_profile(profile_db, _write_reference_wav(tmp_path))
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda engine: _PromptBackend())

    prompt = await create_voice_prompt_for_profile(profile_id, profile_db, engine="tada")

    assert prompt == {"language": "pl"}


@pytest.mark.asyncio
async def test_profile_language_is_not_passed_to_other_backends(tmp_path, profile_db, monkeypatch):
    profile_id = _add_polish_profile(profile_db, _write_reference_wav(tmp_path))
    monkeypatch.setattr(
        "backend.backends.get_tts_backend_for_engine",
        lambda engine: _PromptBackendWithoutLanguage(),
    )

    prompt = await create_voice_prompt_for_profile(profile_id, profile_db, engine="qwen")

    assert prompt == {"language_argument_received": False}
