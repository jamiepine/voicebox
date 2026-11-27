"""Multi-sample reference reuse, invalidation and non-blocking preparation."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import backends, config
from backend.backends import base
from backend.database import Base, ProfileSample, VoiceProfile
from backend.services import profiles
from backend.utils.cache import clear_profile_cache


@pytest.fixture
def voice(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(VoiceProfile(id="voice", name="Test", voice_type="cloned"))
    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for index, path in enumerate(paths):
        path.write_bytes(f"reference audio {index}".encode())
        db.add(ProfileSample(id=str(index), profile_id="voice", audio_path=str(path), reference_text=f"sample {index}"))
    db.commit()
    cache = tmp_path / "cache"
    monkeypatch.setattr(config, "get_cache_dir", lambda: cache)
    combinations, prompts = [], []

    class Backend:
        async def combine_voice_prompts(self, audio_paths, texts):
            combinations.append((list(audio_paths), list(texts)))
            return np.ones(240, dtype=np.float32), " ".join(texts)

        async def create_voice_prompt(self, path, text, **kwargs):
            prompts.append((path, text, kwargs))
            assert Path(path).is_file()
            return {"path": path, "text": text}, False

    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda _: Backend())

    def generate(**kwargs):
        return asyncio.run(profiles.create_voice_prompt_for_profile("voice", db, **kwargs))

    yield SimpleNamespace(
        db=db, paths=paths, cache=cache, combinations=combinations, prompts=prompts, generate=generate
    )
    db.close()
    engine.dispose()


def test_repeated_generation_reuses_combined_wav(voice, monkeypatch):
    first = voice.generate()
    monkeypatch.setattr(profiles, "save_audio", lambda *args: pytest.fail("cache hit rewrote audio"))
    assert voice.generate() == first
    assert len(voice.combinations) == 1


@pytest.mark.parametrize("change", ["audio", "transcript", "engine"])
def test_changed_reference_or_engine_invalidates_combined_wav(voice, change):
    first = voice.generate()
    kwargs = {}
    if change == "audio":
        voice.paths[0].write_bytes(b"replacement audio")
    elif change == "transcript":
        voice.db.get(ProfileSample, "1").reference_text = "corrected transcript"
        voice.db.commit()
    else:
        kwargs["engine"] = "luxtts"
    second = voice.generate(**kwargs)
    assert second["path"] != first["path"]
    assert len(voice.combinations) == 2
    assert voice.generate(**kwargs) == second
    if change == "transcript":
        assert second["text"] == "sample 0 corrected transcript"


def test_sample_order_is_part_of_cache_identity(voice):
    first = voice.generate()
    samples = list(reversed(voice.db.query(ProfileSample).filter_by(profile_id="voice").all()))

    class ReorderedSession:
        def query(self, entity):
            if entity is ProfileSample:
                return SimpleNamespace(filter_by=lambda **kwargs: SimpleNamespace(all=lambda: samples))
            return voice.db.query(entity)

    second = asyncio.run(profiles.create_voice_prompt_for_profile("voice", ReorderedSession()))
    assert second["path"] != first["path"]
    assert second["text"] == "sample 1 sample 0"
    assert len(voice.combinations) == 2


def test_cache_bypass_rebuilds_and_is_forwarded_to_backend(voice):
    first = voice.generate()
    assert voice.generate(use_cache=False) == first
    assert len(voice.combinations) == 2
    assert voice.prompts[-1][2] == {"use_cache": False}
    assert voice.generate() == first
    assert len(voice.combinations) == 2


def test_missing_combined_wav_is_rebuilt(voice):
    first = voice.generate()
    Path(first["path"]).unlink()
    assert voice.generate() == first
    assert len(voice.combinations) == 2


def test_single_sample_keeps_existing_backend_path(voice):
    voice.db.delete(voice.db.get(ProfileSample, "1"))
    voice.db.commit()
    result = voice.generate(use_cache=False)
    assert result == {"path": str(voice.paths[0]), "text": "sample 0"}
    assert voice.combinations == []
    assert voice.prompts[-1][2] == {"use_cache": False}
    assert not voice.cache.exists()


def test_profile_cache_clear_removes_combined_wavs(voice):
    first = voice.generate()
    voice.paths[0].write_bytes(b"replacement audio")
    second = voice.generate()
    assert clear_profile_cache("voice") == 2
    assert not Path(first["path"]).exists()
    assert not Path(second["path"]).exists()


def test_hashing_and_atomic_wav_write_run_off_event_loop(voice, monkeypatch):
    event_loop_thread = threading.get_ident()
    original_key, original_save = profiles.get_cache_key, profiles.save_audio
    calls = []

    def key(*args):
        assert threading.get_ident() != event_loop_thread
        calls.append("hash")
        return original_key(*args)

    def save(*args):
        assert threading.get_ident() != event_loop_thread
        assert not list(voice.cache.glob("combined_voice_*.wav"))
        calls.append("save")
        return original_save(*args)

    monkeypatch.setattr(profiles, "get_cache_key", key)
    monkeypatch.setattr(profiles, "save_audio", save)
    voice.generate()
    assert calls == ["hash", "hash", "save"]


def test_failed_write_does_not_publish_partial_cache_and_can_retry(voice, monkeypatch):
    def fail_save(audio, path, sample_rate):
        Path(path).write_bytes(b"partial WAV")
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(profiles, "save_audio", fail_save)
        with pytest.raises(OSError, match="disk full"):
            voice.generate()
    assert list(voice.cache.iterdir()) == []
    assert voice.prompts == []
    voice.generate()
    assert len(voice.combinations) == 2


def test_failed_refresh_preserves_previous_complete_cache(voice, monkeypatch):
    first = voice.generate()
    previous = Path(first["path"]).read_bytes()

    def fail_save(audio, path, sample_rate):
        Path(path).write_bytes(b"partial WAV")
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(profiles, "save_audio", fail_save)
        with pytest.raises(OSError, match="disk full"):
            voice.generate(use_cache=False)
    assert Path(first["path"]).read_bytes() == previous
    assert voice.generate() == first
    assert len(list(voice.cache.iterdir())) == 1


@pytest.mark.parametrize("sample_rate", [None, 24000])
def test_audio_decoding_runs_off_event_loop(monkeypatch, sample_rate):
    event_loop_thread = threading.get_ident()
    calls = []

    def load(path, **kwargs):
        assert threading.get_ident() != event_loop_thread
        calls.append((path, kwargs))
        return np.ones(100, dtype=np.float32), 24000

    monkeypatch.setattr(base, "load_audio", load)
    monkeypatch.setattr(base, "normalize_audio", lambda audio: audio)
    audio, text = asyncio.run(base.combine_voice_prompts(["a", "b"], ["Hello", "world"], sample_rate=sample_rate))
    assert len(audio) == 200
    assert text == "Hello world"
    kwargs = {"sample_rate": sample_rate} if sample_rate else {}
    assert calls == [("a", kwargs), ("b", kwargs)]
