"""
Regression tests: unloading a TTS model must also drop the in-memory
voice-prompt cache.

`backend/utils/cache.py` keeps a process-lifetime `_memory_cache` dict of
voice-clone prompts (tensors or device-backed dicts produced by whichever
TTS model created them). Before this fix, nothing ever cleared it on
unload, so those prompts stayed referenced — and their memory held —
indefinitely, even after the model that produced them was gone. Whisper
and the LLM backends never produce voice prompts, so their unload paths
must leave the cache alone.
"""

from unittest.mock import MagicMock

import pytest

from backend import backends as backends_module
from backend.backends import ModelConfig
from backend.services import tts as tts_service
from backend.utils import cache as cache_module


@pytest.fixture(autouse=True)
def _reset_memory_cache():
    """Isolate each test from the shared process-lifetime _memory_cache dict."""
    cache_module._memory_cache.clear()
    yield
    cache_module._memory_cache.clear()


def _populate_memory_cache():
    cache_module._memory_cache["some-cache-key"] = {"ref_audio": "x.wav", "ref_text": "hi"}


def test_clear_voice_prompt_memory_cache_leaves_disk_cache_alone(tmp_path, monkeypatch):
    """Clearing the memory cache must not touch cached .prompt files on disk."""
    monkeypatch.setattr(cache_module, "_get_cache_dir", lambda: tmp_path)
    disk_file = tmp_path / "some-cache-key.prompt"
    disk_file.write_bytes(b"fake torch.save payload")
    _populate_memory_cache()

    cache_module.clear_voice_prompt_memory_cache()

    assert cache_module._memory_cache == {}
    assert disk_file.exists()


def test_unload_tts_model_clears_voice_prompt_memory_cache(monkeypatch):
    """/models/unload (the legacy qwen-only endpoint) must clear the prompt cache."""
    fake_backend = MagicMock()
    monkeypatch.setattr(tts_service, "get_tts_backend", lambda: fake_backend)
    _populate_memory_cache()

    tts_service.unload_tts_model()

    fake_backend.unload_model.assert_called_once()
    assert cache_module._memory_cache == {}


def test_unload_model_by_config_clears_cache_for_generic_tts_engine(monkeypatch):
    """Unloading any non-qwen TTS engine (e.g. kokoro) must clear the prompt cache."""
    fake_backend = MagicMock()
    fake_backend.is_loaded.return_value = True
    monkeypatch.setattr(backends_module, "get_tts_backend_for_engine", lambda engine: fake_backend)
    _populate_memory_cache()

    config = ModelConfig(model_name="kokoro", display_name="Kokoro", engine="kokoro", hf_repo_id="x/y")
    was_loaded = backends_module.unload_model_by_config(config)

    assert was_loaded is True
    fake_backend.unload_model.assert_called_once()
    assert cache_module._memory_cache == {}


def test_unload_model_by_config_leaves_cache_alone_for_whisper(monkeypatch):
    """Whisper never produces voice prompts, so unloading it must not touch the cache."""
    fake_whisper = MagicMock()
    fake_whisper.is_loaded.return_value = True
    fake_whisper.model_size = "base"
    monkeypatch.setattr("backend.services.transcribe.get_whisper_model", lambda: fake_whisper)
    _populate_memory_cache()
    cache_before = dict(cache_module._memory_cache)

    config = ModelConfig(
        model_name="whisper-base", display_name="Whisper Base", engine="whisper", hf_repo_id="x/y", model_size="base"
    )
    was_loaded = backends_module.unload_model_by_config(config)

    assert was_loaded is True
    assert cache_module._memory_cache == cache_before
