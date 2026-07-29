"""
Unit tests for the MiniMax cloud TTS backend.

These exercise region/endpoint selection, request-payload construction,
response parsing and audio decoding without making real HTTP calls.
"""

import numpy as np
import pytest

from backend.backends.minimax_backend import (
    MINIMAX_AUDIO_FORMATS,
    MINIMAX_DEFAULT_FORMAT,
    MINIMAX_DEFAULT_VOICE,
    MINIMAX_ENDPOINTS,
    MINIMAX_TTS_DEFAULT_MODEL,
    MINIMAX_TTS_MODELS,
    MINIMAX_VOICES,
    MiniMaxTTSBackend,
    _decode_audio,
    get_endpoint,
    resolve_region,
)

# -- Registry constants ------------------------------------------------------


def test_model_catalog_covers_current_speech_models():
    assert MINIMAX_TTS_DEFAULT_MODEL == "speech-2.8-hd"
    assert MINIMAX_TTS_DEFAULT_MODEL in MINIMAX_TTS_MODELS
    for expected in (
        "speech-2.8-hd",
        "speech-2.8-turbo",
        "speech-2.6-hd",
        "speech-2.6-turbo",
        "speech-02-hd",
        "speech-02-turbo",
        "speech-01-hd",
        "speech-01-turbo",
    ):
        assert expected in MINIMAX_TTS_MODELS


def test_audio_formats_and_default():
    assert MINIMAX_AUDIO_FORMATS == ["mp3", "wav", "flac", "pcm"]
    assert MINIMAX_DEFAULT_FORMAT in MINIMAX_AUDIO_FORMATS


def test_default_voice_is_a_known_preset():
    voice_ids = {vid for vid, _name, _gender, _lang in MINIMAX_VOICES}
    assert MINIMAX_DEFAULT_VOICE in voice_ids


# -- Region / endpoint selection --------------------------------------------


def test_regions_expose_both_hosts():
    assert MINIMAX_ENDPOINTS["global_en"] == "https://api.minimax.io/v1/t2a_v2"
    assert MINIMAX_ENDPOINTS["cn_zh"] == "https://api.minimaxi.com/v1/t2a_v2"


def test_resolve_region_default_is_global(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_REGION", raising=False)
    assert resolve_region() == "global_en"
    assert get_endpoint() == MINIMAX_ENDPOINTS["global_en"]


def test_resolve_region_honours_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_REGION", "cn_zh")
    assert resolve_region() == "cn_zh"
    assert get_endpoint() == MINIMAX_ENDPOINTS["cn_zh"]


def test_resolve_region_falls_back_on_unknown(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_REGION", raising=False)
    assert resolve_region("mars") == "global_en"
    assert resolve_region("CN_ZH") == "cn_zh"  # case-insensitive


# -- Payload construction ----------------------------------------------------


def test_build_payload_uses_requested_controls():
    backend = MiniMaxTTSBackend()
    prompt = {
        "preset_voice_id": "English_Persuasive_Man",
        "tts_model": "speech-2.6-turbo",
        "audio_format": "mp3",
        "sample_rate": 24000,
        "speed": 1.25,
        "vol": 0.8,
        "pitch": 2,
    }
    payload = backend._build_payload("hello", "English_Persuasive_Man", "speech-2.6-turbo", "mp3", 24000, "en", prompt)
    assert payload["model"] == "speech-2.6-turbo"
    assert payload["text"] == "hello"
    assert payload["output_format"] == "hex"
    assert payload["language_boost"] == "English"
    assert payload["voice_setting"] == {
        "voice_id": "English_Persuasive_Man",
        "speed": 1.25,
        "vol": 0.8,
        "pitch": 2,
    }
    assert payload["audio_setting"] == {"sample_rate": 24000, "format": "mp3", "channel": 1}


def test_build_payload_unknown_language_uses_auto():
    backend = MiniMaxTTSBackend()
    payload = backend._build_payload("x", MINIMAX_DEFAULT_VOICE, MINIMAX_TTS_DEFAULT_MODEL, "pcm", 32000, "xx", {})
    assert payload["language_boost"] == "auto"
    assert payload["voice_setting"]["speed"] == 1.0
    assert payload["voice_setting"]["pitch"] == 0


# -- Response parsing --------------------------------------------------------


def test_parse_response_decodes_hex_audio():
    raw = b"\x01\x02\x03\x04"
    body = {"data": {"audio": raw.hex(), "status": 2}, "base_resp": {"status_code": 0}}
    assert MiniMaxTTSBackend._parse_response(body) == raw


def test_parse_response_raises_on_api_error():
    body = {"base_resp": {"status_code": 1004, "status_msg": "invalid token"}}
    with pytest.raises(RuntimeError, match="1004"):
        MiniMaxTTSBackend._parse_response(body)


def test_parse_response_raises_on_missing_audio():
    body = {"data": {"status": 2}, "base_resp": {"status_code": 0}}
    with pytest.raises(RuntimeError, match="no audio"):
        MiniMaxTTSBackend._parse_response(body)


def test_parse_response_raises_on_malformed_hex():
    body = {"data": {"audio": "zzzz"}, "base_resp": {"status_code": 0}}
    with pytest.raises(RuntimeError, match="malformed"):
        MiniMaxTTSBackend._parse_response(body)


# -- Audio decoding ----------------------------------------------------------


def test_decode_pcm_to_float32():
    samples = np.array([0, 16384, -16384, 32767], dtype="<i2")
    audio, sr = _decode_audio(samples.tobytes(), "pcm", 32000)
    assert sr == 32000
    assert audio.dtype == np.float32
    assert audio.shape == (4,)
    assert np.isclose(audio[1], 0.5, atol=1e-3)
    assert np.isclose(audio[2], -0.5, atol=1e-3)


def test_decode_empty_returns_silence():
    audio, sr = _decode_audio(b"", "pcm", 16000)
    assert sr == 16000
    assert audio.shape == (16000,)
    assert not audio.any()


# -- End-to-end generate with mocked transport -------------------------------


async def test_load_model_without_key_raises(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr("backend.backends.minimax_backend._load_api_key", lambda: None)
    backend = MiniMaxTTSBackend()
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
        await backend.load_model()


async def test_generate_returns_decoded_audio(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-placeholder")
    backend = MiniMaxTTSBackend()

    captured = {}

    def fake_generate_sync(payload):
        captured["payload"] = payload
        samples = np.array([0, 8192, -8192], dtype="<i2")
        return samples.tobytes()

    monkeypatch.setattr(backend, "_generate_sync", fake_generate_sync)

    prompt = {"preset_voice_id": MINIMAX_DEFAULT_VOICE, "tts_model": "speech-2.8-turbo"}
    audio, sr = await backend.generate("hello world", prompt, language="en")

    assert sr == 32000
    assert audio.dtype == np.float32
    assert audio.shape == (3,)
    assert captured["payload"]["model"] == "speech-2.8-turbo"
    assert captured["payload"]["voice_setting"]["voice_id"] == MINIMAX_DEFAULT_VOICE


async def test_generate_falls_back_for_unknown_model(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-placeholder")
    backend = MiniMaxTTSBackend()

    captured = {}

    def fake_generate_sync(payload):
        captured["payload"] = payload
        return np.array([0], dtype="<i2").tobytes()

    monkeypatch.setattr(backend, "_generate_sync", fake_generate_sync)

    await backend.generate("hi", {"preset_voice_id": MINIMAX_DEFAULT_VOICE, "tts_model": "bogus"})
    assert captured["payload"]["model"] == MINIMAX_TTS_DEFAULT_MODEL
    assert captured["payload"]["audio_setting"]["format"] == MINIMAX_DEFAULT_FORMAT


async def test_create_voice_prompt_returns_default_preset():
    backend = MiniMaxTTSBackend()
    prompt, cached = await backend.create_voice_prompt("ignored.wav", "ref text")
    assert cached is False
    assert prompt["voice_type"] == "preset"
    assert prompt["preset_engine"] == "minimax"
    assert prompt["preset_voice_id"] == MINIMAX_DEFAULT_VOICE
