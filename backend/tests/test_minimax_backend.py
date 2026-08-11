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
    MINIMAX_FILE_UPLOAD_ENDPOINTS,
    MINIMAX_TTS_DEFAULT_MODEL,
    MINIMAX_TTS_MODELS,
    MINIMAX_VOICE_CLONE_DEFAULT_MODEL,
    MINIMAX_VOICE_CLONE_ENDPOINTS,
    MINIMAX_VOICE_CLONE_MODELS,
    MINIMAX_VOICES,
    MiniMaxTTSBackend,
    _decode_audio,
    get_endpoint,
    get_file_upload_endpoint,
    get_voice_clone_endpoint,
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

    assert MINIMAX_VOICE_CLONE_DEFAULT_MODEL == "speech-2.8-hd"
    assert MINIMAX_VOICE_CLONE_MODELS == [
        "speech-2.8-hd",
        "speech-2.6-hd",
        "speech-02-hd",
        "speech-01-hd",
    ]


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
    assert MINIMAX_FILE_UPLOAD_ENDPOINTS["global_en"] == "https://api.minimax.io/v1/files/upload"
    assert MINIMAX_FILE_UPLOAD_ENDPOINTS["cn_zh"] == "https://api.minimaxi.com/v1/files/upload"
    assert MINIMAX_VOICE_CLONE_ENDPOINTS["global_en"] == "https://api.minimax.io/v1/voice_clone"
    assert MINIMAX_VOICE_CLONE_ENDPOINTS["cn_zh"] == "https://api.minimaxi.com/v1/voice_clone"


def test_resolve_region_default_is_global(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_REGION", raising=False)
    assert resolve_region() == "global_en"
    assert get_endpoint() == MINIMAX_ENDPOINTS["global_en"]


def test_resolve_region_honours_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_REGION", "cn_zh")
    assert resolve_region() == "cn_zh"
    assert get_endpoint() == MINIMAX_ENDPOINTS["cn_zh"]
    assert get_file_upload_endpoint() == MINIMAX_FILE_UPLOAD_ENDPOINTS["cn_zh"]
    assert get_voice_clone_endpoint() == MINIMAX_VOICE_CLONE_ENDPOINTS["cn_zh"]


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
        "pronunciation_dict": {"tone": ["MiniMax/(min-ee-max)"]},
        "voice_modify": {"pitch": 1},
        "subtitle_enable": True,
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
    assert payload["pronunciation_dict"] == prompt["pronunciation_dict"]
    assert payload["voice_modify"] == prompt["voice_modify"]
    assert payload["subtitle_enable"] is True


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

    prompt = {"voice_id": "voicebox_12345678", "tts_model": "speech-2.8-turbo"}
    audio, sr = await backend.generate("hello world", prompt, language="en")

    assert sr == 32000
    assert audio.dtype == np.float32
    assert audio.shape == (3,)
    assert captured["payload"]["model"] == "speech-2.8-turbo"
    assert captured["payload"]["voice_setting"]["voice_id"] == "voicebox_12345678"


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


async def test_create_voice_prompt_clones_and_caches_reference_audio(tmp_path, monkeypatch):
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"reference-audio")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-placeholder")

    backend = MiniMaxTTSBackend()
    captured = {}

    def fake_create_cloned_voice(audio_path, requested_voice_id):
        captured["audio_path"] = audio_path
        captured["requested_voice_id"] = requested_voice_id
        return requested_voice_id

    monkeypatch.setattr(backend, "_create_cloned_voice_sync", fake_create_cloned_voice)
    monkeypatch.setattr("backend.backends.minimax_backend.get_cached_voice_prompt", lambda _key: None)
    monkeypatch.setattr(
        "backend.backends.minimax_backend.cache_voice_prompt",
        lambda key, prompt: captured.update(cache_key=key, cached_prompt=prompt),
    )

    prompt, cached = await backend.create_voice_prompt(str(audio_path), "ref text")
    assert cached is False
    assert prompt["voice_type"] == "cloned"
    assert prompt["voice_id"].startswith("voicebox_")
    assert captured["audio_path"] == str(audio_path)
    assert captured["requested_voice_id"] == prompt["voice_id"]
    assert captured["cache_key"].startswith("minimax_")
    assert captured["cached_prompt"] == prompt


def test_create_cloned_voice_uploads_audio_and_requests_clone(tmp_path, monkeypatch):
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"reference-audio")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-placeholder")

    requests = []

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class FakeClient:
        def __init__(self, timeout):
            assert timeout == 60.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            if url.endswith("/files/upload"):
                return FakeResponse({"file": {"file_id": 12345}, "base_resp": {"status_code": 0}})
            return FakeResponse({"voice_id": "voicebox_12345678", "base_resp": {"status_code": 0}})

    monkeypatch.setattr("httpx.Client", FakeClient)

    backend = MiniMaxTTSBackend()
    voice_id = backend._create_cloned_voice_sync(str(audio_path), "voicebox_12345678")

    assert voice_id == "voicebox_12345678"
    assert requests[0][0] == MINIMAX_FILE_UPLOAD_ENDPOINTS["global_en"]
    assert requests[0][1]["data"] == {"purpose": "voice_clone"}
    assert requests[0][1]["files"]["file"][0] == "reference.wav"
    assert requests[1][0] == MINIMAX_VOICE_CLONE_ENDPOINTS["global_en"]
    assert requests[1][1]["json"] == {
        "file_id": 12345,
        "voice_id": "voicebox_12345678",
        "model": MINIMAX_VOICE_CLONE_DEFAULT_MODEL,
    }
