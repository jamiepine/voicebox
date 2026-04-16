"""
Unit tests for the MiniMax TTS backend.

These tests exercise backend logic without making real HTTP calls.
Integration tests (skipped when MINIMAX_API_KEY is absent) call the
live MiniMax API.

Usage:
    cd backend
    python -m pytest tests/test_minimax_backend.py -v
"""

from __future__ import annotations

import asyncio
import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sse_response(chunks: list[str]) -> list[bytes]:
    """Build raw SSE bytes from a list of hex audio strings."""
    import json

    lines: list[bytes] = []
    for i, hex_chunk in enumerate(chunks):
        event = {"data": {"status": 1, "audio": hex_chunk}, "base_resp": {"status_code": 0, "status_msg": "success"}}
        lines.append(f"data: {json.dumps(event)}\n\n".encode())
    # final [DONE]
    lines.append(b"data: [DONE]\n\n")
    return lines


# ---------------------------------------------------------------------------
# Module-level imports (no heavy ML deps needed)
# ---------------------------------------------------------------------------


def test_backend_can_be_imported():
    """The backend module must import without heavy ML dependencies."""
    from backend.backends.minimax_backend import MiniMaxTTSBackend  # noqa: F401

    assert MiniMaxTTSBackend is not None


def test_constants_are_sane():
    """Check that public constants have expected values."""
    from backend.backends.minimax_backend import (
        MINIMAX_TTS_BASE_URL,
        MINIMAX_TTS_DEFAULT_MODEL,
        MINIMAX_TTS_DEFAULT_VOICE,
        MINIMAX_TTS_SAMPLE_RATE,
        MINIMAX_VOICES,
        MINIMAX_TTS_MODELS,
    )

    assert MINIMAX_TTS_BASE_URL.startswith("https://api.minimax.io")
    assert MINIMAX_TTS_DEFAULT_MODEL in MINIMAX_TTS_MODELS
    assert MINIMAX_TTS_DEFAULT_VOICE in {v[0] for v in MINIMAX_VOICES}
    assert MINIMAX_TTS_SAMPLE_RATE > 0
    assert len(MINIMAX_VOICES) >= 6
    assert "speech-2.8-hd" in MINIMAX_TTS_MODELS
    assert "speech-2.8-turbo" in MINIMAX_TTS_MODELS


def test_pcm_decode():
    """PCM bytes decode correctly to float32 numpy arrays."""
    from backend.backends.minimax_backend import _pcm_bytes_to_numpy, MINIMAX_TTS_SAMPLE_RATE

    # Two samples: 0x7FFF (max positive) and 0x8001 (near max negative)
    import struct

    pcm_bytes = struct.pack("<h", 32767) + struct.pack("<h", -32767)
    audio, sr = _pcm_bytes_to_numpy(pcm_bytes, MINIMAX_TTS_SAMPLE_RATE)

    assert audio.dtype == np.float32
    assert len(audio) == 2
    assert sr == MINIMAX_TTS_SAMPLE_RATE
    assert abs(audio[0] - (32767 / 32768.0)) < 1e-4
    assert abs(audio[1] - (-32767 / 32768.0)) < 1e-4


def test_pcm_decode_empty_returns_silence():
    from backend.backends.minimax_backend import _pcm_bytes_to_numpy, MINIMAX_TTS_SAMPLE_RATE

    audio, sr = _pcm_bytes_to_numpy(b"", MINIMAX_TTS_SAMPLE_RATE)
    assert len(audio) == MINIMAX_TTS_SAMPLE_RATE  # 1 second of silence
    assert (audio == 0).all()


# ---------------------------------------------------------------------------
# _load_api_key
# ---------------------------------------------------------------------------


def test_load_api_key_from_env():
    from backend.backends.minimax_backend import _load_api_key

    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-from-env"}):
        assert _load_api_key() == "test-key-from-env"


def test_load_api_key_missing_returns_none(tmp_path):
    from backend.backends.minimax_backend import _load_api_key

    with patch.dict(os.environ, {}, clear=True):
        # Patch os.path.expanduser to point to a non-existent file
        with patch("backend.backends.minimax_backend.os.path.expanduser", return_value=str(tmp_path / "nonexistent.env")):
            result = _load_api_key()
    assert result is None


def test_load_api_key_from_env_local(tmp_path):
    from backend.backends.minimax_backend import _load_api_key

    env_local = tmp_path / ".env.local"
    env_local.write_text("MINIMAX_API_KEY=key-from-file\n")

    with patch.dict(os.environ, {}, clear=True):
        with patch("backend.backends.minimax_backend.os.path.expanduser", return_value=str(env_local)):
            with patch("backend.backends.minimax_backend.os.path.exists", return_value=True):
                result = _load_api_key()
    assert result == "key-from-file"


# ---------------------------------------------------------------------------
# MiniMaxTTSBackend — structural / non-network tests
# ---------------------------------------------------------------------------


class TestMiniMaxTTSBackendBasics:
    def setup_method(self):
        from backend.backends.minimax_backend import MiniMaxTTSBackend

        self.backend = MiniMaxTTSBackend()

    def test_is_loaded_false_without_key(self):
        with patch("backend.backends.minimax_backend._load_api_key", return_value=None):
            self.backend._api_key = None
            assert not self.backend.is_loaded()

    def test_is_loaded_true_with_key(self):
        with patch("backend.backends.minimax_backend._load_api_key", return_value="sk-test"):
            self.backend._api_key = None
            assert self.backend.is_loaded()

    def test_unload_model_is_noop(self):
        """unload_model must not raise."""
        self.backend.unload_model()

    def test_get_model_path_returns_endpoint(self):
        path = self.backend._get_model_path("default")
        assert "minimax.io" in path
        assert "t2a_v2" in path

    def test_is_model_cached_reflects_key_presence(self):
        with patch.object(self.backend, "is_loaded", return_value=True):
            assert self.backend._is_model_cached()
        with patch.object(self.backend, "is_loaded", return_value=False):
            assert not self.backend._is_model_cached()

    @pytest.mark.asyncio
    async def test_load_model_raises_without_key(self):
        self.backend._api_key = None
        with patch("backend.backends.minimax_backend._load_api_key", return_value=None):
            with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
                await self.backend.load_model()

    @pytest.mark.asyncio
    async def test_load_model_succeeds_with_key(self):
        self.backend._api_key = None
        with patch("backend.backends.minimax_backend._load_api_key", return_value="sk-test"):
            await self.backend.load_model()  # should not raise
        assert self.backend._api_key == "sk-test"


class TestMiniMaxVoicePrompt:
    def setup_method(self):
        from backend.backends.minimax_backend import MiniMaxTTSBackend

        self.backend = MiniMaxTTSBackend()

    @pytest.mark.asyncio
    async def test_create_voice_prompt_returns_preset(self):
        prompt, was_cached = await self.backend.create_voice_prompt("/dummy/path.wav", "hello")
        assert prompt["voice_type"] == "preset"
        assert prompt["preset_engine"] == "minimax"
        assert "preset_voice_id" in prompt
        assert not was_cached


class TestMiniMaxGenerateSyncParsing:
    """Test SSE parsing logic in _generate_sync via unit mocking."""

    def _build_backend_with_key(self):
        from backend.backends.minimax_backend import MiniMaxTTSBackend

        b = MiniMaxTTSBackend()
        b._api_key = "sk-test"
        return b

    def test_hex_audio_collected_from_sse(self):
        import json

        backend = self._build_backend_with_key()

        # Build fake SSE events with known hex data (PCM bytes for 2 samples of silence)
        # 4 bytes = 2 signed int16 samples at 0
        hex1 = "00000000"
        hex2 = "00000000"
        events = [
            {"data": {"status": 1, "audio": hex1}, "base_resp": {"status_code": 0}},
            {"data": {"status": 1, "audio": hex2}, "base_resp": {"status_code": 0}},
            # status=2 aggregated audio should be ignored when exclude_aggregated_audio is set
            {"data": {"status": 2, "audio": "deadbeef"}, "base_resp": {"status_code": 0}},
        ]
        sse_text = "".join(f"data: {json.dumps(e)}\n\n" for e in events)

        # Mock the httpx streaming response
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_text.return_value = iter([sse_text])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_stream = MagicMock()
        mock_stream.__enter__ = lambda s: mock_resp
        mock_stream.__exit__ = MagicMock(return_value=False)

        # We expect the returned bytes to be hex1 + hex2 only (not the aggregated audio)
        expected_bytes = bytes.fromhex(hex1 + hex2)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = lambda s: mock_client
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.return_value = mock_stream

            result = backend._generate_sync("hello", "English_Graceful_Lady", "speech-2.8-hd")

        assert result == expected_bytes

    def test_api_error_status_code_raises(self):
        import json

        backend = self._build_backend_with_key()

        error_event = {"data": {}, "base_resp": {"status_code": 1004, "status_msg": "Auth failed"}}
        sse_text = f"data: {json.dumps(error_event)}\n\n"

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_text.return_value = iter([sse_text])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_stream = MagicMock()
        mock_stream.__enter__ = lambda s: mock_resp
        mock_stream.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = lambda s: mock_client
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.return_value = mock_stream

            with pytest.raises(RuntimeError, match="1004"):
                backend._generate_sync("hello", "English_Graceful_Lady", "speech-2.8-hd")

    def test_empty_response_raises(self):
        import json

        backend = self._build_backend_with_key()

        # SSE with no audio data (e.g., only [DONE])
        sse_text = "data: [DONE]\n\n"

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_text.return_value = iter([sse_text])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_stream = MagicMock()
        mock_stream.__enter__ = lambda s: mock_resp
        mock_stream.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = lambda s: mock_client
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.return_value = mock_stream

            with pytest.raises(RuntimeError, match="no audio data"):
                backend._generate_sync("hello", "English_Graceful_Lady", "speech-2.8-hd")


# ---------------------------------------------------------------------------
# Registration — engine registry tests
# ---------------------------------------------------------------------------


def test_minimax_in_tts_engines():
    from backend.backends import TTS_ENGINES

    assert "minimax" in TTS_ENGINES


def test_get_tts_backend_for_engine_minimax():
    """get_tts_backend_for_engine('minimax') must return a MiniMaxTTSBackend."""
    from backend.backends import get_tts_backend_for_engine, reset_backends
    from backend.backends.minimax_backend import MiniMaxTTSBackend

    reset_backends()
    backend = get_tts_backend_for_engine("minimax")
    assert isinstance(backend, MiniMaxTTSBackend)
    reset_backends()


# ---------------------------------------------------------------------------
# Profile service — preset voice validation
# ---------------------------------------------------------------------------


def test_minimax_preset_voice_ids_registered():
    from backend.services.profiles import _get_preset_voice_ids
    from backend.backends.minimax_backend import MINIMAX_VOICES

    ids = _get_preset_voice_ids("minimax")
    assert len(ids) > 0
    for voice_id, _name, _lang in MINIMAX_VOICES:
        assert voice_id in ids


def test_minimax_not_in_cloning_engines():
    from backend.services.profiles import CLONING_ENGINES

    assert "minimax" not in CLONING_ENGINES


# ---------------------------------------------------------------------------
# GenerationRequest model — engine field validation
# ---------------------------------------------------------------------------


def test_generation_request_accepts_minimax():
    from backend.models import GenerationRequest

    req = GenerationRequest(profile_id="abc", text="hello", engine="minimax")
    assert req.engine == "minimax"


def test_generation_request_rejects_unknown_engine():
    from backend.models import GenerationRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GenerationRequest(profile_id="abc", text="hello", engine="openai-tts")


# ---------------------------------------------------------------------------
# Integration test (real API call — skipped without key)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_generate_speech():
    """
    Live integration test: call MiniMax TTS API and verify we get audio back.

    Skipped automatically when MINIMAX_API_KEY is not configured.
    """
    from backend.backends.minimax_backend import MiniMaxTTSBackend, _load_api_key, _pcm_bytes_to_numpy, MINIMAX_TTS_SAMPLE_RATE

    api_key = _load_api_key()
    if not api_key:
        pytest.skip("MINIMAX_API_KEY not configured")

    backend = MiniMaxTTSBackend()
    await backend.load_model()

    voice_prompt = {
        "voice_type": "preset",
        "preset_engine": "minimax",
        "preset_voice_id": "English_Graceful_Lady",
    }

    audio, sr = await backend.generate(
        text="Hello, this is a MiniMax TTS integration test.",
        voice_prompt=voice_prompt,
    )

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert sr == MINIMAX_TTS_SAMPLE_RATE
