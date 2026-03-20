"""Unit tests for MiniMax Cloud TTS backend."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestMiniMaxTTSBackend(unittest.TestCase):
    """Tests for MiniMaxTTSBackend."""

    def _make_backend(self):
        from backend.backends.minimax_backend import MiniMaxTTSBackend
        return MiniMaxTTSBackend()

    def test_initial_state(self):
        backend = self._make_backend()
        self.assertFalse(backend.is_loaded())
        self.assertTrue(backend._is_model_cached())

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_load_model_sets_ready(self):
        import asyncio
        backend = self._make_backend()
        asyncio.get_event_loop().run_until_complete(backend.load_model())
        self.assertTrue(backend.is_loaded())
        self.assertEqual(backend._api_key, "test-key-123")

    @patch.dict(os.environ, {}, clear=True)
    def test_load_model_without_api_key_raises(self):
        import asyncio
        backend = self._make_backend()
        # Remove MINIMAX_API_KEY if present
        os.environ.pop("MINIMAX_API_KEY", None)
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.get_event_loop().run_until_complete(backend.load_model())
        self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_unload_model(self):
        import asyncio
        backend = self._make_backend()
        asyncio.get_event_loop().run_until_complete(backend.load_model())
        self.assertTrue(backend.is_loaded())
        backend.unload_model()
        self.assertFalse(backend.is_loaded())
        self.assertIsNone(backend._api_key)

    def test_create_voice_prompt_returns_preset(self):
        import asyncio
        backend = self._make_backend()
        prompt, cached = asyncio.get_event_loop().run_until_complete(
            backend.create_voice_prompt("/fake/audio.wav", "test text")
        )
        self.assertEqual(prompt["voice_type"], "preset")
        self.assertEqual(prompt["preset_engine"], "minimax")
        self.assertIn("preset_voice_id", prompt)
        self.assertFalse(cached)

    def test_combine_voice_prompts_raises(self):
        import asyncio
        backend = self._make_backend()
        with self.assertRaises(NotImplementedError):
            asyncio.get_event_loop().run_until_complete(
                backend.combine_voice_prompts(["/a.wav"], ["text"])
            )

    def test_get_model_path(self):
        backend = self._make_backend()
        self.assertEqual(backend._get_model_path(), "speech-2.8-hd")

    def test_is_model_cached_always_true(self):
        backend = self._make_backend()
        self.assertTrue(backend._is_model_cached())
        self.assertTrue(backend._is_model_cached("anything"))


class TestMiniMaxVoices(unittest.TestCase):
    """Tests for MiniMax voice definitions."""

    def test_voices_structure(self):
        from backend.backends.minimax_backend import MINIMAX_VOICES
        self.assertGreater(len(MINIMAX_VOICES), 0)
        for voice_id, name, gender, lang in MINIMAX_VOICES:
            self.assertIsInstance(voice_id, str)
            self.assertIsInstance(name, str)
            self.assertIn(gender, ("male", "female"))
            self.assertIsInstance(lang, str)

    def test_default_voice_id_in_list(self):
        from backend.backends.minimax_backend import MINIMAX_VOICES, DEFAULT_VOICE_ID
        voice_ids = [v[0] for v in MINIMAX_VOICES]
        self.assertIn(DEFAULT_VOICE_ID, voice_ids)


class TestMiniMaxGenerate(unittest.TestCase):
    """Tests for MiniMax TTS generate with mocked API."""

    def _make_mock_response(self, audio_samples=None):
        """Create a mock API response with valid PCM audio."""
        if audio_samples is None:
            # Generate 1 second of silence at 24kHz
            audio_samples = np.zeros(24000, dtype=np.int16)
        audio_hex = audio_samples.tobytes().hex()
        return json.dumps({
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": audio_hex},
        }).encode("utf-8")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_returns_audio(self, mock_urlopen):
        import asyncio

        mock_resp = MagicMock()
        mock_resp.read.return_value = self._make_mock_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = self._make_backend()
        audio, sr = asyncio.get_event_loop().run_until_complete(
            backend.generate(
                "Hello world",
                {"voice_type": "preset", "preset_voice_id": "English_Graceful_Lady"},
            )
        )
        self.assertEqual(sr, 24000)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), 24000)  # 1 second

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_sends_correct_payload(self, mock_urlopen):
        import asyncio

        mock_resp = MagicMock()
        mock_resp.read.return_value = self._make_mock_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = self._make_backend()
        asyncio.get_event_loop().run_until_complete(
            backend.generate(
                "Test text",
                {"preset_voice_id": "Deep_Voice_Man"},
            )
        )

        # Verify the request was made with correct parameters
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))

        self.assertEqual(payload["model"], "speech-2.8-hd")
        self.assertEqual(payload["text"], "Test text")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["voice_setting"]["voice_id"], "Deep_Voice_Man")
        self.assertEqual(payload["audio_setting"]["format"], "pcm")
        self.assertEqual(payload["audio_setting"]["sample_rate"], 24000)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_api_error_raises(self, mock_urlopen):
        import asyncio

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "base_resp": {"status_code": 1001, "status_msg": "Invalid API key"},
            "data": {},
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = self._make_backend()
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.get_event_loop().run_until_complete(
                backend.generate("test", {"preset_voice_id": "English_Graceful_Lady"})
            )
        self.assertIn("Invalid API key", str(ctx.exception))

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("urllib.request.urlopen")
    def test_generate_uses_default_voice_id(self, mock_urlopen):
        import asyncio

        mock_resp = MagicMock()
        mock_resp.read.return_value = self._make_mock_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = self._make_backend()
        asyncio.get_event_loop().run_until_complete(
            backend.generate("test", {})
        )

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["voice_setting"]["voice_id"], "English_Graceful_Lady")

    def _make_backend(self):
        from backend.backends.minimax_backend import MiniMaxTTSBackend
        return MiniMaxTTSBackend()


class TestMiniMaxEngineRegistration(unittest.TestCase):
    """Tests for MiniMax engine registration in backends __init__."""

    def test_minimax_in_tts_engines(self):
        from backend.backends import TTS_ENGINES
        self.assertIn("minimax", TTS_ENGINES)

    def test_get_tts_backend_for_engine(self):
        from backend.backends import get_tts_backend_for_engine, reset_backends
        reset_backends()
        backend = get_tts_backend_for_engine("minimax")
        from backend.backends.minimax_backend import MiniMaxTTSBackend
        self.assertIsInstance(backend, MiniMaxTTSBackend)
        reset_backends()


if __name__ == "__main__":
    unittest.main()
