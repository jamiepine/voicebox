"""
Smoke test for the MLX backend dependencies on Apple Silicon.

Guards the `--no-deps` install of mlx-audio/mlx-lm done by `just setup-python`
and release.yml: those packages skip their declared dependencies (transformers
>=5.x conflict), so a missing transitive dep only surfaces at import time.
This test fails fast if the MLX STT/TTS entry points the backend uses stop
importing (e.g. the `miniaudio` regression from issue #505).

Usage:
    python -m pytest backend/tests/test_mlx_smoke.py -v
"""

import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not (sys.platform == "darwin" and platform.machine() == "arm64"),
    reason="MLX packages are only installed on Apple Silicon macOS",
)


def test_mlx_core_runs():
    """The MLX runtime itself works (Metal array op)."""
    import mlx.core as mx

    assert mx.array([1, 2]).sum().item() == 3


def test_mlx_audio_tts_entry_point():
    """`from mlx_audio.tts import load` — used by MLXBackend.load_model_async."""
    from mlx_audio.tts import load

    assert callable(load)


def test_mlx_audio_stt_entry_point():
    """`from mlx_audio.stt import load` — used by the Whisper MLX STT path.

    Importing mlx_audio.stt also pulls in miniaudio, so this catches the
    ModuleNotFoundError from issue #505 on fresh installs.
    """
    from mlx_audio.stt import load

    assert callable(load)


def test_mlx_lm_entry_points():
    """`mlx_lm.load` / `mlx_lm.generate` — used by qwen_llm_backend."""
    from mlx_lm import generate, load

    assert callable(load)
    assert callable(generate)
