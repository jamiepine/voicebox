"""Integration check for the Higgs Audio v3 MLX engine (Apple Silicon + cached weights only)."""

import asyncio
import platform

import numpy as np
import pytest

from backend.backends import get_model_config, get_tts_backend_for_engine
from backend.backends.higgs_mlx_backend import HiggsMLXTTSBackend
from backend.utils.platform_detect import get_backend_type

pytestmark = pytest.mark.skipif(
    platform.machine() != "arm64" or get_backend_type() != "mlx",
    reason="Higgs Audio v3 runs on MLX only",
)


def test_registry_wiring():
    cfg = get_model_config("higgs-audio-v3")
    assert cfg is not None and cfg.engine == "higgs" and "fr" in cfg.languages
    assert isinstance(get_tts_backend_for_engine("higgs"), HiggsMLXTTSBackend)


def test_generate_and_clone(tmp_path):
    backend = HiggsMLXTTSBackend()
    if not backend._is_model_cached():
        pytest.skip("bosonai/higgs-tts-3-4b not downloaded")

    async def run():
        audio, sr = await backend.generate("Bonjour, ceci est un test.", {}, language="fr", seed=1)
        assert sr == 24000 and audio.dtype == np.float32 and len(audio) > sr

        import soundfile as sf

        ref = tmp_path / "ref.wav"
        sf.write(ref, audio, sr)
        prompt, cached = await backend.create_voice_prompt(str(ref), "Bonjour, ceci est un test.")
        assert not cached
        cloned, _ = await backend.generate("Une deuxième phrase, avec la même voix.", prompt, seed=1)
        assert len(cloned) > sr and np.isfinite(cloned).all() and np.abs(cloned).max() > 0.01
        backend.unload_model()
        assert not backend.is_loaded()

    asyncio.run(run())
