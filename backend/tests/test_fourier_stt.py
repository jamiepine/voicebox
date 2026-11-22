import numpy as np

from backend.backends import fourier_stt


def test_long_audio_chunks_preserve_all_samples():
    rng = np.random.default_rng(5)
    audio = rng.normal(size=fourier_stt.SAMPLE_RATE * 68).astype(np.float32)
    audio[27 * fourier_stt.SAMPLE_RATE : 27 * fourier_stt.SAMPLE_RATE + 320] = 0
    chunks = list(fourier_stt.audio_chunks(audio))
    assert len(chunks) == 3
    assert len(chunks[0]) == 27 * fourier_stt.SAMPLE_RATE + 160
    assert max(map(len, chunks)) <= 28 * fourier_stt.SAMPLE_RATE
    np.testing.assert_array_equal(np.concatenate(chunks), audio)


def test_cache_requires_tokenizer_and_external_graph_data(tmp_path, monkeypatch):
    monkeypatch.setattr(fourier_stt, "cached_directory", lambda: tmp_path)
    for name in fourier_stt.FILES:
        file = tmp_path / name
        file.parent.mkdir(exist_ok=True, parents=True)
        file.touch()
    assert fourier_stt.is_cached()
    (tmp_path / "encoder.int8.onnx.data").unlink()
    assert not fourier_stt.is_cached()


def test_language_filter():
    assert all(fourier_stt.supports_language(lang) for lang in [None, "auto", "en", "English", "en-GB"])
    assert not fourier_stt.supports_language("de")
