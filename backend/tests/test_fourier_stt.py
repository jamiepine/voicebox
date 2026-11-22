import numpy as np
import pytest

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


def test_capture_readiness_recognizes_complete_fourier_download(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from backend.routes import captures

    model = fourier_stt.MODEL
    settings = SimpleNamespace(
        stt_model=model,
        llm_model=captures.get_llm_model_configs()[0].model_size,
        custom_llm_endpoint=None,
        custom_llm_model=None,
    )
    monkeypatch.setattr(captures.settings_service, "get_capture_settings", lambda db: settings)
    monkeypatch.setattr(captures, "is_model_cached", lambda repo: False)
    monkeypatch.setattr(fourier_stt, "cached_directory", lambda: tmp_path)

    def readiness():
        return asyncio.run(captures.capture_readiness_endpoint(db=None)).stt

    assert readiness().size == model
    assert not readiness().ready
    for name in fourier_stt.FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert readiness().ready
    (tmp_path / "encoder.int8.onnx.data").unlink()
    assert not readiness().ready


def test_qwen_transcript_boundary():
    assert fourier_stt.transcript_text("Algerian language<asr_text>And so, my fellow Americans.") == "And so, my fellow Americans."
    assert fourier_stt.transcript_text(" I speak English. ") == "I speak English."
    assert fourier_stt.transcript_text("language None<asr_text>") == ""


@pytest.mark.parametrize("can_download", [True, False])
def test_explicit_download_finishes_ui_progress(monkeypatch, can_download):
    import asyncio

    from backend import backends, models
    from backend.routes import models as routes
    from backend.utils.progress import ProgressManager
    from backend.utils.tasks import TaskManager

    progress, tasks = ProgressManager(), TaskManager()
    pending = []
    monkeypatch.setattr(routes, "get_progress_manager", lambda: progress)
    monkeypatch.setattr(routes, "get_task_manager", lambda: tasks)
    monkeypatch.setattr(routes, "create_background_task", pending.append)

    async def load():
        if not can_download:
            raise RuntimeError("download interrupted")

    monkeypatch.setattr(backends, "get_model_load_func", lambda config: load)

    async def download():
        await routes.trigger_model_download(models.ModelDownloadRequest(model_name=fourier_stt.MODEL))
        await pending.pop()

    asyncio.run(download())
    state = progress.get_progress(fourier_stt.MODEL)
    assert state["status"] == ("complete" if can_download else "error")
    if can_download:
        assert not progress.get_all_active()
    else:
        assert state["error"] == "download interrupted"
