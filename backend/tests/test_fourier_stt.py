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
def test_explicit_download_finishes_ui_progress(tmp_path, monkeypatch, can_download):
    import asyncio

    import huggingface_hub.constants as hf_constants

    from backend import backends, models
    from backend.routes import models as routes
    from backend.utils.download_state import is_download_in_progress
    from backend.utils.progress import ProgressManager
    from backend.utils.tasks import TaskManager

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
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
    assert is_download_in_progress(fourier_stt.REPO) is False
    if can_download:
        assert not progress.get_all_active()
    else:
        assert state["error"] == "download interrupted"


def test_download_includes_root_config_for_hf_attribution(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    import huggingface_hub
    from huggingface_hub.utils import filter_repo_objects

    repo_root = tmp_path / "voicebox" / fourier_stt.MODEL / fourier_stt.REVISION
    onnx_root = repo_root / fourier_stt.SUBDIR
    monkeypatch.setattr(fourier_stt, "cached_directory", lambda: onnx_root)

    download_calls = []

    def fake_snapshot_download(repo_id, *, revision, local_dir, allow_patterns):
        download_calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": local_dir,
                "allow_patterns": list(allow_patterns),
            }
        )
        return str(local_dir)

    recognizer_calls = []

    class FakeOfflineRecognizer:
        @staticmethod
        def from_qwen3_asr(**kwargs):
            recognizer_calls.append(kwargs)
            return object()

    fake_sherpa = ModuleType("sherpa_onnx")
    fake_sherpa.OfflineRecognizer = FakeOfflineRecognizer
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    backend = fourier_stt.FourierSTTBackend()
    backend._load()

    assert len(download_calls) == 1
    call = download_calls[0]
    assert call["repo_id"] == "oruk/oruk-fourier-2b"
    assert call["revision"] == fourier_stt.REVISION
    assert call["local_dir"] == repo_root
    assert "config.json" in call["allow_patterns"]
    assert call["allow_patterns"] == fourier_stt.download_patterns()
    assert len(recognizer_calls) == 1

    # Verify HF Hub's allow_patterns matcher selects root config.json + the 10
    # pinned ONNX/tokenizer files and nothing else in oruk/oruk-fourier-2b.
    repo_files = [
        "README.md",
        "config.json",
        "model.safetensors",
        "Fourier-2B-Q8_0.gguf",
        "mlx/8bit/config.json",
        "onnx/sherpa-int8/README.md",
        "onnx/sherpa-int8/SHA256SUMS",
        *(f"{fourier_stt.SUBDIR}/{name}" for name in fourier_stt.FILES),
    ]
    matched = list(filter_repo_objects(repo_files, allow_patterns=call["allow_patterns"]))
    assert matched == ["config.json", *(f"{fourier_stt.SUBDIR}/{name}" for name in fourier_stt.FILES)]


def test_load_stt_model_and_force_reload_for_fourier(tmp_path, monkeypatch):
    import asyncio

    import huggingface_hub.constants as hf_constants

    from backend import backends
    from backend.utils.download_state import is_download_in_progress

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    fourier_cfg = backends.get_model_config(fourier_stt.MODEL)
    assert fourier_cfg is not None
    assert fourier_cfg.engine == "fourier"

    state = {"loaded": False, "loads": 0, "unloads": 0}

    class FakeFourier:
        model_size = fourier_stt.MODEL

        def is_loaded(self):
            return state["loaded"]

        async def load_model(self, model_size=None):
            assert model_size == fourier_stt.MODEL
            state["loads"] += 1
            state["loaded"] = True

        def unload_model(self):
            state["unloads"] += 1
            state["loaded"] = False

    fake = FakeFourier()
    monkeypatch.setattr("backend.services.transcribe.get_transcription_model", lambda size=None: fake)

    asyncio.run(backends.load_stt_model(fake, fourier_stt.MODEL))
    assert state == {"loaded": True, "loads": 1, "unloads": 0}
    assert is_download_in_progress(fourier_stt.REPO) is False

    # Forced download while resident but missing on-disk files unloads and reloads.
    asyncio.run(backends.track_model_download(fourier_cfg, lambda: fake.load_model(fourier_stt.MODEL), force=True))
    assert state == {"loaded": True, "loads": 2, "unloads": 1}
    assert is_download_in_progress(fourier_stt.REPO) is False


def test_models_status_and_delete_for_fourier(tmp_path, monkeypatch):
    import huggingface_hub.constants as hf_constants
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from backend import backends
    from backend.routes.models import router as models_router

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    fourier_cfg = backends.get_model_config(fourier_stt.MODEL)
    assert fourier_cfg is not None
    monkeypatch.setattr(backends, "get_all_model_configs", lambda: [fourier_cfg])
    monkeypatch.setattr(backends, "check_model_loaded", lambda config: False)
    monkeypatch.setattr(backends, "unload_model_by_config", lambda config: False)

    app = FastAPI()
    app.include_router(models_router)

    with TestClient(app) as client:
        status = client.get("/models/status").json()["models"][0]
        assert status["model_name"] == fourier_stt.MODEL
        assert status["downloaded"] is False
        assert status["size_mb"] is None

        root = fourier_stt.cached_directory()
        (root.parent.parent / fourier_stt.ROOT_CONFIG).parent.mkdir(parents=True, exist_ok=True)
        (root.parent.parent / fourier_stt.ROOT_CONFIG).write_text("{}")
        for name in fourier_stt.FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        status = client.get("/models/status").json()["models"][0]
        assert status["downloaded"] is True
        assert status["size_mb"] is not None
        assert status["size_mb"] > 2200

        resp = client.delete(f"/models/{fourier_stt.MODEL}")
        assert resp.status_code == 200
        assert not root.parent.parent.exists()
        assert client.get("/models/status").json()["models"][0]["downloaded"] is False


def test_transcribe_route_validates_language_and_loads_fourier(tmp_path, monkeypatch):
    import asyncio
    import io

    import huggingface_hub.constants as hf_constants
    import soundfile as sf
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from backend.routes import transcription as route
    from backend.services import transcribe as transcribe_service
    from backend.utils.download_state import is_download_in_progress
    from backend.utils.tasks import TaskManager

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))

    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    wav_bytes = buf.getvalue()

    calls = {"loaded": 0, "transcribed": 0}

    class FakeFourier:
        model_size = fourier_stt.MODEL

        def __init__(self, cached: bool):
            self._cached = cached
            self._loaded = False

        def is_loaded(self):
            return self._loaded

        def _is_model_cached(self, model_size):
            return model_size == fourier_stt.MODEL and self._cached

        async def load_model(self, model_size=None):
            calls["loaded"] += 1
            self._loaded = True
            self._cached = True

        async def transcribe(self, audio_path, language=None, model_size=None):
            calls["transcribed"] += 1
            return "hello world"

    fake = FakeFourier(cached=False)
    tasks = TaskManager()
    pending = []
    monkeypatch.setattr(transcribe_service, "get_transcription_model", lambda size=None: fake)
    monkeypatch.setattr(route, "get_task_manager", lambda: tasks)
    monkeypatch.setattr(route, "create_background_task", pending.append)

    app = FastAPI()
    app.include_router(route.router)

    with TestClient(app) as client:
        bad = client.post(
            "/transcribe",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
            data={"model": fourier_stt.MODEL, "language": "fr"},
        )
        assert bad.status_code == 400
        assert "English only" in bad.json()["detail"]

        cold = client.post(
            "/transcribe",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
            data={"model": fourier_stt.MODEL},
        )
        assert cold.status_code == 202
        assert cold.json()["detail"]["model_name"] == fourier_stt.MODEL
        assert len(pending) == 1
        asyncio.run(pending.pop())
        assert calls["loaded"] == 1
        assert is_download_in_progress(fourier_stt.REPO) is False

        warm = client.post(
            "/transcribe",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
            data={"model": fourier_stt.MODEL},
        )
        assert warm.status_code == 200
        assert warm.json()["text"] == "hello world"
        assert calls["transcribed"] == 1


def test_mcp_transcribe_file_and_models_load_dispatch_fourier(tmp_path, monkeypatch):
    import asyncio

    import huggingface_hub.constants as hf_constants
    import soundfile as sf
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from backend.mcp_server import tools as mcp_tools
    from backend.routes.models import router as models_router
    from backend.services import transcribe as transcribe_service
    from backend.utils.download_state import is_download_in_progress

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    wav_path = tmp_path / "sample.wav"
    sf.write(wav_path, np.zeros(1600, dtype=np.float32), 16000)

    calls = {"loaded": 0, "transcribed": 0}

    class FakeFourier:
        model_size = fourier_stt.MODEL

        def __init__(self):
            self._loaded = False

        def is_loaded(self):
            return self._loaded

        def _is_model_cached(self, model_size):
            return model_size == fourier_stt.MODEL

        async def load_model(self, model_size=None):
            calls["loaded"] += 1
            self._loaded = True

        async def transcribe(self, audio_path, language=None, model_size=None):
            calls["transcribed"] += 1
            return "mcp transcript"

    fake = FakeFourier()
    monkeypatch.setattr(transcribe_service, "get_transcription_model", lambda size=None: fake)

    result = asyncio.run(mcp_tools._transcribe_file(wav_path, "en", fourier_stt.MODEL))
    assert result["text"] == "mcp transcript"
    assert result["model"] == fourier_stt.MODEL
    assert calls == {"loaded": 1, "transcribed": 1}
    assert is_download_in_progress(fourier_stt.REPO) is False

    app = FastAPI()
    app.include_router(models_router)
    with TestClient(app) as client:
        resp = client.post("/models/load", json={"model_name": fourier_stt.MODEL})
        assert resp.status_code == 200
        assert fourier_stt.MODEL in resp.json()["message"]
        assert is_download_in_progress(fourier_stt.REPO) is False
