"""Orukeet's opt-in download, input validation and request boundaries."""

import asyncio
import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import huggingface_hub
import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.backends import orukeet_backend as backend
from backend.routes.transcription import router
from backend.services import transcribe


@pytest.fixture
def model_directory(tmp_path, monkeypatch):
    """Build a small cache with matching hashes and the required license files."""
    directory = tmp_path / backend.SUBFOLDER
    directory.mkdir(parents=True)
    hashes = {}
    for filename in (*backend.FILE_HASHES, *backend.LICENSE_FILES):
        data = filename.encode()
        (directory / filename).write_bytes(data)
        if filename in backend.FILE_HASHES:
            hashes[filename] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(backend, "FILE_HASHES", hashes)
    return directory


def test_complete_cache_does_not_use_network(monkeypatch, model_directory):
    """A complete pinned cache must remain usable without a download request."""
    lookup = Mock(side_effect=lambda _repo, filename, **_kwargs: str(model_directory / Path(filename).name))
    download = Mock(side_effect=AssertionError("Unexpected network access"))
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lookup)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert backend.get_model_directory() == model_directory
    assert all(call.kwargs["revision"] == backend.REVISION for call in lookup.call_args_list)
    download.assert_not_called()


def test_download_is_pinned_and_includes_config_and_notices(monkeypatch, model_directory):
    """A fresh download includes runtime configuration and licenses at one revision."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", Mock(return_value=None))
    download = Mock(return_value=str(model_directory.parents[1]))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert backend.get_model_directory() == model_directory
    assert download.call_args.kwargs["revision"] == backend.REVISION
    assert download.call_args.kwargs["repo_id"] == backend.REPO_ID
    files = download.call_args.kwargs["allow_patterns"]
    assert f"{backend.SUBFOLDER}/config.json" in files
    assert all(f"{backend.SUBFOLDER}/{name}" in files for name in backend.LICENSE_FILES)


def test_corrupted_model_fails_before_inference(monkeypatch, model_directory):
    """Reject altered cached weights before they reach the native recognizer."""
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda _repo, filename, **_kwargs: str(model_directory / Path(filename).name),
    )
    (model_directory / "encoder-model.int8.onnx").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        backend.get_model_directory()


def test_download_failure_propagates(monkeypatch):
    """Preserve the download error when required files are unavailable."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", Mock(return_value=None))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", Mock(side_effect=OSError("offline")))
    with pytest.raises(OSError, match="offline"):
        backend.get_model_directory()


@pytest.mark.asyncio
async def test_empty_audio_does_not_load_model():
    """Empty input returns empty text without initializing the runtime."""
    assert await backend.OrukeetSTTBackend().transcribe(np.empty(0), 16000) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("audio", "sample_rate"), [(np.array([np.nan]), 16000), (np.zeros((2, 2)), 16000), (np.zeros(10), 0)]
)
async def test_invalid_audio_is_rejected(audio, sample_rate):
    """Reject nonfinite samples, multichannel arrays, and invalid sample rates."""
    with pytest.raises(ValueError, match="finite mono audio"):
        await backend.OrukeetSTTBackend().transcribe(audio, sample_rate)


@pytest.mark.asyncio
async def test_unsupported_language_is_rejected():
    """Report an unsupported language before loading a model."""
    with pytest.raises(ValueError, match="does not support language 'ja'"):
        await backend.OrukeetSTTBackend().transcribe(np.zeros(16000), 16000, "ja")


def test_concurrent_first_requests_load_once(monkeypatch):
    """Concurrent first requests share one lazily initialized CPU recognizer."""
    onnx_asr = pytest.importorskip("onnx_asr")

    model = Mock(recognize=Mock(return_value="Hello."))
    load = Mock(return_value=model)
    monkeypatch.setattr(onnx_asr, "load_model", load)
    monkeypatch.setattr(backend, "get_model_directory", lambda: Path("cached"))
    instance = backend.OrukeetSTTBackend()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: instance._transcribe(np.zeros(16000, dtype=np.float32), 16000), range(4)))
    assert results == ["Hello."] * 4
    load.assert_called_once_with(
        "nemo-conformer-tdt", path=Path("cached"), quantization="int8", providers=["CPUExecutionProvider"]
    )


@pytest.mark.asyncio
async def test_inference_does_not_block_event_loop(monkeypatch):
    """The event loop stays responsive while a worker performs recognition."""
    instance = backend.OrukeetSTTBackend()
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_recognize(_audio, sample_rate):
        """Signal worker entry, then simulate a blocking native recognition call."""
        loop.call_soon_threadsafe(started.set)
        time.sleep(0.1)
        return "Hello."

    instance._model = SimpleNamespace(recognize=slow_recognize)
    task = asyncio.create_task(instance.transcribe(np.zeros(16000), 16000))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not task.done()
    assert await task == "Hello."


@pytest.fixture
def client():
    """Expose the real transcription route through an isolated test application."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def wav_bytes():
    """Encode one second of mono silence for multipart upload tests."""
    stream = io.BytesIO()
    sf.write(stream, np.zeros(16000), 16000, format="WAV")
    return stream.getvalue()


def test_endpoint_selects_orukeet_and_preserves_duration(client, monkeypatch):
    """Explicit Orukeet selection keeps the existing text and duration response."""
    model = SimpleNamespace(transcribe=AsyncMock(return_value="Hello."))
    monkeypatch.setattr(transcribe, "get_orukeet_model", lambda: model)
    monkeypatch.setattr(transcribe, "get_whisper_model", Mock(side_effect=AssertionError("Whisper loaded")))
    response = client.post(
        "/transcribe", files={"file": ("clip.wav", wav_bytes(), "audio/wav")}, data={"model": "orukeet"}
    )
    assert response.status_code == 200
    assert response.json() == {"text": "Hello.", "duration": 1.0}
    model.transcribe.assert_awaited_once()


@pytest.mark.parametrize(
    ("error", "status"),
    [(ImportError("install optional dependencies"), 503), (ValueError("unsupported language"), 400)],
)
def test_endpoint_reports_actionable_errors(client, monkeypatch, error, status):
    """Map missing runtime dependencies and invalid input to useful HTTP errors."""
    model = SimpleNamespace(transcribe=AsyncMock(side_effect=error))
    monkeypatch.setattr(transcribe, "get_orukeet_model", lambda: model)
    response = client.post(
        "/transcribe", files={"file": ("clip.wav", wav_bytes(), "audio/wav")}, data={"model": "orukeet"}
    )
    assert response.status_code == status
    assert response.json()["detail"] == str(error)


def test_omitting_model_uses_whisper(client, monkeypatch):
    """Requests without a model selection continue to use the Whisper backend."""
    model = SimpleNamespace(model_size="base", is_loaded=lambda: True, transcribe=AsyncMock(return_value="Whisper."))
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: model)
    monkeypatch.setattr(transcribe, "get_orukeet_model", Mock(side_effect=AssertionError("Orukeet loaded")))
    response = client.post("/transcribe", files={"file": ("clip.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    assert response.json()["text"] == "Whisper."
