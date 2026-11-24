"""
Regression tests for issue #1261: a partial HuggingFace snapshot was reported
as downloaded, and the forced-offline guard then kept the missing files from
ever being fetched.

Every test runs against a temporary HF cache; nothing touches the network.

Usage:
    python -m pytest backend/tests/test_download_state.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import backends
from backend.backends import ModelConfig, track_model_download
from backend.backends.base import is_model_cached, model_load_progress
from backend.routes.models import router as models_router
from backend.utils.download_state import (
    is_download_in_progress,
    mark_download_started,
    mark_download_verified,
    verified_snapshot_complete,
)

REPO = "org/voice-design"
COMMIT = "0123abcd"
FULL_SNAPSHOT = {
    "model.safetensors": b"weights",
    "config.json": b"{}",
    "tokenizer_config.json": b"{}",
    "speech_tokenizer/model.safetensors": b"tokenizer weights",
    "speech_tokenizer/config.json": b"{}",
}
CONFIG = ModelConfig(model_name="voice-design", display_name="Voice Design", engine="qwen", hf_repo_id=REPO)


@pytest.fixture
def hf_cache(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    return tmp_path


def _write_snapshot(cache: Path, files: dict[str, bytes]) -> Path:
    """Lay files out the way huggingface_hub does: blobs plus snapshot symlinks."""
    repo_dir = cache / ("models--" + REPO.replace("/", "--"))
    blobs = repo_dir / "blobs"
    snapshot = repo_dir / "snapshots" / COMMIT
    blobs.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    (repo_dir / "refs" / "main").write_text(COMMIT)
    for name, content in files.items():
        blob = blobs / f"blob-{abs(hash(name))}"
        blob.write_bytes(content)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
    return snapshot


def _hf_offline() -> bool:
    import huggingface_hub.constants as hf_constants

    return hf_constants.HF_HUB_OFFLINE


def test_verified_snapshot_with_every_file_is_installed(hf_cache):
    _write_snapshot(hf_cache, FULL_SNAPSHOT)
    assert mark_download_verified(REPO) is True

    assert verified_snapshot_complete(REPO) is True
    assert is_model_cached(REPO) is True


def test_verified_snapshot_missing_a_recorded_file_is_not_installed(hf_cache):
    snapshot = _write_snapshot(hf_cache, FULL_SNAPSHOT)
    mark_download_verified(REPO)

    (snapshot / "speech_tokenizer" / "model.safetensors").unlink()

    assert verified_snapshot_complete(REPO) is False
    assert is_model_cached(REPO) is False


def test_dangling_snapshot_symlink_counts_as_missing(hf_cache):
    snapshot = _write_snapshot(hf_cache, FULL_SNAPSHOT)
    mark_download_verified(REPO)

    (snapshot / "config.json").resolve().unlink()

    assert is_model_cached(REPO) is False


def test_unrecorded_legacy_download_falls_back_to_weights_heuristic(hf_cache):
    _write_snapshot(hf_cache, {"model.safetensors": b"weights"})

    assert verified_snapshot_complete(REPO) is None
    assert is_model_cached(REPO) is True


def test_marker_present_means_not_installed_until_verified(hf_cache):
    _write_snapshot(hf_cache, FULL_SNAPSHOT)
    mark_download_started(REPO)

    assert is_download_in_progress(REPO) is True
    assert is_model_cached(REPO) is False

    assert mark_download_verified(REPO) is True
    assert is_download_in_progress(REPO) is False
    assert is_model_cached(REPO) is True


def test_verification_without_a_snapshot_keeps_the_marker(hf_cache):
    mark_download_started(REPO)

    assert mark_download_verified(REPO) is False
    assert is_download_in_progress(REPO) is True


@pytest.mark.asyncio
async def test_failed_download_leaves_marker_and_model_not_installed(hf_cache):
    async def load_dies_after_main_weights():
        _write_snapshot(hf_cache, {"model.safetensors": b"weights"})
        raise OSError("connection reset")

    with pytest.raises(OSError, match="connection reset"):
        await track_model_download(CONFIG, load_dies_after_main_weights, force=True)

    assert is_download_in_progress(REPO) is True
    assert is_model_cached(REPO) is False


@pytest.mark.asyncio
async def test_successful_download_records_snapshot_and_clears_marker(hf_cache):
    async def load():
        assert is_download_in_progress(REPO) is True
        _write_snapshot(hf_cache, FULL_SNAPSHOT)

    await track_model_download(CONFIG, load)

    assert is_download_in_progress(REPO) is False
    assert verified_snapshot_complete(REPO) is True
    assert is_model_cached(REPO) is True


@pytest.mark.asyncio
async def test_redownload_of_partial_snapshot_runs_online_and_recovers(hf_cache):
    """A legacy partial snapshot looks cached; re-triggering download must still fetch."""
    _write_snapshot(hf_cache, {"model.safetensors": b"weights"})
    assert is_model_cached(REPO) is True

    seen = {}

    async def load():
        is_cached = is_model_cached(REPO)
        with model_load_progress(CONFIG.model_name, is_cached):
            seen["is_cached"] = is_cached
            seen["offline"] = _hf_offline()
            missing = {k: v for k, v in FULL_SNAPSHOT.items() if k != "model.safetensors"}
            _write_snapshot(hf_cache, missing)

    await track_model_download(CONFIG, load, force=True)

    assert seen == {"is_cached": False, "offline": False}
    assert is_download_in_progress(REPO) is False
    assert verified_snapshot_complete(REPO) is True


@pytest.mark.asyncio
async def test_failed_load_of_unrecorded_partial_snapshot_retries_online(hf_cache):
    """An upgrade from 0.6.0 with a partial snapshot: no marker, no record."""
    _write_snapshot(hf_cache, {"model.safetensors": b"weights"})
    attempts = []

    async def load():
        is_cached = is_model_cached(REPO)
        with model_load_progress(CONFIG.model_name, is_cached):
            attempts.append((is_cached, _hf_offline()))
            if is_cached:
                raise OSError("We couldn't connect to 'https://huggingface.co'")
            _write_snapshot(hf_cache, {k: v for k, v in FULL_SNAPSHOT.items() if k != "model.safetensors"})

    await track_model_download(CONFIG, load)

    assert attempts == [(True, True), (False, False)]
    assert is_download_in_progress(REPO) is False
    assert verified_snapshot_complete(REPO) is True


@pytest.mark.asyncio
async def test_failed_load_of_verified_snapshot_is_not_retried(hf_cache):
    _write_snapshot(hf_cache, FULL_SNAPSHOT)
    mark_download_verified(REPO)
    attempts = []

    async def load():
        attempts.append(1)
        raise OSError("disk read error")

    with pytest.raises(OSError, match="disk read error"):
        await track_model_download(CONFIG, load)

    assert attempts == [1]
    assert is_download_in_progress(REPO) is False


@pytest.mark.asyncio
async def test_non_file_error_on_unrecorded_snapshot_is_not_retried(hf_cache):
    _write_snapshot(hf_cache, FULL_SNAPSHOT)
    attempts = []

    async def load():
        attempts.append(1)
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        await track_model_download(CONFIG, load)

    assert attempts == [1]
    assert is_download_in_progress(REPO) is False


@pytest.mark.asyncio
async def test_failed_load_that_left_model_resident_is_not_retried(hf_cache, monkeypatch):
    _write_snapshot(hf_cache, FULL_SNAPSHOT)
    state = {"resident": False, "attempts": 0}
    monkeypatch.setattr(backends, "check_model_loaded", lambda config: state["resident"])

    async def load():
        state["attempts"] += 1
        state["resident"] = True
        raise OSError("device placement failed")

    with pytest.raises(OSError, match="device placement failed"):
        await track_model_download(CONFIG, load)

    assert state["attempts"] == 1
    assert verified_snapshot_complete(REPO) is None


def test_tada_residency_is_size_aware(monkeypatch):
    class FakeTada:
        model_size = "1B"

        def is_loaded(self):
            return True

    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: FakeTada())
    tada = {c.model_size: c for c in backends.get_tts_model_configs() if c.engine == "tada"}

    assert backends.check_model_loaded(tada["1B"]) is True
    assert backends.check_model_loaded(tada["3B"]) is False


@pytest.mark.asyncio
async def test_failed_retry_reraises_original_error_and_keeps_marker(hf_cache):
    _write_snapshot(hf_cache, {"model.safetensors": b"weights"})
    errors = iter([OSError("first failure"), OSError("retry failure")])

    async def load():
        raise next(errors)

    with pytest.raises(OSError, match="first failure"):
        await track_model_download(CONFIG, load)

    assert is_download_in_progress(REPO) is True
    assert is_model_cached(REPO) is False


@pytest.mark.asyncio
async def test_resident_model_does_not_certify_an_incomplete_snapshot(hf_cache, monkeypatch):
    snapshot = _write_snapshot(hf_cache, FULL_SNAPSHOT)
    mark_download_verified(REPO)
    (snapshot / "config.json").unlink()
    monkeypatch.setattr(backends, "check_model_loaded", lambda config: True)

    async def load():
        pass

    await track_model_download(CONFIG, load, force=True)

    assert is_download_in_progress(REPO) is False
    assert verified_snapshot_complete(REPO) is False
    assert is_model_cached(REPO) is False


def test_offline_patch_is_not_forced_while_marker_exists(hf_cache):
    _write_snapshot(hf_cache, FULL_SNAPSHOT)
    mark_download_started(REPO)

    with model_load_progress(CONFIG.model_name, is_model_cached(REPO)):
        assert _hf_offline() is False

    mark_download_verified(REPO)

    with model_load_progress(CONFIG.model_name, is_model_cached(REPO)):
        assert _hf_offline() is True
    assert _hf_offline() is False


def test_status_reports_partial_verified_snapshot_as_not_downloaded(hf_cache, monkeypatch):
    snapshot = _write_snapshot(hf_cache, FULL_SNAPSHOT)
    mark_download_verified(REPO)
    monkeypatch.setattr(backends, "get_all_model_configs", lambda: [CONFIG])
    monkeypatch.setattr(backends, "check_model_loaded", lambda config: False)

    app = FastAPI()
    app.include_router(models_router)
    with TestClient(app) as client:
        assert client.get("/models/status").json()["models"][0]["downloaded"] is True

        (snapshot / "tokenizer_config.json").unlink()
        assert client.get("/models/status").json()["models"][0]["downloaded"] is False

        mark_download_started(REPO)
        (snapshot / "tokenizer_config.json").symlink_to(snapshot / "config.json")
        assert client.get("/models/status").json()["models"][0]["downloaded"] is False
