"""GET /models/status and DELETE /models/{name} must recognize models
downloaded via ModelScope into the local get_models_dir()/modelscope/
directory, not just the HuggingFace cache — see
specs/001-modelscope-download-source/data-model.md and contracts/settings-model-source.md.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from backend.backends import ModelConfig
from backend.routes import models as models_routes


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    import backend.config as config

    monkeypatch.setattr(config, "_data_dir", tmp_path)
    # No HuggingFace cache entries anywhere in these tests — force the HF
    # side of the status/delete checks to always come up empty so the
    # ModelScope-directory path is what's actually being exercised.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache-empty"))
    yield


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(models_routes.router)
    return TestClient(app)


KOKORO_CONFIG = ModelConfig(
    model_name="kokoro",
    display_name="Kokoro 82M",
    engine="kokoro",
    hf_repo_id="hexgrad/Kokoro-82M",
    ms_repo_id="AI-ModelScope/Kokoro-82M",
    size_mb=350,
)


def _patch_registry():
    return patch("backend.backends.get_all_model_configs", return_value=[KOKORO_CONFIG])


def _patch_loaded(loaded=False):
    return patch("backend.backends.check_model_loaded", return_value=loaded)


def test_status_reports_downloaded_for_a_modelscope_local_dir(tmp_path, client):
    local_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text("{}")
    (local_dir / "kokoro-v1_0.pth").write_bytes(b"x" * 1024 * 1024)  # 1 MB

    with _patch_registry(), _patch_loaded(False):
        response = client.get("/models/status")

    assert response.status_code == 200
    body = response.json()
    kokoro = next(m for m in body["models"] if m["model_name"] == "kokoro")
    assert kokoro["downloaded"] is True
    assert kokoro["size_mb"] is not None and kokoro["size_mb"] > 0


def test_status_reports_not_downloaded_when_neither_location_has_it(client):
    with _patch_registry(), _patch_loaded(False):
        response = client.get("/models/status")

    kokoro = next(m for m in response.json()["models"] if m["model_name"] == "kokoro")
    assert kokoro["downloaded"] is False


def test_delete_removes_a_modelscope_local_dir(tmp_path, client):
    local_dir = tmp_path / "models" / "modelscope" / "AI-ModelScope--Kokoro-82M"
    local_dir.mkdir(parents=True)
    (local_dir / "kokoro-v1_0.pth").write_bytes(b"x")

    with (
        _patch_registry(),
        patch("backend.backends.get_model_config", return_value=KOKORO_CONFIG),
        patch("backend.backends.unload_model_by_config", return_value=False),
    ):
        response = client.delete("/models/kokoro")

    assert response.status_code == 200
    assert not local_dir.exists()


def test_delete_returns_404_when_model_not_found_anywhere(client):
    with (
        _patch_registry(),
        patch("backend.backends.get_model_config", return_value=KOKORO_CONFIG),
        patch("backend.backends.unload_model_by_config", return_value=False),
    ):
        response = client.delete("/models/kokoro")

    assert response.status_code == 404
