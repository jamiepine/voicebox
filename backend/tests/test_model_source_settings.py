"""Contract tests for GET/PUT /settings/model-source — see
specs/001-modelscope-download-source/contracts/settings-model-source.md
"""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from backend.routes import settings as settings_routes
from backend.utils import model_source


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    import backend.config as config

    monkeypatch.setattr(config, "_data_dir", tmp_path)
    yield


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings_routes.router)
    return TestClient(app)


def test_get_defaults_to_huggingface(client):
    response = client.get("/settings/model-source")
    assert response.status_code == 200
    assert response.json() == {"source": "huggingface"}


def test_put_persists_a_valid_source(client):
    response = client.put("/settings/model-source", json={"source": "modelscope"})
    assert response.status_code == 200
    assert response.json() == {"source": "modelscope"}
    assert model_source.get_model_source() == "modelscope"


def test_put_rejects_an_invalid_source(client):
    response = client.put("/settings/model-source", json={"source": "bogus"})
    assert response.status_code == 422


def test_put_rejects_the_removed_hf_mirror_source(client):
    response = client.put("/settings/model-source", json={"source": "hf_mirror"})
    assert response.status_code == 422
