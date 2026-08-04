"""
Regression tests for POST /models/load and POST /models/unload targeting a
model by name (issue #977).

``/models/load`` only accepted a ``model_size`` query parameter and always
dispatched to the default Qwen TTS backend. A caller following the documented
form — ``POST /models/load {"model_name": "kokoro"}`` — got a 200 back while
the server silently started a 3.6 GB Qwen 1.7B download instead. ``/models/unload``
had the same shape: the documented ``{"model_name": ...}`` body was ignored and
the default Qwen model was unloaded regardless.

Usage:
    python -m pytest backend/tests/test_models_load_by_name.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

# Repo root on sys.path so ``backend`` imports as a package (the routes use
# package-relative imports).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import backends
from backend.routes.models import router as models_router
from backend.services import tts


@pytest.fixture
def loaded(monkeypatch):
    """Record what the route asked the registry to load/unload."""
    calls: dict[str, list] = {"by_config": [], "qwen_size": [], "qwen_unload": 0}

    def fake_load_func(config):
        async def _load():
            calls["by_config"].append(config.model_name)

        return _load

    def fake_unload_by_config(config):
        calls["by_config"].append(f"unload:{config.model_name}")
        return True

    class FakeQwenBackend:
        async def load_model_async(self, model_size):
            calls["qwen_size"].append(model_size)

    def fake_unload_tts_model():
        calls["qwen_unload"] += 1

    monkeypatch.setattr(backends, "get_model_load_func", fake_load_func)
    monkeypatch.setattr(backends, "unload_model_by_config", fake_unload_by_config)
    monkeypatch.setattr(tts, "get_tts_model", lambda: FakeQwenBackend())
    monkeypatch.setattr(tts, "unload_tts_model", fake_unload_tts_model)
    return calls


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(models_router)
    with TestClient(app) as test_client:
        yield test_client


def test_load_by_model_name_targets_that_model(client, loaded):
    """The whole point of #977 — asking for kokoro must not load Qwen."""
    response = client.post("/models/load", json={"model_name": "kokoro"})

    assert response.status_code == 200
    assert loaded["by_config"] == ["kokoro"]
    assert loaded["qwen_size"] == [], "must not touch the default Qwen backend"
    assert "kokoro" in response.json()["message"]


@pytest.mark.parametrize(
    "model_name",
    ["qwen-tts-0.6B", "chatterbox-tts", "whisper-turbo", "qwen3-0.6b"],
)
def test_load_dispatches_across_engine_families(client, loaded, model_name):
    """TTS, STT, and LLM entries all resolve through the same registry."""
    assert client.post("/models/load", json={"model_name": model_name}).status_code == 200
    assert loaded["by_config"] == [model_name]


def test_load_unknown_model_is_rejected(client, loaded):
    """An unknown name is a 400 that names the valid ids, not a silent Qwen load."""
    response = client.post("/models/load", json={"model_name": "not-a-model"})

    assert response.status_code == 400
    assert "kokoro" in response.json()["detail"]
    assert loaded["by_config"] == []
    assert loaded["qwen_size"] == []


def test_load_empty_model_name_is_rejected(client, loaded):
    """A supplied-but-empty name is a bad request, not a request for Qwen.

    Falling back here would reproduce #977 in miniature: a caller that named a
    model (an unbound UI select, say) gets a silent 3.6 GB Qwen download.
    """
    response = client.post("/models/load", json={"model_name": ""})

    assert response.status_code == 400
    assert loaded["by_config"] == []
    assert loaded["qwen_size"] == []


def test_load_without_body_keeps_the_qwen_default(client, loaded):
    """Pre-existing callers post no body at all."""
    assert client.post("/models/load").status_code == 200
    assert loaded["qwen_size"] == ["1.7B"]


def test_load_null_model_name_keeps_the_qwen_default(client, loaded):
    """An explicit null is 'unspecified', unlike an empty string."""
    assert client.post("/models/load", json={"model_name": None}).status_code == 200
    assert loaded["qwen_size"] == ["1.7B"]
    assert loaded["by_config"] == []


def test_load_still_honors_the_model_size_query_param(client, loaded):
    """The legacy ?model_size= form must keep selecting the Qwen variant."""
    assert client.post("/models/load?model_size=0.6B").status_code == 200
    assert loaded["qwen_size"] == ["0.6B"]


def test_load_model_size_in_body_also_works(client, loaded):
    assert client.post("/models/load", json={"model_size": "0.6B"}).status_code == 200
    assert loaded["qwen_size"] == ["0.6B"]


def test_unload_by_model_name_targets_that_model(client, loaded):
    response = client.post("/models/unload", json={"model_name": "chatterbox-tts"})

    assert response.status_code == 200
    assert loaded["by_config"] == ["unload:chatterbox-tts"]
    assert loaded["qwen_unload"] == 0, "must not unload the default Qwen model instead"


def test_unload_empty_model_name_is_rejected(client, loaded):
    response = client.post("/models/unload", json={"model_name": ""})

    assert response.status_code == 400
    assert loaded["by_config"] == []
    assert loaded["qwen_unload"] == 0


def test_unload_without_body_keeps_the_qwen_default(client, loaded):
    assert client.post("/models/unload").status_code == 200
    assert loaded["qwen_unload"] == 1
    assert loaded["by_config"] == []


def test_unload_null_model_name_keeps_the_qwen_default(client, loaded):
    assert client.post("/models/unload", json={"model_name": None}).status_code == 200
    assert loaded["qwen_unload"] == 1
    assert loaded["by_config"] == []


def test_unload_unknown_model_is_rejected(client, loaded):
    response = client.post("/models/unload", json={"model_name": "not-a-model"})

    assert response.status_code == 400
    assert loaded["qwen_unload"] == 0
