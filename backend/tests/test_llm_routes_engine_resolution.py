"""
backend/routes/llm.py must resolve a model_name identifier (e.g. "minicpm5-1b")
to its (engine, model_size) pair before dispatching, instead of assuming the
qwen_llm engine and a bare Qwen3 size the way it did when only one engine
existed. See specs/001-minicpm5-llm-engine/contracts/llm-model-selection.md.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("torch")

from backend import models
from backend.routes import llm as llm_routes


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    from backend import backends as backends_module

    backends_module.reset_backends()
    yield
    backends_module.reset_backends()


def _fake_backend(*, loaded=True, model_size="1B", cached=True):
    backend = MagicMock()
    backend.is_loaded.return_value = loaded
    backend.model_size = model_size
    backend._is_model_cached.return_value = cached
    backend.generate = AsyncMock(return_value="minicpm says hi")
    return backend


@pytest.mark.asyncio
async def test_generate_resolves_minicpm_engine_and_reports_model_used(monkeypatch):
    fake_backend = _fake_backend(loaded=True, model_size="1B", cached=True)
    monkeypatch.setattr(llm_routes.llm, "get_llm_model", lambda engine="qwen_llm": fake_backend)

    request = models.LLMGenerateRequest(prompt="hi", model_size="minicpm5-1b")
    response = await llm_routes.llm_generate(request)

    fake_backend.generate.assert_called_once()
    _, kwargs = fake_backend.generate.call_args
    assert kwargs["model_size"] == "1B"
    assert response.model_size == "minicpm5-1b"
    assert response.text == "minicpm says hi"


@pytest.mark.asyncio
async def test_generate_still_works_for_qwen_engine(monkeypatch):
    fake_backend = _fake_backend(loaded=True, model_size="0.6B", cached=True)
    monkeypatch.setattr(llm_routes.llm, "get_llm_model", lambda engine="qwen_llm": fake_backend)

    request = models.LLMGenerateRequest(prompt="hi", model_size="qwen3-0.6b")
    response = await llm_routes.llm_generate(request)

    _, kwargs = fake_backend.generate.call_args
    assert kwargs["model_size"] == "0.6B"
    assert response.model_size == "qwen3-0.6b"


@pytest.mark.asyncio
async def test_generate_starts_download_when_minicpm_not_cached(monkeypatch):
    fake_backend = _fake_backend(loaded=False, model_size=None, cached=False)
    fake_backend.load_model = AsyncMock()
    monkeypatch.setattr(llm_routes.llm, "get_llm_model", lambda engine="qwen_llm": fake_backend)
    monkeypatch.setattr(llm_routes, "create_background_task", lambda coro: coro.close())

    request = models.LLMGenerateRequest(prompt="hi", model_size="minicpm5-1b")
    response = await llm_routes.llm_generate(request)

    assert response.status_code == 202
    import json

    body = json.loads(response.body)
    assert body["model_name"] == "minicpm5-1b"
    assert body["downloading"] is True
