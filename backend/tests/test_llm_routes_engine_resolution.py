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


def _fake_backend(*, loaded=True, model_size="1B", cached=True, reply="minicpm says hi"):
    backend = MagicMock()
    backend.is_loaded.return_value = loaded
    backend.model_size = model_size
    backend._is_model_cached.return_value = cached
    backend.generate = AsyncMock(return_value=reply)
    return backend


def _patch_engines(monkeypatch, engines: dict):
    """Route get_llm_model to a distinguishable per-engine mock, so a request
    that resolves to the wrong engine drives (and can be asserted against)
    the *other* engine's backend instead of silently reusing one mock."""
    monkeypatch.setattr(llm_routes.llm, "get_llm_model", lambda engine="qwen_llm": engines[engine])


@pytest.mark.asyncio
async def test_generate_resolves_minicpm_engine_and_reports_model_used(monkeypatch):
    minicpm_backend = _fake_backend(loaded=True, model_size="1B", cached=True, reply="minicpm says hi")
    qwen_backend = _fake_backend(loaded=True, model_size="0.6B", cached=True, reply="qwen says hi")
    _patch_engines(monkeypatch, {"qwen_llm": qwen_backend, "minicpm_llm": minicpm_backend})

    request = models.LLMGenerateRequest(prompt="hi", model_size="minicpm5-1b")
    response = await llm_routes.llm_generate(request)

    minicpm_backend.generate.assert_called_once()
    qwen_backend.generate.assert_not_called()
    _, kwargs = minicpm_backend.generate.call_args
    assert kwargs["model_size"] == "1B"
    assert response.model_size == "minicpm5-1b"
    assert response.text == "minicpm says hi"


@pytest.mark.asyncio
async def test_generate_still_works_for_qwen_engine(monkeypatch):
    qwen_backend = _fake_backend(loaded=True, model_size="0.6B", cached=True, reply="qwen says hi")
    minicpm_backend = _fake_backend(loaded=True, model_size="1B", cached=True, reply="minicpm says hi")
    _patch_engines(monkeypatch, {"qwen_llm": qwen_backend, "minicpm_llm": minicpm_backend})

    request = models.LLMGenerateRequest(prompt="hi", model_size="qwen3-0.6b")
    response = await llm_routes.llm_generate(request)

    qwen_backend.generate.assert_called_once()
    minicpm_backend.generate.assert_not_called()
    _, kwargs = qwen_backend.generate.call_args
    assert kwargs["model_size"] == "0.6B"
    assert response.model_size == "qwen3-0.6b"
    assert response.text == "qwen says hi"


@pytest.mark.asyncio
async def test_generate_starts_download_when_minicpm_not_cached(monkeypatch):
    minicpm_backend = _fake_backend(loaded=False, model_size=None, cached=False)
    minicpm_backend.load_model = AsyncMock()
    qwen_backend = _fake_backend(loaded=True, model_size="0.6B", cached=True, reply="qwen says hi")
    _patch_engines(monkeypatch, {"qwen_llm": qwen_backend, "minicpm_llm": minicpm_backend})
    monkeypatch.setattr(llm_routes, "create_background_task", lambda coro: coro.close())

    request = models.LLMGenerateRequest(prompt="hi", model_size="minicpm5-1b")
    response = await llm_routes.llm_generate(request)

    assert response.status_code == 202
    import json

    body = json.loads(response.body)
    assert body["model_name"] == "minicpm5-1b"
    assert body["downloading"] is True
    qwen_backend.generate.assert_not_called()
