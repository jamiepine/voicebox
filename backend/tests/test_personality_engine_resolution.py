"""
backend/services/personality.py's compose_as_profile/rewrite_as_profile take
an optional model identifier that used to always mean a bare Qwen3 size
resolved against the single qwen_llm engine. Now that identifier can be a
model_name naming any engine (e.g. "minicpm5-1b") — these must resolve to
the correct engine's backend, not always qwen_llm's.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("torch")

from backend.services import personality


@pytest.fixture(autouse=True)
def _reset():
    from backend import backends as backends_module

    backends_module.reset_backends()
    yield
    backends_module.reset_backends()


def _fake_backend(model_size="0.6B", reply=" a witty remark "):
    backend = MagicMock()
    backend.model_size = model_size
    backend.generate = AsyncMock(return_value=reply)
    return backend


def _fake_backend_per_engine() -> dict:
    """One distinguishable mock per engine, so a call routed to the wrong
    engine drives (and can be asserted against) the *other* engine's mock
    instead of silently reusing a single look-alike backend."""
    return {
        "qwen_llm": _fake_backend(model_size="0.6B", reply=" a witty remark "),
        "minicpm_llm": _fake_backend(model_size="1B", reply=" a minicpm remark "),
    }


def _patch_engines(monkeypatch, engines: dict):
    monkeypatch.setattr(personality.llm_service, "get_llm_model", lambda engine="qwen_llm": engines[engine])


@pytest.mark.asyncio
async def test_compose_uses_default_qwen_engine_when_no_model_given(monkeypatch):
    engines = _fake_backend_per_engine()
    _patch_engines(monkeypatch, engines)

    result = await personality.compose_as_profile("A grumpy pirate.")

    engines["qwen_llm"].generate.assert_called_once()
    engines["minicpm_llm"].generate.assert_not_called()
    assert result.text == "a witty remark"
    # Reports the model_name, not the bare size — "0.6B" alone wouldn't
    # identify a unique model once more than one engine exists.
    assert result.model_size == "qwen3-0.6b"


@pytest.mark.asyncio
async def test_compose_resolves_minicpm_engine_when_model_name_given(monkeypatch):
    engines = _fake_backend_per_engine()
    _patch_engines(monkeypatch, engines)

    result = await personality.compose_as_profile("A grumpy pirate.", model_size="minicpm5-1b")

    engines["minicpm_llm"].generate.assert_called_once()
    engines["qwen_llm"].generate.assert_not_called()
    assert result.text == "a minicpm remark"
    assert result.model_size == "minicpm5-1b"


@pytest.mark.asyncio
async def test_rewrite_resolves_minicpm_engine_when_model_name_given(monkeypatch):
    engines = _fake_backend_per_engine()
    _patch_engines(monkeypatch, engines)

    result = await personality.rewrite_as_profile(
        "A grumpy pirate.", "hello there", model_size="minicpm5-1b"
    )

    engines["minicpm_llm"].generate.assert_called_once()
    engines["qwen_llm"].generate.assert_not_called()
    assert result.text == "a minicpm remark"
    assert result.model_size == "minicpm5-1b"
