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


def _fake_backend(model_size="0.6B"):
    backend = MagicMock()
    backend.model_size = model_size
    backend.generate = AsyncMock(return_value=" a witty remark ")
    return backend


@pytest.mark.asyncio
async def test_compose_uses_default_qwen_engine_when_no_model_given(monkeypatch):
    fake_backend = _fake_backend(model_size="0.6B")
    monkeypatch.setattr(
        personality.llm_service, "get_llm_model", lambda engine="qwen_llm": fake_backend
    )

    result = await personality.compose_as_profile("A grumpy pirate.")

    assert result.text == "a witty remark"
    # Reports the model_name, not the bare size — "0.6B" alone wouldn't
    # identify a unique model once more than one engine exists.
    assert result.model_size == "qwen3-0.6b"


@pytest.mark.asyncio
async def test_compose_resolves_minicpm_engine_when_model_name_given(monkeypatch):
    calls = []

    def fake_get_llm_model(engine="qwen_llm"):
        calls.append(engine)
        return _fake_backend(model_size="1B")

    monkeypatch.setattr(personality.llm_service, "get_llm_model", fake_get_llm_model)

    result = await personality.compose_as_profile("A grumpy pirate.", model_size="minicpm5-1b")

    assert calls == ["minicpm_llm"]
    assert result.model_size == "minicpm5-1b"


@pytest.mark.asyncio
async def test_rewrite_resolves_minicpm_engine_when_model_name_given(monkeypatch):
    calls = []

    def fake_get_llm_model(engine="qwen_llm"):
        calls.append(engine)
        return _fake_backend(model_size="1B")

    monkeypatch.setattr(personality.llm_service, "get_llm_model", fake_get_llm_model)

    result = await personality.rewrite_as_profile(
        "A grumpy pirate.", "hello there", model_size="minicpm5-1b"
    )

    assert calls == ["minicpm_llm"]
    assert result.model_size == "minicpm5-1b"
