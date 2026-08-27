"""
backend/services/refinement.py's refine_transcript takes an optional model
identifier — same resolution requirement as personality.py's compose/rewrite
(see test_personality_engine_resolution.py): it can now name any engine, not
just qwen_llm.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("torch")

from backend.services import refinement
from backend.services.refinement import RefinementFlags


@pytest.fixture(autouse=True)
def _reset():
    from backend import backends as backends_module

    backends_module.reset_backends()
    yield
    backends_module.reset_backends()


def _fake_backend(model_size="0.6B"):
    backend = MagicMock()
    backend.model_size = model_size
    backend.generate = AsyncMock(return_value=" refined text ")
    return backend


@pytest.mark.asyncio
async def test_refine_uses_default_qwen_engine_when_no_model_given(monkeypatch):
    fake_backend = _fake_backend(model_size="0.6B")
    monkeypatch.setattr(
        refinement.llm_service, "get_llm_model", lambda engine="qwen_llm": fake_backend
    )

    text, model_name = await refinement.refine_transcript("hello world", RefinementFlags())

    assert text == "refined text"
    # Reports the model_name, not the bare size — needed for accurate
    # capture attribution once more than one engine exists.
    assert model_name == "qwen3-0.6b"


@pytest.mark.asyncio
async def test_refine_resolves_minicpm_engine_when_model_name_given(monkeypatch):
    calls = []

    def fake_get_llm_model(engine="qwen_llm"):
        calls.append(engine)
        return _fake_backend(model_size="1B")

    monkeypatch.setattr(refinement.llm_service, "get_llm_model", fake_get_llm_model)

    text, model_name = await refinement.refine_transcript(
        "hello world", RefinementFlags(), model_size="minicpm5-1b"
    )

    assert calls == ["minicpm_llm"]
    assert model_name == "minicpm5-1b"
