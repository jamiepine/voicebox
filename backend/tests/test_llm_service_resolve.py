"""
services/llm.py's resolve_backend_and_size must return the *model_name*
alongside the bare size — not just the bare size — because callers persist
it as capture attribution (backend/services/captures.py: row.llm_model).
A bare size like "1B" is ambiguous once more than one engine exists; only
the model_name ("minicpm5-1b") uniquely identifies which model ran.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")

from backend.services import llm as llm_service


@pytest.fixture(autouse=True)
def _reset():
    from backend import backends as backends_module

    backends_module.reset_backends()
    yield
    backends_module.reset_backends()


def test_resolve_with_explicit_model_name_returns_engine_bare_size_and_name(monkeypatch):
    fake_backend = MagicMock()
    monkeypatch.setattr(
        llm_service, "get_llm_backend_for_engine", lambda engine: fake_backend
    )

    backend, bare_size, model_name = llm_service.resolve_backend_and_size("minicpm5-1b")

    assert backend is fake_backend
    assert bare_size == "1B"
    assert model_name == "minicpm5-1b"


def test_resolve_with_none_reverse_resolves_default_backend_to_its_model_name(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.model_size = "0.6B"
    monkeypatch.setattr(
        llm_service, "get_llm_backend_for_engine", lambda engine: fake_backend
    )

    backend, bare_size, model_name = llm_service.resolve_backend_and_size(None)

    assert backend is fake_backend
    assert bare_size == "0.6B"
    assert model_name == "qwen3-0.6b"
