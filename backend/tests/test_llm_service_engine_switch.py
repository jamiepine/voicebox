"""
Cross-engine unload guard for backend/services/llm.py.

Before this feature, only one LLM engine (qwen_llm) ever existed, so
switching "which size is loaded" happened inside a single backend
singleton's own load_model() (it unloads itself before loading a new
size). That safety net does not extend across two different engines'
separate singletons in backends._llm_backends — introducing minicpm_llm
without a guard would let both a Qwen3 size and MiniCPM5-1B sit loaded
in memory simultaneously. See specs/001-minicpm5-llm-engine/data-model.md's
"State / relationships" section.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")

from backend.services import llm as llm_service


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    from backend import backends as backends_module

    backends_module.reset_backends()
    yield
    backends_module.reset_backends()


def _patch_backend_for_engine(monkeypatch, engines: dict):
    monkeypatch.setattr(
        llm_service, "get_llm_backend_for_engine", lambda engine: engines[engine]
    )


def test_switching_to_a_new_engine_unloads_every_other_loaded_engine(monkeypatch):
    qwen_backend = MagicMock()
    qwen_backend.is_loaded.return_value = True
    minicpm_backend = MagicMock()
    minicpm_backend.is_loaded.return_value = False

    _patch_backend_for_engine(
        monkeypatch, {"qwen_llm": qwen_backend, "minicpm_llm": minicpm_backend}
    )

    result = llm_service.get_llm_model(engine="minicpm_llm")

    qwen_backend.unload_model.assert_called_once()
    minicpm_backend.unload_model.assert_not_called()
    assert result is minicpm_backend


def test_switching_the_other_direction_also_unloads(monkeypatch):
    qwen_backend = MagicMock()
    qwen_backend.is_loaded.return_value = False
    minicpm_backend = MagicMock()
    minicpm_backend.is_loaded.return_value = True

    _patch_backend_for_engine(
        monkeypatch, {"qwen_llm": qwen_backend, "minicpm_llm": minicpm_backend}
    )

    result = llm_service.get_llm_model(engine="qwen_llm")

    minicpm_backend.unload_model.assert_called_once()
    qwen_backend.unload_model.assert_not_called()
    assert result is qwen_backend


def test_requesting_the_already_active_engine_does_not_unload_anything(monkeypatch):
    qwen_backend = MagicMock()
    qwen_backend.is_loaded.return_value = True
    minicpm_backend = MagicMock()
    minicpm_backend.is_loaded.return_value = False

    _patch_backend_for_engine(
        monkeypatch, {"qwen_llm": qwen_backend, "minicpm_llm": minicpm_backend}
    )

    llm_service.get_llm_model(engine="qwen_llm")

    qwen_backend.unload_model.assert_not_called()


def test_default_engine_is_qwen_llm(monkeypatch):
    qwen_backend = MagicMock()
    qwen_backend.is_loaded.return_value = False
    minicpm_backend = MagicMock()
    minicpm_backend.is_loaded.return_value = False

    _patch_backend_for_engine(
        monkeypatch, {"qwen_llm": qwen_backend, "minicpm_llm": minicpm_backend}
    )

    result = llm_service.get_llm_model()

    assert result is qwen_backend


def test_unload_llm_model_unloads_the_requested_engine(monkeypatch):
    qwen_backend = MagicMock()
    _patch_backend_for_engine(monkeypatch, {"qwen_llm": qwen_backend, "minicpm_llm": MagicMock()})

    llm_service.unload_llm_model(engine="qwen_llm")

    qwen_backend.unload_model.assert_called_once()
