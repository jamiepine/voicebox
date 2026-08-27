"""
Registry/factory tests for the multi-engine LLM subsystem.

Before this feature, `qwen_llm` was the only LLM engine, so the three
engine-specific blocks in unload_model_by_config/check_model_loaded/
get_model_load_func hardcoded the literal string "qwen_llm". These tests
prove the generalized version (branching on membership in LLM_ENGINES)
still works for the pre-existing engine and now also works for the new
minicpm_llm engine, without hardcoding "minicpm_llm" as a new special case.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")

from backend import backends as backends_module
from backend.backends import (
    LLM_ENGINES,
    ModelConfig,
    check_model_loaded,
    get_llm_backend_for_engine,
    get_llm_model_configs,
    get_model_load_func,
    reset_backends,
    unload_model_by_config,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_backends()
    yield
    reset_backends()


def test_llm_engines_includes_both_families():
    assert LLM_ENGINES["qwen_llm"] == "Qwen3 LLM"
    assert LLM_ENGINES["minicpm_llm"] == "MiniCPM5 LLM"


def test_get_llm_model_configs_includes_minicpm5_1b():
    configs = get_llm_model_configs()
    minicpm = [c for c in configs if c.model_name == "minicpm5-1b"]
    assert len(minicpm) == 1
    assert minicpm[0].engine == "minicpm_llm"
    assert minicpm[0].model_size == "1B"

    qwen_names = {c.model_name for c in configs if c.engine == "qwen_llm"}
    assert qwen_names == {"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"}


def test_get_llm_backend_for_engine_minicpm_selects_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(backends_module, "get_backend_type", lambda: "mlx")
    from backend.backends.minicpm_llm_backend import MLXMiniCPMLLMBackend

    backend = get_llm_backend_for_engine("minicpm_llm")
    assert isinstance(backend, MLXMiniCPMLLMBackend)


def test_get_llm_backend_for_engine_minicpm_selects_pytorch_elsewhere(monkeypatch):
    monkeypatch.setattr(backends_module, "get_backend_type", lambda: "pytorch")
    from backend.backends.minicpm_llm_backend import PyTorchMiniCPMLLMBackend

    backend = get_llm_backend_for_engine("minicpm_llm")
    assert isinstance(backend, PyTorchMiniCPMLLMBackend)


def _minicpm_config() -> ModelConfig:
    return next(c for c in get_llm_model_configs() if c.model_name == "minicpm5-1b")


def _fake_backend_per_engine() -> dict:
    """One distinguishable mock per engine, so a test that gets routed to the
    wrong engine's backend fails instead of silently passing against a
    look-alike mock."""
    qwen_backend = MagicMock(name="qwen_backend")
    qwen_backend.is_loaded.return_value = False
    minicpm_backend = MagicMock(name="minicpm_backend")
    minicpm_backend.is_loaded.return_value = True
    minicpm_backend._current_model_size = "1B"
    return {"qwen_llm": qwen_backend, "minicpm_llm": minicpm_backend}


def test_check_model_loaded_works_for_minicpm_engine(monkeypatch):
    engines = _fake_backend_per_engine()
    monkeypatch.setattr(backends_module, "get_llm_backend_for_engine", lambda engine: engines[engine])

    assert check_model_loaded(_minicpm_config()) is True
    engines["minicpm_llm"].is_loaded.assert_called()
    engines["qwen_llm"].is_loaded.assert_not_called()


def test_check_model_loaded_does_not_evict_a_different_active_engine(monkeypatch):
    """Regression test: check_model_loaded is a status query and must stay
    read-only. It must not evict whichever *other* LLM engine is currently
    loaded just to answer a question about a different engine — that used to
    happen because it called llm_service.get_llm_model (which unloads every
    other engine as a side effect) instead of the pure lookup
    get_llm_backend_for_engine.
    """
    engines = _fake_backend_per_engine()
    monkeypatch.setattr(backends_module, "get_llm_backend_for_engine", lambda engine: engines[engine])

    qwen_config = next(c for c in get_llm_model_configs() if c.engine == "qwen_llm")
    check_model_loaded(qwen_config)

    assert engines["minicpm_llm"].is_loaded() is True
    engines["minicpm_llm"].unload_model.assert_not_called()


def test_unload_model_by_config_works_for_minicpm_engine(monkeypatch):
    engines = _fake_backend_per_engine()
    monkeypatch.setattr(backends_module, "get_llm_backend_for_engine", lambda engine: engines[engine])

    assert unload_model_by_config(_minicpm_config()) is True
    engines["minicpm_llm"].unload_model.assert_called_once()
    engines["qwen_llm"].unload_model.assert_not_called()


def test_unload_model_by_config_does_not_touch_a_different_active_engine(monkeypatch):
    """Regression test: unloading Qwen3 by config must not reach into
    minicpm_llm's backend, even though it's the currently active engine."""
    engines = _fake_backend_per_engine()
    monkeypatch.setattr(backends_module, "get_llm_backend_for_engine", lambda engine: engines[engine])

    qwen_config = next(c for c in get_llm_model_configs() if c.engine == "qwen_llm")
    unload_model_by_config(qwen_config)

    assert engines["minicpm_llm"].is_loaded() is True
    engines["minicpm_llm"].unload_model.assert_not_called()


def test_get_model_load_func_works_for_minicpm_engine(monkeypatch):
    fake_backend = MagicMock(name="minicpm_backend")
    other_backend = MagicMock(name="qwen_backend")
    from backend.services import llm as llm_service_module

    engines = {"qwen_llm": other_backend, "minicpm_llm": fake_backend}
    monkeypatch.setattr(llm_service_module, "get_llm_backend_for_engine", lambda engine: engines[engine])

    load_func = get_model_load_func(_minicpm_config())
    load_func()

    fake_backend.load_model.assert_called_once_with("1B")
    other_backend.load_model.assert_not_called()
