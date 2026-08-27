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


def test_check_model_loaded_works_for_minicpm_engine(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.is_loaded.return_value = True
    fake_backend._current_model_size = "1B"
    from backend.services import llm as llm_service_module

    monkeypatch.setattr(llm_service_module, "get_llm_model", lambda engine="qwen_llm": fake_backend)

    assert check_model_loaded(_minicpm_config()) is True
    fake_backend.is_loaded.assert_called()


def test_unload_model_by_config_works_for_minicpm_engine(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.is_loaded.return_value = True
    fake_backend._current_model_size = "1B"
    from backend.services import llm as llm_service_module

    monkeypatch.setattr(llm_service_module, "get_llm_model", lambda engine="qwen_llm": fake_backend)

    assert unload_model_by_config(_minicpm_config()) is True
    fake_backend.unload_model.assert_called_once()


def test_get_model_load_func_works_for_minicpm_engine(monkeypatch):
    fake_backend = MagicMock()
    from backend.services import llm as llm_service_module

    monkeypatch.setattr(llm_service_module, "get_llm_model", lambda engine="qwen_llm": fake_backend)

    load_func = get_model_load_func(_minicpm_config())
    load_func()

    fake_backend.load_model.assert_called_once_with("1B")
