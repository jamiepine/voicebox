"""Regression tests for the post-generation VRAM unload.

Bug: ``run_generation``'s ``finally`` block imported ``unload_all_models``
from ``backend.backends`` — a function that did not exist. The ImportError
was swallowed by a bare ``except Exception``, so models were never unloaded
and chained generations accumulated VRAM until CUDA OOM.

These tests pin both halves of the fix: ``unload_all_models`` exists and
actually unloads every registered backend, and ``run_generation`` invokes
it on the success path, the exception path, and even when it raises.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import backend.backends as backends
import backend.services.generation as generation
from backend.services import history, profiles
from backend.utils import chunked_tts


class _FakeBackend:
    """Minimal loaded-backend stand-in for the registry tests."""

    def __init__(self):
        self.unload_calls = 0

    def is_loaded(self) -> bool:
        return True

    def unload_model(self) -> None:
        self.unload_calls += 1


@pytest.fixture(autouse=True)
def clean_registry():
    backends.reset_backends()
    yield
    backends.reset_backends()


def test_unload_all_models_exists():
    assert callable(backends.unload_all_models)


def test_unload_all_models_unloads_every_registered_backend(monkeypatch):
    tts_a, tts_b = _FakeBackend(), _FakeBackend()
    stt = _FakeBackend()
    llm = _FakeBackend()
    monkeypatch.setattr(backends, "_tts_backends", {"a": tts_a, "b": tts_b})
    monkeypatch.setattr(backends, "_stt_backend", stt)
    monkeypatch.setattr(backends, "_llm_backends", {"qwen_llm": llm})

    backends.unload_all_models()

    assert tts_a.unload_calls == 1
    assert tts_b.unload_calls == 1
    assert stt.unload_calls == 1
    assert llm.unload_calls == 1
    # Registry is empty afterwards — the factories recreate instances lazily.
    assert backends._tts_backends == {}
    assert backends._llm_backends == {}
    assert backends._stt_backend is None


def test_unload_all_models_keeps_unloading_after_one_fails(monkeypatch):
    broken = _FakeBackend()
    broken.unload_model = MagicMock(side_effect=RuntimeError("boom"))
    healthy = _FakeBackend()
    monkeypatch.setattr(backends, "_tts_backends", {"broken": broken, "ok": healthy})

    backends.unload_all_models()

    broken.unload_model.assert_called_once_with()
    assert healthy.unload_calls == 1
    assert backends._tts_backends == {}


def test_unload_all_models_without_torch(monkeypatch):
    # CPU-only / MLX installs have no torch; the guarded import must not raise.
    monkeypatch.setitem(sys.modules, "torch", None)
    backends.unload_all_models()


def test_unload_all_models_clears_cuda_and_mps_caches(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, empty_cache=MagicMock())
    mps = SimpleNamespace(is_available=lambda: True, empty_cache=MagicMock())
    fake_torch = SimpleNamespace(cuda=cuda, mps=mps)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backends.unload_all_models()

    cuda.empty_cache.assert_called_once_with()
    mps.empty_cache.assert_called_once_with()


class _RunGenerationMocks:
    """Bundle of spies/stubs for driving ``run_generation`` without backends."""

    def __init__(self, monkeypatch, *, chunk_error: Exception | None = None):
        self.statuses: list[str] = []

        fake_tts = _FakeBackend()
        monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: fake_tts)

        async def fake_load_engine_model(engine, model_size="default"):
            return None

        monkeypatch.setattr(backends, "load_engine_model", fake_load_engine_model)

        async def fake_voice_prompt(profile_id, db, use_cache=True, engine=None):
            return {}

        monkeypatch.setattr(profiles, "create_voice_prompt_for_profile", fake_voice_prompt)

        async def fake_update_status(generation_id, status, db, **kwargs):
            self.statuses.append(status)

        monkeypatch.setattr(history, "update_generation_status", fake_update_status)

        async def fake_generate_chunked(model, text, voice_prompt, **kwargs):
            if chunk_error is not None:
                raise chunk_error
            return np.zeros(2400, dtype=np.float32), 24000

        monkeypatch.setattr(chunked_tts, "generate_chunked", fake_generate_chunked)

        monkeypatch.setattr(generation, "_save_generate", lambda **kwargs: "clean.wav")
        monkeypatch.setattr(
            generation,
            "get_task_manager",
            lambda: SimpleNamespace(complete_generation=lambda gid: None),
        )
        fake_db = SimpleNamespace(close=lambda: None)
        monkeypatch.setattr(generation, "get_db", lambda: iter([fake_db]))

        self.unload_spy = MagicMock()
        monkeypatch.setattr(backends, "unload_all_models", self.unload_spy)


async def _run(**overrides):
    kwargs = dict(
        generation_id="gen-test",
        profile_id="p1",
        text="hello",
        language="en",
        engine="qwen",
        model_size="1.7B",
        seed=None,
        mode="generate",
    )
    kwargs.update(overrides)
    await generation.run_generation(**kwargs)


async def test_run_generation_unloads_models_on_success(monkeypatch):
    mocks = _RunGenerationMocks(monkeypatch)
    await _run()
    assert "completed" in mocks.statuses
    mocks.unload_spy.assert_called_once_with()


async def test_run_generation_unloads_models_on_exception(monkeypatch):
    mocks = _RunGenerationMocks(monkeypatch, chunk_error=RuntimeError("inference boom"))
    await _run()
    assert "failed" in mocks.statuses
    mocks.unload_spy.assert_called_once_with()


async def test_run_generation_survives_unload_failure(monkeypatch):
    mocks = _RunGenerationMocks(monkeypatch)
    mocks.unload_spy.side_effect = RuntimeError("unload boom")
    # The except in run_generation's finally must log and swallow.
    await _run()
    assert "completed" in mocks.statuses
    mocks.unload_spy.assert_called_once_with()
