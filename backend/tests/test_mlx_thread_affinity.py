"""Regression tests for MLX single-thread serialization.

MLX's Metal stream is thread-local, so every load/generate/unload must run on
one dedicated worker thread (issue #699), and a load+infer pair must run as one
atomic job so a concurrent unload or different-size load can't land between the
load and the inference that reads the model.

These drive the real async orchestration on ``MLXQwenLLMBackend`` with the
heavy mlx-lm calls faked, so they exercise the shipped code paths without
needing MLX installed.
"""

import asyncio
import threading
import time

import pytest

from backend.backends.qwen_llm_backend import MLXQwenLLMBackend
from backend.services import llm as llm_service
from backend.services.mlx_thread import run_on_mlx_thread


@pytest.mark.asyncio
async def test_run_on_mlx_thread_uses_a_single_worker():
    idents = set()

    def record():
        idents.add(threading.get_ident())

    await asyncio.gather(*(run_on_mlx_thread(record) for _ in range(12)))

    assert len(idents) == 1, "MLX work must stay pinned to one worker thread"
    assert idents.pop() != threading.get_ident(), "MLX work must not run on the event loop thread"


def _install_fakes(backend, worker_threads):
    """Replace the heavy sync internals with fakes that record their thread.

    ``_load_model_sync`` and ``_generate_sync`` sleep briefly so that, if the
    load and inference of one request were ever split into separate jobs, a
    second request could interleave and be observed.
    """

    def fake_load(model_size):
        worker_threads.add(threading.get_ident())
        time.sleep(0.02)
        backend.model = {"size": model_size}
        backend._current_model_size = model_size
        backend.model_size = model_size

    def fake_unload():
        worker_threads.add(threading.get_ident())
        backend.model = None
        backend._current_model_size = None

    def fake_generate(prompt, system, max_tokens, temperature, examples=None):
        worker_threads.add(threading.get_ident())
        # Capture the resident model, do "work", then confirm it wasn't
        # swapped or freed underneath us — that is exactly the interleave the
        # atomic load+infer job is meant to prevent.
        resident = backend.model
        assert resident is not None, "model was freed mid-generation"
        time.sleep(0.02)
        assert backend.model is resident, "model was swapped mid-generation"
        return resident["size"]

    backend._load_model_sync = fake_load
    backend.unload_model = fake_unload
    backend._generate_sync = fake_generate


@pytest.mark.asyncio
async def test_concurrent_generate_does_not_cross_models():
    backend = MLXQwenLLMBackend()
    worker_threads = set()
    _install_fakes(backend, worker_threads)

    small, large = await asyncio.gather(
        backend.generate("a", model_size="0.6B"),
        backend.generate("b", model_size="4B"),
    )

    assert small == "0.6B"
    assert large == "4B"
    assert len(worker_threads) == 1, "load and generate must share the one MLX thread"


@pytest.mark.asyncio
async def test_unload_cannot_free_model_mid_generation():
    backend = MLXQwenLLMBackend()
    worker_threads = set()
    _install_fakes(backend, worker_threads)

    await backend.load_model("0.6B")

    # An unload issued while a generation is in flight must serialize behind it
    # on the worker rather than free the model out from under it.
    size, _ = await asyncio.gather(
        backend.generate("a", model_size="0.6B"),
        backend.unload(),
    )

    assert size == "0.6B"
    assert backend.model is None, "unload should still take effect once generation completes"
    assert len(worker_threads) == 1


@pytest.mark.asyncio
async def test_service_path_unload_serializes_with_generation(monkeypatch):
    # The service unload helpers (tts/stt/llm) all route through unload_backend,
    # which must serialize on the MLX worker rather than free the model on the
    # event-loop thread mid-generation.
    backend = MLXQwenLLMBackend()
    worker_threads = set()
    _install_fakes(backend, worker_threads)
    monkeypatch.setattr(llm_service, "get_llm_backend", lambda: backend)

    await backend.load_model("0.6B")

    size, _ = await asyncio.gather(
        backend.generate("a", model_size="0.6B"),
        llm_service.unload_llm_model(),
    )

    assert size == "0.6B"
    assert backend.model is None
    assert len(worker_threads) == 1
