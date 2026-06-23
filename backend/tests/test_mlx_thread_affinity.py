"""Regression tests for MLX worker-thread affinity (issues #675, #699)."""

import asyncio
import threading

import pytest

from backend.backends.mlx_backend import _run_on_mlx_thread


class TestMLXThreadAffinity:
    """MLX model load and inference must stay pinned to one OS thread.

    MLX streams are thread-local and mlx-audio caches one at load time, so a
    model loaded on one thread cannot be used from another -- inference aborts
    with "There is no Stream(gpu, 1) in current thread". These tests lock in the
    single-dedicated-thread guarantee that prevents that.
    """

    @pytest.mark.asyncio
    async def test_all_calls_run_on_one_thread(self):
        """Concurrently dispatched MLX calls all land on the same OS thread.

        Dispatching concurrently (not sequentially) is what makes this prove a
        single worker: with more than one worker the gather would fan out across
        threads and the id set would have more than one entry.
        """
        thread_ids = await asyncio.gather(*[_run_on_mlx_thread(threading.get_ident) for _ in range(25)])

        assert len(set(thread_ids)) == 1, f"MLX work spread across threads: {set(thread_ids)}"

    @pytest.mark.asyncio
    async def test_thread_local_stream_survives_across_calls(self):
        """A stream cached at load time stays visible to later inference calls.

        Mirrors mlx-audio: the stream is created on the load thread and reused
        on every generate. On a different thread the second step would raise,
        reproducing the user-facing error verbatim.
        """
        stream_registry = threading.local()

        def _load_model():
            stream_registry.stream = "Stream(gpu, 1)"

        def _generate():
            stream = getattr(stream_registry, "stream", None)
            if stream is None:
                raise RuntimeError("There is no Stream(gpu, 1) in current thread.")
            return stream

        await _run_on_mlx_thread(_load_model)
        results = [await _run_on_mlx_thread(_generate) for _ in range(10)]

        assert results == ["Stream(gpu, 1)"] * 10
