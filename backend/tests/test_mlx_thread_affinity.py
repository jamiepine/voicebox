"""Regression tests for MLX worker-thread affinity (issues #675, #699)."""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends.mlx_backend import _mlx_executor, _run_on_mlx_thread  # noqa: E402 -- needs the sys.path shim above


class TestMLXThreadAffinity:
    """MLX model load and inference must stay pinned to one OS thread.

    MLX streams are thread-local and mlx-audio caches one at load time, so a
    model loaded on one thread cannot be used from another -- inference aborts
    with "There is no Stream(gpu, 1) in current thread". These tests lock in the
    single-dedicated-thread guarantee that prevents that.
    """

    def test_executor_is_single_worker(self):
        """A single worker is what guarantees cross-call thread affinity."""
        assert _mlx_executor._max_workers == 1

    @pytest.mark.asyncio
    async def test_all_calls_share_one_thread(self):
        """Load plus many inference calls all run on the same OS thread."""
        thread_ids = [await _run_on_mlx_thread(threading.get_ident) for _ in range(25)]

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
