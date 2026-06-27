"""Unit tests for the MLX backend async streaming bridge."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("torch") is None:
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.load = lambda *args, **kwargs: None
    torch.save = lambda *args, **kwargs: None
    sys.modules.setdefault("torch", torch)

from backend.backends.mlx_backend import MLXTTSBackend


class CountingMLXBackend(MLXTTSBackend):
    def __init__(self, total_chunks=5000):
        super().__init__()
        self.generated_count = 0
        self.total_chunks = total_chunks

    async def load_model_async(self, model_size=None):
        return None

    def _iter_generate_sync(
        self,
        *,
        text,
        voice_prompt,
        language="en",
        seed=None,
    ):
        for _ in range(self.total_chunks):
            self.generated_count += 1
            yield np.array([0.0], dtype=np.float32), 24000


@pytest.mark.asyncio
async def test_generate_stream_stops_worker_when_consumer_breaks_early():
    backend = CountingMLXBackend()

    async for _chunk, _sample_rate in backend.generate_stream("hello", {}):
        break

    await asyncio.sleep(0.05)

    assert backend.generated_count < backend.total_chunks
