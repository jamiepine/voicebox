"""Opt-in GPU checks for TADA long text and post-generation cleanup.

The script-style all-model harness owns profile setup. These tests provide a
small, repeatable endpoint contract for a locally prepared profile:

    VOICEBOX_EVAL_PROFILE_ID=<uuid> \
      python -m pytest backend/tests/test_tts_vram_churn.py -m 'gpu and e2e'
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.e2e]

SHORT_TEXT = "This is a short deterministic Voicebox evaluation sentence."
LONG_TEXT = ("This is a public deterministic evaluation sentence. " * 40).strip()


def _profile_id() -> str:
    value = os.environ.get("VOICEBOX_EVAL_PROFILE_ID")
    if not value:
        pytest.skip("set VOICEBOX_EVAL_PROFILE_ID to a locally prepared cloned profile")
    return value


async def _wait_for_generation(client: httpx.AsyncClient, base_url: str, generation_id: str) -> dict:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        response = await client.get(f"{base_url}/generate/{generation_id}/status")
        response.raise_for_status()
        payload = None
        for line in response.text.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                break
        if payload and payload.get("status") in {"completed", "failed"}:
            return payload
        await asyncio.sleep(1)
    pytest.fail(f"generation {generation_id} did not finish within 900 seconds")


async def _generate(
    client: httpx.AsyncClient,
    base_url: str,
    profile_id: str,
    engine: str,
    text: str,
) -> dict:
    response = await client.post(
        f"{base_url}/generate",
        json={
            "profile_id": profile_id,
            "text": text,
            "language": "en",
            "engine": engine,
            "model_size": "1B" if engine == "tada" else "1.7B",
        },
    )
    response.raise_for_status()
    return await _wait_for_generation(client, base_url, response.json()["id"])


async def test_alternating_generations_finish_and_release_vram(live_backend: str, gpu_or_skip, vram_mb):
    profile_id = _profile_id()
    baseline = vram_mb()
    async with httpx.AsyncClient(timeout=900.0) as client:
        for engine in ("tada", "qwen", "tada", "qwen", "tada"):
            result = await _generate(client, live_backend, profile_id, engine, SHORT_TEXT)
            assert result["status"] == "completed", result
            assert vram_mb() - baseline < 500, f"VRAM retained after {engine}: {vram_mb()} MiB"


async def test_tada_long_text_not_truncated(live_backend: str, gpu_or_skip):
    profile_id = _profile_id()
    async with httpx.AsyncClient(timeout=900.0) as client:
        result = await _generate(client, live_backend, profile_id, "tada", LONG_TEXT)
    assert result["status"] == "completed", result
    assert result.get("duration", 0) > 30, result
