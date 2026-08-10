"""Opt-in Whisper quality checks for generated Common Voice fixtures.

Run locally after preparing fixtures and downloading the selected Whisper model:

    python -m pytest backend/tests/test_whisper_long_audio_e2e.py -m 'gpu and e2e'

These tests deliberately do not download data or models themselves.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.e2e]

FIXTURES = Path(__file__).parent / "fixtures" / "generated"
AUDIO_FILES = [
    "cv_10s.wav",
    "cv_30s.wav",
    "cv_6m43s.wav",
    "cv_6m43s.flac",
    "cv_6m43s.mp3",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _require_fixture(name: str) -> Path:
    path = FIXTURES / name
    if not path.is_file():
        pytest.skip(f"missing local evaluation fixture: {path}")
    return path


async def _transcribe(base_url: str, path: Path) -> dict:
    async with httpx.AsyncClient(timeout=900.0) as client:
        with path.open("rb") as audio:
            response = await client.post(
                f"{base_url}/transcribe",
                files={"file": (path.name, audio, "audio/wav")},
                data={"language": "en", "model": "small"},
            )
    if response.status_code == 202:
        pytest.fail("Whisper model is not cached; download it in Voicebox before running GPU evaluation")
    response.raise_for_status()
    return response.json()


@pytest.mark.parametrize("filename", AUDIO_FILES)
async def test_whisper_transcribes_each_duration_and_format(filename: str, live_backend: str, gpu_or_skip):
    path = _require_fixture(filename)
    result = await _transcribe(live_backend, path)

    assert _normalize(result["text"]), f"empty transcription for {filename}"
    if filename == "cv_30s.wav":
        assert 29.0 <= result["duration"] <= 31.0
    if filename.startswith("cv_6m43s"):
        assert result["duration"] >= 400.0
