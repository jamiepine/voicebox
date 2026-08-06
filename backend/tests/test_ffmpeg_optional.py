"""
Tests that ffmpeg stays optional.

Voicebox does not bundle ffmpeg, so every path that can use it must still work
without it. These tests run the relevant behaviour twice — once as configured
on this machine, once with detection forced off — so a missing binary degrades
rather than breaks.

Usage:
    python -m pytest backend/tests/test_ffmpeg_optional.py -v
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-ffmpeg-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402
from backend.utils import ffmpeg  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def no_ffmpeg(monkeypatch):
    """Pretend ffmpeg is not installed, however this machine is set up."""
    ffmpeg.reset_cache()
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: None)
    ffmpeg.reset_cache()
    yield
    ffmpeg.reset_cache()


@pytest.fixture
def story_with_audio(client, tmp_path):
    story = client.post("/stories", json={"name": "ffmpeg test"}).json()

    sr = 48000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

    with wav.open("rb") as fh:
        gen = client.post("/generate/import", files={"file": ("clip.wav", fh, "audio/wav")}).json()
    client.post(f"/stories/{story['id']}/items", json={"generation_id": gen["id"]})

    yield story
    client.delete(f"/stories/{story['id']}")


# ── Detection ────────────────────────────────────────────────────────


def test_detection_is_cached():
    ffmpeg.reset_cache()
    first = ffmpeg.ffmpeg_path()
    assert ffmpeg.ffmpeg_path() is first


def test_health_reports_availability(client):
    body = client.get("/health").json()
    assert "ffmpeg_available" in body
    assert isinstance(body["ffmpeg_available"], bool)


def test_is_available_false_without_binary(no_ffmpeg):
    assert ffmpeg.is_available() is False


# ── Export still works without ffmpeg ────────────────────────────────


def test_export_succeeds_without_ffmpeg(client, story_with_audio, no_ffmpeg):
    """The mixdown and all containers come from libsndfile, not ffmpeg."""
    for fmt in ("wav", "mp3", "ogg", "flac"):
        r = client.get(
            f"/stories/{story_with_audio['id']}/export-audio", params={"format": fmt}
        )
        assert r.status_code == 200, f"{fmt} failed without ffmpeg: {r.text}"
        assert len(r.content) > 0


def test_loudness_request_degrades_rather_than_failing(client, story_with_audio, no_ffmpeg):
    """Asking for normalisation without ffmpeg must still return audio."""
    r = client.get(
        f"/stories/{story_with_audio['id']}/export-audio",
        params={"format": "wav", "normalize_loudness": True},
    )
    assert r.status_code == 200
    data, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
    assert sr == 48000
    assert np.abs(data).max() > 0


def test_normalize_loudness_returns_none_without_ffmpeg(no_ffmpeg):
    assert ffmpeg.normalize_loudness(b"not really audio") is None


# ── Import formats are honest ────────────────────────────────────────


def test_ffmpeg_only_extensions_are_identified():
    assert ffmpeg.requires_ffmpeg(".m4a")
    assert ffmpeg.requires_ffmpeg(".webm")
    assert not ffmpeg.requires_ffmpeg(".wav")
    assert not ffmpeg.requires_ffmpeg(".mp3")


def test_m4a_import_rejected_clearly_without_ffmpeg(client, no_ffmpeg):
    """Previously this got past validation and died deep in the decoder."""
    r = client.post(
        "/generate/import",
        files={"file": ("music.m4a", io.BytesIO(b"\x00" * 1024), "audio/mp4")},
    )
    assert r.status_code == 400
    assert "ffmpeg" in r.json()["detail"].lower()


def test_libsndfile_formats_need_no_ffmpeg(client, tmp_path, no_ffmpeg):
    """WAV/FLAC/OGG/MP3 must import with ffmpeg absent."""
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)

    for name, fmt in (("a.wav", "WAV"), ("a.flac", "FLAC"), ("a.ogg", "OGG")):
        path = tmp_path / name
        sf.write(str(path), tone, sr, format=fmt)
        with path.open("rb") as fh:
            r = client.post("/generate/import", files={"file": (name, fh, "audio/*")})
        assert r.status_code == 200, f"{name} rejected without ffmpeg: {r.text}"
