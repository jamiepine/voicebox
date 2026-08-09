"""
Tests for the ``format`` parameter on the audio-serving endpoints (#869).

Generations are stored as WAV. Callers that want anything else previously had
to transcode themselves; these tests pin the container negotiation, that the
default path is untouched, and that a bad format is rejected rather than
silently served as WAV.

Usage:
    python -m pytest backend/tests/test_audio_format_param.py -v
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

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-audio-format-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _tone(seconds: float, sr: int, freq: float = 440.0, amp: float = 0.3):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def generation(client, tmp_path):
    """A stored WAV generation to serve back in various containers."""
    path = tmp_path / "source.wav"
    sf.write(str(path), _tone(1.0, 48000), 48000)
    with path.open("rb") as fh:
        r = client.post("/generate/import", files={"file": ("source.wav", fh, "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


# ── Default behaviour ────────────────────────────────────────────────


def test_no_format_serves_the_stored_file(client, generation):
    """Existing callers must be untouched: no query param, no transcode."""
    r = client.get(f"/audio/{generation['id']}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/")

    data, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert sr == 48000
    assert len(data) == pytest.approx(48000, rel=0.01)


def test_wav_requested_on_a_wav_file_is_not_re_encoded(client, generation):
    """Same container: hand back the bytes on disk rather than round-tripping
    them through the encoder."""
    plain = client.get(f"/audio/{generation['id']}")
    as_wav = client.get(f"/audio/{generation['id']}", params={"format": "wav"})

    assert as_wav.status_code == 200, as_wav.text
    assert as_wav.content == plain.content


# ── Transcoding ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fmt,mime",
    [
        ("mp3", "audio/mpeg"),
        ("ogg", "audio/ogg"),
        ("opus", "audio/ogg"),
        ("flac", "audio/flac"),
    ],
)
def test_format_returns_that_container(client, generation, fmt, mime):
    r = client.get(f"/audio/{generation['id']}", params={"format": fmt})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith(mime)
    assert f".{fmt}" in r.headers.get("content-disposition", "")

    # Decodable, and still about a second of audio.
    data, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert len(data) / sr == pytest.approx(1.0, abs=0.05)


def test_lossless_transcode_preserves_the_sample_rate(client, generation):
    """FLAC is the format where a resample would be a bug, not a trade-off."""
    r = client.get(f"/audio/{generation['id']}", params={"format": "flac"})
    _, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert sr == 48000


def test_opus_is_served_at_48k(client, tmp_path):
    """Opus only encodes at 48 kHz; a 24 kHz source has to be resampled rather
    than erroring out of libsndfile."""
    path = tmp_path / "narrow.wav"
    sf.write(str(path), _tone(1.0, 24000), 24000)
    with path.open("rb") as fh:
        created = client.post(
            "/generate/import", files={"file": ("narrow.wav", fh, "audio/wav")}
        ).json()

    r = client.get(f"/audio/{created['id']}", params={"format": "opus"})
    assert r.status_code == 200, r.text
    _, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert sr == 48000


# ── Rejections ───────────────────────────────────────────────────────


def test_unsupported_format_is_a_400(client, generation):
    """Not a silent fallback to WAV — a caller asking for m4a should learn that
    it is not on offer."""
    r = client.get(f"/audio/{generation['id']}", params={"format": "m4a"})
    assert r.status_code == 400
    assert "m4a" in r.json()["detail"]


def test_missing_generation_still_404s_with_a_format(client):
    r = client.get("/audio/does-not-exist", params={"format": "mp3"})
    assert r.status_code == 404
