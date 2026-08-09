"""
Tests for the story mixdown: stereo/project-rate mixing, fades, speed,
track gain, mute/solo, ducking, and export formats.

The mixer previously flattened everything to 24 kHz mono, which destroyed an
imported music bed (12 kHz Nyquist, stereo image folded flat). These tests pin
the new behaviour and the placement rules that let a bed sit under narration.

Usage:
    python -m pytest backend/tests/test_story_mixdown.py -v
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

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-mixdown-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _tone(seconds: float, sr: int, freq: float, channels: int = 1, amp: float = 0.3):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 1:
        return mono
    # Distinct content per channel so a mono fold-down is detectable.
    right = (amp * np.sin(2 * np.pi * (freq * 1.5) * t)).astype(np.float32)
    return np.stack([mono, right], axis=1)


def _import_audio(client, tmp_path, name, seconds=2.0, sr=48000, freq=440.0, channels=1):
    """Register an audio file as an importable generation."""
    path = tmp_path / name
    sf.write(str(path), _tone(seconds, sr, freq, channels), sr)
    with path.open("rb") as fh:
        r = client.post("/generate/import", files={"file": (name, fh, "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def story(client):
    r = client.post("/stories", json={"name": "Mixdown Test"})
    assert r.status_code == 200, r.text
    created = r.json()
    yield created
    client.delete(f"/stories/{created['id']}")


def _export(client, story_id, fmt=None):
    params = {"format": fmt} if fmt else None
    r = client.get(f"/stories/{story_id}/export-audio", params=params)
    assert r.status_code == 200, r.text
    return r.content


def _decode(raw):
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return data, sr


# ── Placement ────────────────────────────────────────────────────────


def test_imported_audio_lands_on_its_own_lane_at_zero(client, story, tmp_path):
    """The bug this fixes: music used to be appended after the narration on
    track 0 instead of playing underneath it."""
    voice = _import_audio(client, tmp_path, "voice.wav", seconds=2.0)
    client.post(f"/stories/{story['id']}/items", json={"generation_id": voice["id"], "track": 0})

    bed = _import_audio(client, tmp_path, "bed.wav", seconds=2.0, freq=220.0)
    r = client.post(f"/stories/{story['id']}/items", json={"generation_id": bed["id"]})
    assert r.status_code == 200, r.text

    item = r.json()
    assert item["track"] != 0, "bed landed on the voice lane"
    assert item["start_time_ms"] == 0, "bed did not start at the top of the timeline"


def test_explicit_track_zero_is_respected(client, story, tmp_path):
    """track=0 must mean track 0, not 'unspecified' — the reason
    StoryItemCreate.track had to become nullable."""
    clip = _import_audio(client, tmp_path, "explicit.wav")
    r = client.post(
        f"/stories/{story['id']}/items",
        json={"generation_id": clip["id"], "track": 0, "start_time_ms": 500},
    )
    assert r.json()["track"] == 0
    assert r.json()["start_time_ms"] == 500


# ── Project rate and channels ────────────────────────────────────────


def test_mixdown_keeps_48k_stereo(client, story, tmp_path):
    """A 48 kHz stereo bed must survive; it used to come out 24 kHz mono."""
    bed = _import_audio(client, tmp_path, "stereo48.wav", sr=48000, channels=2)
    client.post(f"/stories/{story['id']}/items", json={"generation_id": bed["id"]})

    data, sr = _decode(_export(client, story["id"]))
    assert sr == 48000, f"project rate collapsed to {sr}"
    assert data.shape[1] == 2
    # Channels carry different tones, so a mono fold-down would make them equal.
    assert not np.allclose(data[:, 0], data[:, 1]), "stereo image was folded to mono"


def test_project_rate_is_capped_at_48k(client, story, tmp_path):
    bed = _import_audio(client, tmp_path, "hires.wav", sr=96000)
    client.post(f"/stories/{story['id']}/items", json={"generation_id": bed["id"]})

    _data, sr = _decode(_export(client, story["id"]))
    assert sr == 48000


# ── Fades ────────────────────────────────────────────────────────────


def test_fades_ramp_from_and_to_silence(client, story, tmp_path):
    clip = _import_audio(client, tmp_path, "fade.wav", seconds=2.0, sr=48000)
    item = client.post(
        f"/stories/{story['id']}/items", json={"generation_id": clip["id"]}
    ).json()

    r = client.put(
        f"/stories/{story['id']}/items/{item['id']}/fades",
        json={"fade_in_ms": 500, "fade_out_ms": 500},
    )
    assert r.status_code == 200, r.text

    data, sr = _decode(_export(client, story["id"]))
    mono = data.mean(axis=1)

    head = np.abs(mono[: sr // 100]).max()
    tail = np.abs(mono[-sr // 100 :]).max()
    middle = np.abs(mono[len(mono) // 2 - sr // 20 : len(mono) // 2 + sr // 20]).max()

    assert head < 0.02, f"fade-in did not start near silence ({head:.4f})"
    assert tail < 0.02, f"fade-out did not end near silence ({tail:.4f})"
    assert middle > 0.1, "fades swallowed the whole clip"


def test_overlong_fades_are_scaled_not_clipped(client, story, tmp_path):
    """Fades longer than the clip must stay monotonic rather than
    re-brightening in the middle."""
    clip = _import_audio(client, tmp_path, "shortfade.wav", seconds=1.0, sr=48000)
    item = client.post(
        f"/stories/{story['id']}/items", json={"generation_id": clip["id"]}
    ).json()
    client.put(
        f"/stories/{story['id']}/items/{item['id']}/fades",
        json={"fade_in_ms": 5000, "fade_out_ms": 5000},
    )

    data, _sr = _decode(_export(client, story["id"]))
    mono = np.abs(data.mean(axis=1))
    peak_at = int(np.argmax(mono))
    # Peak should sit near the middle, where the two ramps cross.
    assert 0.3 < peak_at / len(mono) < 0.7


# ── Speed ────────────────────────────────────────────────────────────


def test_double_speed_halves_the_clip(client, story, tmp_path):
    clip = _import_audio(client, tmp_path, "speed.wav", seconds=4.0, sr=48000)
    item = client.post(
        f"/stories/{story['id']}/items", json={"generation_id": clip["id"]}
    ).json()

    baseline, _sr = _decode(_export(client, story["id"]))

    r = client.put(f"/stories/{story['id']}/items/{item['id']}/speed", json={"speed": 2.0})
    assert r.status_code == 200, r.text

    faster, _sr = _decode(_export(client, story["id"]))
    ratio = len(faster) / len(baseline)
    assert 0.45 < ratio < 0.55, f"expected ~half length, got ratio {ratio:.2f}"


def test_speed_is_bounded(client, story, tmp_path):
    clip = _import_audio(client, tmp_path, "speedbound.wav")
    item = client.post(
        f"/stories/{story['id']}/items", json={"generation_id": clip["id"]}
    ).json()
    assert (
        client.put(
            f"/stories/{story['id']}/items/{item['id']}/speed", json={"speed": 99.0}
        ).status_code
        == 422
    )


# ── Track gain, mute, solo ───────────────────────────────────────────


def _two_lane_story(client, story, tmp_path, prefix):
    voice = _import_audio(client, tmp_path, f"{prefix}-voice.wav", seconds=2.0, sr=48000, freq=440)
    bed = _import_audio(client, tmp_path, f"{prefix}-bed.wav", seconds=2.0, sr=48000, freq=220)
    client.post(
        f"/stories/{story['id']}/items", json={"generation_id": voice["id"], "track": 0}
    )
    client.post(
        f"/stories/{story['id']}/items",
        json={"generation_id": bed["id"], "track": 1, "start_time_ms": 0},
    )


def test_track_volume_attenuates_that_lane(client, story, tmp_path):
    _two_lane_story(client, story, tmp_path, "vol")
    before, _ = _decode(_export(client, story["id"]))

    client.put(
        f"/stories/{story['id']}/tracks/1",
        json={"volume": 0.0, "muted": False, "soloed": False},
    )
    after, _ = _decode(_export(client, story["id"]))

    assert np.abs(after).max() < np.abs(before).max()


def test_mute_silences_a_lane(client, story, tmp_path):
    _two_lane_story(client, story, tmp_path, "mute")
    client.put(
        f"/stories/{story['id']}/tracks/0",
        json={"volume": 1.0, "muted": True, "soloed": False},
    )
    client.put(
        f"/stories/{story['id']}/tracks/1",
        json={"volume": 0.0, "muted": True, "soloed": False},
    )
    data, _ = _decode(_export(client, story["id"]))
    assert np.abs(data).max() < 1e-6, "muting every lane should render silence"


def test_solo_silences_every_other_lane(client, story, tmp_path):
    """Solo is global: one soloed lane mutes the rest regardless of their own
    mute flags."""
    _two_lane_story(client, story, tmp_path, "solo")
    client.put(
        f"/stories/{story['id']}/tracks/1",
        json={"volume": 1.0, "muted": False, "soloed": True},
    )
    data, sr = _decode(_export(client, story["id"]))

    # Only the 220 Hz bed should remain; check via a coarse spectrum.
    spectrum = np.abs(np.fft.rfft(data.mean(axis=1)))
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    energy_220 = spectrum[(freqs > 200) & (freqs < 240)].sum()
    energy_440 = spectrum[(freqs > 420) & (freqs < 460)].sum()
    assert energy_220 > energy_440 * 5, "soloed lane did not dominate"


def test_deleting_a_story_removes_its_track_settings(client, tmp_path):
    """Track rows are keyed by story_id with no FK cascade, so deleting a
    story has to clear them or they linger as unreachable rows."""
    doomed = client.post("/stories", json={"name": "Doomed"}).json()
    clip = _import_audio(client, tmp_path, "doomed.wav")
    client.post(f"/stories/{doomed['id']}/items", json={"generation_id": clip["id"], "track": 0})
    client.put(
        f"/stories/{doomed['id']}/tracks/0",
        json={"volume": 0.5, "muted": False, "soloed": False},
    )
    assert len(client.get(f"/stories/{doomed['id']}/tracks").json()) == 1

    assert client.delete(f"/stories/{doomed['id']}").status_code == 200
    assert client.get(f"/stories/{doomed['id']}/tracks").json() == []


def test_deleting_track_settings_keeps_the_clips(client, story, tmp_path):
    _two_lane_story(client, story, tmp_path, "del")
    client.put(
        f"/stories/{story['id']}/tracks/1",
        json={"volume": 0.5, "muted": False, "soloed": False},
    )
    assert client.delete(f"/stories/{story['id']}/tracks/1").status_code == 200

    detail = client.get(f"/stories/{story['id']}").json()
    assert any(i["track"] == 1 for i in detail["items"]), "clips vanished with the settings row"
    assert client.get(f"/stories/{story['id']}/tracks").json() == []


def test_ducking_lowers_the_bed(client, story, tmp_path):
    _two_lane_story(client, story, tmp_path, "duck")
    before, _ = _decode(_export(client, story["id"]))

    client.put(
        f"/stories/{story['id']}/tracks/1",
        json={"volume": 1.0, "muted": False, "soloed": False, "duck_under_track": 0},
    )
    after, sr = _decode(_export(client, story["id"]))

    spectrum_before = np.abs(np.fft.rfft(before.mean(axis=1)))
    spectrum_after = np.abs(np.fft.rfft(after.mean(axis=1)))
    freqs = np.fft.rfftfreq(len(before), 1 / sr)
    band = (freqs > 200) & (freqs < 240)
    assert spectrum_after[band].sum() < spectrum_before[band].sum(), "bed was not ducked"


# ── Export formats ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fmt", "magic"),
    [
        ("wav", (b"RIFF",)),
        ("mp3", (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")),
        ("ogg", (b"OggS",)),
        ("opus", (b"OggS",)),
        ("flac", (b"fLaC",)),
    ],
)
def test_export_formats(client, story, tmp_path, fmt, magic):
    clip = _import_audio(client, tmp_path, f"fmt-{fmt}.wav", sr=48000)
    client.post(f"/stories/{story['id']}/items", json={"generation_id": clip["id"]})

    raw = _export(client, story["id"], fmt=fmt)
    assert any(raw.startswith(m) for m in magic), f"{fmt} magic bytes wrong: {raw[:4]!r}"


def test_default_export_is_still_wav(client, story, tmp_path):
    clip = _import_audio(client, tmp_path, "default.wav")
    client.post(f"/stories/{story['id']}/items", json={"generation_id": clip["id"]})
    assert _export(client, story["id"]).startswith(b"RIFF")


def test_unknown_format_is_rejected(client, story):
    r = client.get(f"/stories/{story['id']}/export-audio", params={"format": "aiff"})
    assert r.status_code == 400


# ── Track validation (review findings on #1007) ──────────────────────


def test_a_lane_cannot_duck_under_itself(client, story):
    """It would attenuate by its own envelope — quieter wherever it is loudest."""
    r = client.put(f"/stories/{story['id']}/tracks/1", json={"duck_under_track": 1})
    assert r.status_code == 400
    assert "itself" in r.json()["detail"]


def test_a_negative_duck_target_is_rejected(client, story):
    """Lane indices are non-negative, so a negative target is not a lane — it
    would just silently never match one at mix time."""
    r = client.put(f"/stories/{story['id']}/tracks/0", json={"duck_under_track": -1})
    assert r.status_code == 422


def test_a_valid_duck_target_is_accepted(client, story):
    r = client.put(f"/stories/{story['id']}/tracks/1", json={"duck_under_track": 0})
    assert r.status_code == 200, r.text
    assert r.json()["duck_under_track"] == 0
