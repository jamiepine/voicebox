"""
Tests for the sample transcript coverage warning (jamiepine/voicebox#1086).

Qwen's in-context cloning continues from the reference audio *and* its
transcript. When the transcript covers only part of the clip (typically
because Whisper was forced to the profile's language on a clip spoken in
another one), generation yields a fraction of a second of noise or minutes
of babble. The backend now returns a ``warning`` on the sample response when
the transcript's density is implausible for the clip's duration.

Usage:
    python -m pytest backend/tests/test_sample_transcript_coverage.py -v
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import config
from backend.database import Base, VoiceProfile, get_db
from backend.routes.profiles import router as profiles_router
from backend.utils.audio import transcript_coverage_warning

# A faithful transcript of 10 s of read speech is ~100-200 characters.
TEN_SECONDS_OF_SPEECH = (
    "The quick brown fox jumps over the lazy dog while the old clock on the "
    "wall keeps ticking through the long, quiet afternoon in the village."
)


class TestTranscriptCoverageWarning:
    def test_plausible_transcript_passes(self):
        assert transcript_coverage_warning(TEN_SECONDS_OF_SPEECH, 10.0) is None

    def test_one_sentence_for_a_long_clip_warns(self):
        # The #1086 shape: a 29 s clip whose forced-language transcript came
        # back as a single sentence.
        warning = transcript_coverage_warning("Hello there, how are you?", 29.0)
        assert warning is not None
        assert "too short" in warning
        assert "29 s" in warning

    def test_pasted_paragraph_for_a_short_clip_warns(self):
        warning = transcript_coverage_warning(TEN_SECONDS_OF_SPEECH * 3, 3.0)
        assert warning is not None
        assert "too long" in warning

    def test_cjk_characters_count_as_syllables(self):
        # 10 s of Mandarin is ~30-50 characters; without weighting this would
        # read as "too short".
        text = "今天天气很好，我们一起去公园散步，看看花开得怎么样，然后再去喝杯咖啡吧。"
        assert transcript_coverage_warning(text, 10.0) is None

    def test_surrounding_and_repeated_whitespace_is_ignored(self):
        padded = "   " + TEN_SECONDS_OF_SPEECH.replace(" ", "     ") + "\n\n"
        assert transcript_coverage_warning(padded, 10.0) is None
        assert "too short" in transcript_coverage_warning("hi" + " " * 500, 20.0)

    def test_degenerate_inputs_are_silent(self):
        assert transcript_coverage_warning("", 10.0) is None
        assert transcript_coverage_warning("   ", 10.0) is None
        assert transcript_coverage_warning("anything at all", 0.0) is None
        assert transcript_coverage_warning("anything at all", -1.0) is None


def _sine_wav_bytes(seconds: float, sr: int = 24000) -> bytes:
    """A WAV that passes reference-audio validation (2-30 s, RMS >= 0.01)."""
    t = np.arange(int(seconds * sr)) / sr
    audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Minimal app with only the profile routes and a temp sqlite DB."""
    monkeypatch.setattr(config, "_data_dir", tmp_path)

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = testing_session_local()
    session.add(VoiceProfile(id="profile-1", name="Test Profile", language="en"))
    session.commit()
    session.close()

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(profiles_router)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def _add_sample(client, reference_text: str, seconds: float = 10.0):
    return client.post(
        "/profiles/profile-1/samples",
        files={"file": ("sample.wav", _sine_wav_bytes(seconds), "audio/wav")},
        data={"reference_text": reference_text},
    )


class TestSampleRoutes:
    def test_add_sample_with_full_transcript_has_no_warning(self, client):
        response = _add_sample(client, TEN_SECONDS_OF_SPEECH)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["reference_text"] == TEN_SECONDS_OF_SPEECH
        assert body["warning"] is None

    def test_add_sample_with_partial_transcript_is_stored_with_warning(self, client):
        response = _add_sample(client, "Hello there.", seconds=20.0)
        assert response.status_code == 200, response.text
        body = response.json()
        # The sample is kept (the user may be right), but they are told.
        assert body["reference_text"] == "Hello there."
        assert "too short" in body["warning"]
        assert "20 s" in body["warning"]
        assert client.get("/profiles/profile-1/samples").json()[0]["id"] == body["id"]

    def test_update_sample_rechecks_the_new_text(self, client):
        sample_id = _add_sample(client, "Hello there.", seconds=20.0).json()["id"]

        fixed = client.put(
            f"/profiles/samples/{sample_id}",
            json={"reference_text": TEN_SECONDS_OF_SPEECH * 2},
        )
        assert fixed.status_code == 200, fixed.text
        assert fixed.json()["warning"] is None

        broken = client.put(f"/profiles/samples/{sample_id}", json={"reference_text": "Hi."})
        assert broken.status_code == 200, broken.text
        assert "too short" in broken.json()["warning"]

    def test_update_sample_with_missing_audio_still_updates_text(self, client):
        sample_id = _add_sample(client, TEN_SECONDS_OF_SPEECH).json()["id"]
        audio_path = client.get("/profiles/profile-1/samples").json()[0]["audio_path"]
        config.resolve_storage_path(audio_path).unlink()

        response = client.put(f"/profiles/samples/{sample_id}", json={"reference_text": "Hi."})
        assert response.status_code == 200, response.text
        assert response.json()["reference_text"] == "Hi."
        # No duration means no basis for a warning, and no 500.
        assert response.json()["warning"] is None
