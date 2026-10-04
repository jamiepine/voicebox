"""Tests for ``GET /history/{id}/export-audio?format=`` and the shared ffmpeg helper.

The real transcode runs only when ffmpeg/ffprobe are installed; the missing-ffmpeg
path is exercised everywhere by patching the lookup.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import config
from backend.database import Base, Generation, VoiceProfile, get_db
from backend.routes.history import router as history_router
from backend.utils import ffmpeg as ffmpeg_util
from backend.utils.audio import save_audio

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    (tmp_path / "generations").mkdir()
    tone = 0.2 * np.sin(np.linspace(0, 2 * np.pi * 440 * 2, 24000 * 2)).astype(np.float32)
    save_audio(tone, str(tmp_path / "generations" / "gen-1.wav"), 24000)

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()
    session.add(VoiceProfile(id="profile-1", name="Test Profile"))
    session.add(
        Generation(
            id="gen-1",
            profile_id="profile-1",
            text="Hello there, this is a test.",
            audio_path="generations/gen-1.wav",
            status="completed",
        )
    )
    session.commit()
    session.close()

    app = FastAPI()
    app.include_router(history_router)

    def override_get_db():
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_default_export_is_unchanged_wav(client, tmp_path):
    response = client.get("/history/gen-1/export-audio")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert 'filename="Hello there this is a test-gen-1.wav"' in response.headers["content-disposition"]
    assert response.content == (tmp_path / "generations" / "gen-1.wav").read_bytes()


def test_unknown_format_is_400(client):
    assert client.get("/history/gen-1/export-audio?format=ogg").status_code == 400


def test_missing_ffmpeg_is_503_with_install_hint(client, monkeypatch):
    monkeypatch.setattr(ffmpeg_util, "find_ffmpeg", lambda name="ffmpeg": None)
    response = client.get("/history/gen-1/export-audio?format=mp3")
    assert response.status_code == 503
    assert response.json()["detail"] == ffmpeg_util.MISSING_MESSAGE


def test_encode_audio_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported export format"):
        ffmpeg_util.encode_audio(tmp_path / "in.wav", tmp_path / "out.ogg", "ogg")


def test_find_ffmpeg_honours_override_dir(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VOICEBOX_FFMPEG_DIR", str(tmp_path))
    assert ffmpeg_util.find_ffmpeg() == str(fake)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_mp3_export_is_real_mp3(client, tmp_path):
    response = client.get("/history/gen-1/export-audio?format=mp3")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert 'filename="Hello there this is a test-gen-1.mp3"' in response.headers["content-disposition"]

    out = tmp_path / "probe.mp3"
    out.write_bytes(response.content)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(out)],
        capture_output=True,
        check=True,
        text=True,
    )
    info = json.loads(probe.stdout)
    assert info["format"]["format_name"] == "mp3"
    assert info["streams"][0]["codec_name"] == "mp3"
    assert abs(float(info["format"]["duration"]) - 2.0) < 0.2
    # Nothing was cached beside the WAV.
    assert sorted(p.name for p in (tmp_path / "generations").iterdir()) == ["gen-1.wav"]
