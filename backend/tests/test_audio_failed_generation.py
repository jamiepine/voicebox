"""
Regression tests for GET /audio/{generation_id} on failed generations.

A failed generation stores an empty ``audio_path``. Previously,
``config.resolve_storage_path("")`` resolved to the data directory itself,
which exists, so the route's 404 guard passed and ``FileResponse`` raised
``RuntimeError: File at path .../data is not a file`` — a 500 instead of
a clean 404.

Usage:
    python -m pytest backend/tests/test_audio_failed_generation.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

# Repo root on sys.path so ``backend`` imports as a package (the audio
# routes use package-relative imports).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import config
from backend.database import (
    Base,
    Generation,
    GenerationVersion,
    ProfileSample,
    VoiceProfile,
    get_db,
)
from backend.routes.audio import router as audio_router


def test_resolve_storage_path_empty_returns_none():
    """An empty stored path must not resolve to the data dir itself."""
    assert config.resolve_storage_path("") is None
    assert config.resolve_storage_path(None) is None
    # Path("") is truthy, so it must be rejected via its (empty) parts.
    assert config.resolve_storage_path(Path("")) is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Minimal app with only the audio routes and a temp sqlite DB."""
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    # An existing directory that a stored audio_path may wrongly point to.
    (tmp_path / "somedir").mkdir()

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = testing_session_local()
    profile = VoiceProfile(id="profile-1", name="Test Profile")
    session.add(profile)

    session.add_all(
        [
            Generation(
                id="gen-failed-empty",
                profile_id="profile-1",
                text="failed generation",
                audio_path="",
                status="failed",
                error="engine exploded",
            ),
            Generation(
                id="gen-failed-null",
                profile_id="profile-1",
                text="failed generation",
                audio_path=None,
                status="failed",
            ),
            Generation(
                id="gen-missing-file",
                profile_id="profile-1",
                text="completed but file deleted",
                audio_path="generations/does-not-exist.wav",
                status="completed",
            ),
            Generation(
                id="gen-with-version",
                profile_id="profile-1",
                text="generation with a broken version",
                audio_path="somedir",
                status="completed",
            ),
            GenerationVersion(
                id="version-dir",
                generation_id="gen-with-version",
                label="original",
                audio_path="somedir",
            ),
            ProfileSample(
                id="sample-dir",
                profile_id="profile-1",
                audio_path="somedir",
                reference_text="sample pointing at a directory",
            ),
        ]
    )
    session.commit()
    session.close()

    app = FastAPI()
    app.include_router(audio_router)

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.mark.parametrize("generation_id", ["gen-failed-empty", "gen-failed-null"])
def test_failed_generation_returns_404(client, generation_id):
    """Failed generations (empty/null audio_path) get a clean 404, not a 500."""
    response = client.get(f"/audio/{generation_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Generation failed; no audio available"


def test_missing_audio_file_returns_404(client):
    """A completed generation whose file vanished still 404s."""
    response = client.get("/audio/gen-missing-file")
    assert response.status_code == 404
    assert response.json()["detail"] == "Audio file not found"


def test_unknown_generation_returns_404(client):
    response = client.get("/audio/no-such-generation")
    assert response.status_code == 404
    assert response.json()["detail"] == "Generation not found"


@pytest.mark.parametrize(
    "url",
    [
        "/audio/gen-with-version",
        "/audio/version/version-dir",
        "/samples/sample-dir",
    ],
)
def test_audio_path_pointing_at_directory_returns_404(client, url):
    """A stored path resolving to an existing directory must 404, not 500.

    Guards the is_file() checks: a directory passes exists() and would
    crash FileResponse.
    """
    response = client.get(url)
    assert response.status_code == 404
    assert response.json()["detail"] == "Audio file not found"
