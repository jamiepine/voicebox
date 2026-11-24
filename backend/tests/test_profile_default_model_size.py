"""Tests for profile ``default_model_size`` persistence and generation fallback."""

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import GenerationRequest, VoiceProfileCreate
from backend.routes.generations import _resolve_generation_model_size
from backend.services import profiles


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.get_profiles_dir", lambda: tmp_path / "profiles")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.mark.asyncio
async def test_profile_persists_and_clears_default_model_size(db_session):
    created = await profiles.create_profile(
        VoiceProfileCreate(
            name="Nando",
            language="pt",
            default_engine="qwen",
            default_model_size="0.6B",
        ),
        db_session,
    )
    assert created.default_engine == "qwen"
    assert created.default_model_size == "0.6B"

    cleared = await profiles.update_profile(
        created.id,
        VoiceProfileCreate(
            name="Nando",
            language="pt",
            default_engine="",
            default_model_size="0.6B",
        ),
        db_session,
    )
    assert cleared.default_engine is None
    assert cleared.default_model_size is None


def test_generation_falls_back_to_profile_default_model_size():
    profile = SimpleNamespace(default_engine="qwen", default_model_size="0.6B", preset_engine=None)
    engine = profiles.default_engine_for_profile(profile)
    req_without_size = GenerationRequest(profile_id="p1", text="Olá", language="pt")
    assert _resolve_generation_model_size(req_without_size, profile, engine) == "0.6B"

    req_with_size = GenerationRequest(profile_id="p1", text="Olá", language="pt", model_size="1.7B")
    assert _resolve_generation_model_size(req_with_size, profile, engine) == "1.7B"
