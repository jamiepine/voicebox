"""
Regression test for #711: speak-in-character generation (and the compose
endpoint) must honor the refinement model size configured in Settings ->
Model Management (``capture_settings.llm_model``) instead of silently
falling back to whatever the LLM backend singleton last happened to load.

Before the fix, ``routes/generations.py`` and ``routes/profiles.py`` called
into ``personality.rewrite_as_profile`` / ``personality.compose_as_profile``
without a ``model_size`` at all, so the user's selection in the UI had no
effect on which model actually ran.
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import the FastAPI app module first: routes/profiles.py and routes/app.py
# have a circular import (routes -> app -> register_routers -> routes) that
# only resolves cleanly when backend.app is imported before backend.routes.*
# directly, which is what the real entrypoint (backend/main.py) always does.
import backend.app  # noqa: F401
from backend.database import Base
from backend.models import GenerationRequest, VoiceProfileCreate
from backend.services.profiles import create_profile
from backend.services import settings as settings_service


class _StopAfterCapture(Exception):
    """Raised by the patched personality functions once their arguments are
    recorded, so the test doesn't need to mock the rest of the (unrelated)
    generation/composition pipeline."""


@pytest.fixture
def test_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def mock_profiles_dir(monkeypatch, tmp_path):
    from backend import config

    monkeypatch.setattr(config, "get_profiles_dir", lambda: tmp_path)
    return tmp_path


async def _make_personality_profile(test_db):
    data = VoiceProfileCreate(
        name="Personality Test Profile",
        description="",
        language="en",
        voice_type="preset",
        preset_engine="kokoro",
        preset_voice_id="af_heart",
        default_engine="kokoro",
        personality="A grumpy old pirate captain who only speaks in nautical metaphors.",
    )
    return await create_profile(data, test_db)


@pytest.mark.asyncio
async def test_generate_speak_in_character_uses_configured_refinement_model(
    test_db, mock_profiles_dir, monkeypatch
):
    from backend.routes import generations as generations_route

    settings_service.update_capture_settings(test_db, {"llm_model": "1.7B"})
    profile = await _make_personality_profile(test_db)

    captured = {}

    async def fake_rewrite_as_profile(personality_text, user_text, model_size=None):
        captured["model_size"] = model_size
        raise _StopAfterCapture()

    monkeypatch.setattr(generations_route.personality, "rewrite_as_profile", fake_rewrite_as_profile)

    request = GenerationRequest(
        profile_id=profile.id,
        text="Install the dependencies before the deploy.",
        language="en",
        engine="kokoro",
        personality=True,
    )

    with pytest.raises(_StopAfterCapture):
        await generations_route.generate_speech(request, test_db)

    assert captured["model_size"] == "1.7B"


@pytest.mark.asyncio
async def test_generate_speak_in_character_picks_up_changed_refinement_model(
    test_db, mock_profiles_dir, monkeypatch
):
    """A different llm_model setting should reach the LLM call unchanged —
    proves the value is read fresh per request, not cached."""
    from backend.routes import generations as generations_route

    settings_service.update_capture_settings(test_db, {"llm_model": "4B"})
    profile = await _make_personality_profile(test_db)

    captured = {}

    async def fake_rewrite_as_profile(personality_text, user_text, model_size=None):
        captured["model_size"] = model_size
        raise _StopAfterCapture()

    monkeypatch.setattr(generations_route.personality, "rewrite_as_profile", fake_rewrite_as_profile)

    request = GenerationRequest(
        profile_id=profile.id,
        text="The build is broken, roll back to yesterday's version.",
        language="en",
        engine="kokoro",
        personality=True,
    )

    with pytest.raises(_StopAfterCapture):
        await generations_route.generate_speech(request, test_db)

    assert captured["model_size"] == "4B"


@pytest.mark.asyncio
async def test_compose_in_character_uses_configured_refinement_model(
    test_db, mock_profiles_dir, monkeypatch
):
    from backend.routes import profiles as profiles_route

    settings_service.update_capture_settings(test_db, {"llm_model": "1.7B"})
    profile = await _make_personality_profile(test_db)

    captured = {}

    async def fake_compose_as_profile(personality_text, model_size=None):
        captured["model_size"] = model_size
        raise _StopAfterCapture()

    monkeypatch.setattr(profiles_route.personality, "compose_as_profile", fake_compose_as_profile)

    with pytest.raises(_StopAfterCapture):
        await profiles_route.compose_in_character(profile.id, test_db)

    assert captured["model_size"] == "1.7B"
