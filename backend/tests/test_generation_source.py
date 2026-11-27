"""
Generations record the surface that asked for them (#635).

The frontend skips main-window autoplay for agent sources (``mcp``, ``rest``),
so ``generate_speech`` must store the caller's source instead of always
writing ``manual``, and speak-in-character must not downgrade an explicit
agent source to ``personality_speak``.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.app  # noqa: F401
from backend.database import Base, Generation as DBGeneration
from backend.models import GenerationRequest, VoiceProfileCreate
from backend.services.profiles import create_profile


@pytest.fixture
def test_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_local()
    yield db
    db.close()


@pytest.fixture
def generations_route(monkeypatch, tmp_path):
    from backend import config
    from backend.routes import generations

    monkeypatch.setattr(config, "get_profiles_dir", lambda: tmp_path)

    def fake_enqueue(generation_id, coro):
        coro.close()

    async def fake_rewrite(personality_text, user_text, model_size=None):
        return SimpleNamespace(text="Arr, " + user_text)

    monkeypatch.setattr(generations, "enqueue_generation", fake_enqueue)
    monkeypatch.setattr(generations.personality, "rewrite_as_profile", fake_rewrite)
    return generations


async def _profile(test_db):
    return await create_profile(
        VoiceProfileCreate(
            name="Source Test Profile",
            language="en",
            voice_type="preset",
            preset_engine="kokoro",
            preset_voice_id="af_heart",
            default_engine="kokoro",
            personality="A pirate.",
        ),
        test_db,
    )


async def _stored_source(generations_route, test_db, personality, source):
    profile = await _profile(test_db)
    request = GenerationRequest(
        profile_id=profile.id,
        text="hello",
        language="en",
        engine="kokoro",
        personality=personality,
    )
    generation = await generations_route.generate_speech(request, test_db, source=source)
    row = test_db.query(DBGeneration).filter_by(id=generation.id).first()
    assert row.source == generation.source
    return row.source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("personality", "source", "expected"),
    [
        (False, None, "manual"),
        (True, None, "personality_speak"),
        (False, "mcp", "mcp"),
        (False, "rest", "rest"),
        (True, "mcp", "mcp"),
        (True, "rest", "rest"),
    ],
)
async def test_generation_stores_caller_source(
    generations_route, test_db, personality, source, expected
):
    assert await _stored_source(generations_route, test_db, personality, source) == expected


def test_generate_endpoint_keeps_its_operation_id():
    from backend.app import app

    schema = app.openapi()
    assert schema["paths"]["/generate"]["post"]["operationId"] == "generate_speech_generate_post"
    params = schema["paths"]["/generate"]["post"].get("parameters", [])
    assert all(p["name"] != "source" for p in params)
