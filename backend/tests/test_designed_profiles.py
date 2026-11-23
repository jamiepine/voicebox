"""Designed voice profiles: creation, engine validation, voice prompt, and the
VoiceDesign backend's instruct layering. None of this loads the model."""

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend import models
from backend.backends import TTS_ENGINES, get_tts_model_configs
from backend.backends.qwen_voice_design_backend import QwenVoiceDesignBackend
from backend.database import Base, get_db
from backend.models import VoiceProfileCreate
from backend.routes import profiles as profile_routes
from backend.routes.generations import _resolve_generation_engine
from backend.services.profiles import (
    DEFAULT_DESIGN_ENGINE,
    create_profile,
    create_voice_prompt_for_profile,
    validate_profile_engine,
)

DESIGN = "A warm, gravelly older man with a slow Scottish lilt."


@pytest.fixture
def test_db():
    temp_dir = tempfile.mkdtemp()
    engine = create_engine(f"sqlite:///{Path(temp_dir) / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    yield db
    db.close()
    engine.dispose()
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_profiles_dir(monkeypatch, tmp_path):
    from backend import config

    monkeypatch.setattr(config, "get_profiles_dir", lambda: tmp_path)
    return tmp_path


def test_engine_is_registered():
    assert "qwen_voice_design" in TTS_ENGINES
    configs = [c for c in get_tts_model_configs() if c.engine == "qwen_voice_design"]
    assert [c.model_name for c in configs] == ["qwen-voice-design-1.7B"]
    assert configs[0].hf_repo_id == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def test_generation_request_accepts_engine():
    req = models.GenerationRequest(profile_id="p", text="hi", engine="qwen_voice_design")
    assert req.engine == "qwen_voice_design"


@pytest.mark.asyncio
async def test_designed_profile_gets_design_engine_by_default(test_db, mock_profiles_dir):
    profile = await create_profile(
        VoiceProfileCreate(name="Designed", language="en", voice_type="designed", design_prompt=DESIGN),
        test_db,
    )
    assert profile.voice_type == "designed"
    assert profile.default_engine == DEFAULT_DESIGN_ENGINE
    assert profile.design_prompt == DESIGN

    voice_prompt = await create_voice_prompt_for_profile(profile.id, test_db, engine=DEFAULT_DESIGN_ENGINE)
    assert voice_prompt == {"voice_type": "designed", "design_prompt": DESIGN}


@pytest.mark.asyncio
async def test_designed_profile_api_persists_updated_design_prompt(test_db, mock_profiles_dir):
    app = FastAPI()
    app.include_router(profile_routes.router)
    sessions = sessionmaker(bind=test_db.get_bind())

    def isolated_db() -> Iterator[Session]:
        with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = isolated_db
    payload = {
        "name": "Designed",
        "language": "en",
        "voice_type": "designed",
        "design_prompt": DESIGN,
    }
    updated_prompt = "A measured voice that speaks clearly."
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://voicebox.test") as client:
        created = await client.post("/profiles", json=payload)
        assert created.status_code == 200, created.text
        assert created.json()["design_prompt"] == DESIGN
        path = f"/profiles/{created.json()['id']}"

        updated = await client.put(path, json={**payload, "design_prompt": updated_prompt})
        assert updated.status_code == 200, updated.text
        assert updated.json()["design_prompt"] == updated_prompt

        loaded = await client.get(path)
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["design_prompt"] == updated_prompt

        renamed = await client.put(path, json={"name": "Renamed", "language": "en"})
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["design_prompt"] == updated_prompt

        invalid = await client.put(path, json={**payload, "design_prompt": " "})
        assert invalid.status_code == 400, invalid.text
        loaded = await client.get(path)
        assert loaded.json()["design_prompt"] == updated_prompt


@pytest.mark.asyncio
async def test_designed_profile_rejects_cloning_engine(test_db, mock_profiles_dir):
    with pytest.raises(ValueError, match="cannot use default engine"):
        await create_profile(
            VoiceProfileCreate(
                name="Bad", language="en", voice_type="designed", design_prompt=DESIGN, default_engine="qwen"
            ),
            test_db,
        )

    profile = await create_profile(
        VoiceProfileCreate(name="Designed", language="en", voice_type="designed", design_prompt=DESIGN),
        test_db,
    )
    with pytest.raises(ValueError, match="does not support designed"):
        validate_profile_engine(profile, "chatterbox")
    validate_profile_engine(profile, "qwen_voice_design")


@pytest.mark.asyncio
async def test_cloned_profile_rejects_design_engine(test_db, mock_profiles_dir):
    profile = await create_profile(VoiceProfileCreate(name="Cloned", language="en"), test_db)
    assert profile.voice_type == "cloned"
    assert profile.default_engine is None
    with pytest.raises(ValueError, match="does not support cloned"):
        validate_profile_engine(profile, "qwen_voice_design")


def test_legacy_designed_profile_without_default_engine_resolves_to_design_engine():
    request = models.GenerationRequest(profile_id="p", text="hi")
    legacy = SimpleNamespace(voice_type="designed", default_engine=None, preset_engine=None)
    assert _resolve_generation_engine(request, legacy) == DEFAULT_DESIGN_ENGINE

    cloned = SimpleNamespace(voice_type="cloned", default_engine=None, preset_engine=None)
    assert _resolve_generation_engine(request, cloned) == "qwen"

    pinned = SimpleNamespace(voice_type="designed", default_engine="qwen_voice_design", preset_engine=None)
    assert _resolve_generation_engine(request, pinned) == "qwen_voice_design"


@pytest.mark.asyncio
async def test_backend_layers_instruct_on_design_prompt(monkeypatch):
    backend = QwenVoiceDesignBackend()
    calls = []

    class FakeModel:
        def generate_voice_design(self, text, instruct, language):
            calls.append((text, instruct, language))
            import numpy as np

            return [np.zeros(24000, dtype=np.float32)], 24000

    async def fake_load(model_size=None):
        backend.model = FakeModel()

    monkeypatch.setattr(backend, "load_model_async", fake_load)
    vp = {"voice_type": "designed", "design_prompt": DESIGN}

    audio, sr = await backend.generate("Hello.", vp, language="en", seed=1)
    assert sr == 24000 and len(audio) == 24000
    assert calls[-1] == ("Hello.", DESIGN, "English")

    await backend.generate("Hello.", vp, language="fr", instruct="Whisper it.")
    assert calls[-1][1] == f"{DESIGN.rstrip('. ')}. Whisper it."
    assert calls[-1][2] == "French"


@pytest.mark.asyncio
async def test_update_coerces_legacy_designed_profile_engine(test_db, mock_profiles_dir):
    from backend.database.models import VoiceProfile as DBVoiceProfile
    from backend.services.profiles import update_profile

    profile = await create_profile(
        VoiceProfileCreate(name="Legacy", language="en", voice_type="designed", design_prompt=DESIGN),
        test_db,
    )
    row = test_db.query(DBVoiceProfile).filter_by(id=profile.id).first()
    row.default_engine = "qwen"
    test_db.commit()

    updated = await update_profile(profile.id, VoiceProfileCreate(name="Renamed", language="en"), test_db)
    assert updated.name == "Renamed"
    assert updated.default_engine == DEFAULT_DESIGN_ENGINE


def test_openai_compat_alias_resolves_legacy_designed_profile():
    from backend.routes.openai_compat import resolve_model

    legacy = SimpleNamespace(voice_type="designed", default_engine=None, preset_engine=None)
    assert resolve_model("tts-1", legacy) == (DEFAULT_DESIGN_ENGINE, None)
