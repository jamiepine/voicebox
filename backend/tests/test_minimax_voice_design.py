"""Tests for MiniMax voice design — designed profiles get a real, persisted voice.

The ``designed`` voice_type has existed in the profiles schema since v0.3.x, but
nothing ever built the voice: ``create_voice_prompt_for_profile`` handed engines
a bare text description, so a designed profile silently rendered in whatever
default voice the engine happened to use. These tests pin the wiring — the
written description plus a freshly minted ``voice_id`` go to the regional
voice-design endpoint, the voice id that comes back is persisted on the profile,
and later synthesis references it instead of re-designing the voice.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, VoiceProfile as DBVoiceProfile
from backend.services import profiles, voice_design

DESIGN_PROMPT = "A warm, low-pitched narrator with a steady pace."
A_VOICE_ID = "voicebox_design_abc12345"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Keep ambient credentials and region settings out of the assertions."""
    monkeypatch.delenv(voice_design.API_KEY_ENV_VAR, raising=False)
    monkeypatch.delenv(voice_design.API_REGION_ENV_VAR, raising=False)


@pytest.fixture
def test_db():
    """Temporary SQLite database carrying the real profiles schema."""
    temp_dir = tempfile.mkdtemp()
    engine = create_engine(f"sqlite:///{Path(temp_dir) / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()

    yield session

    session.close()
    shutil.rmtree(temp_dir)


def _add_designed_profile(db) -> DBVoiceProfile:
    profile = DBVoiceProfile(
        id="profile-designed",
        name="Designed narrator",
        language="en",
        voice_type="designed",
        design_prompt=DESIGN_PROMPT,
    )
    db.add(profile)
    db.commit()
    return profile


# ── Region and endpoint resolution ───────────────────────────────────


def test_default_region_is_used_when_unset():
    assert voice_design.resolve_region() == voice_design.MINIMAX_DEFAULT_REGION


def test_region_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(voice_design.API_REGION_ENV_VAR, "cn_zh")
    assert voice_design.resolve_region() == "cn_zh"


def test_explicit_region_beats_the_environment(monkeypatch):
    monkeypatch.setenv(voice_design.API_REGION_ENV_VAR, "cn_zh")
    assert voice_design.resolve_region("global_en") == "global_en"


def test_each_region_has_its_own_voice_design_endpoint():
    endpoints = voice_design.MINIMAX_VOICE_DESIGN_ENDPOINTS
    assert voice_design.MINIMAX_DEFAULT_REGION in endpoints
    assert len(set(endpoints.values())) == len(endpoints)
    for url in endpoints.values():
        assert url.startswith("https://")
        assert url.endswith("/v1/voice_design")


def test_unknown_region_is_rejected():
    with pytest.raises(ValueError, match="Unknown MiniMax API region"):
        voice_design.resolve_region("nowhere")


# ── Minting and validating the voice id we claim ──────────────────────


def test_generated_voice_ids_are_prefixed_unique_and_valid():
    first = voice_design.generate_voice_id()
    second = voice_design.generate_voice_id()
    assert first != second
    assert first.startswith(voice_design.VOICE_ID_PREFIX)
    assert voice_design.validate_voice_id(first) == first


@pytest.mark.parametrize("bad_voice_id", ["", "   ", "short", "1leads_with_digit", "has spaces", "has.dot"])
def test_invalid_voice_ids_are_rejected(bad_voice_id):
    with pytest.raises(ValueError, match="voice_id"):
        voice_design.validate_voice_id(bad_voice_id)


def test_overlong_voice_id_is_rejected():
    with pytest.raises(ValueError, match="at most"):
        voice_design.validate_voice_id("a" * (voice_design.VOICE_ID_MAX_CHARS + 1))


# ── Request payload ──────────────────────────────────────────────────


def test_payload_carries_the_prompt_the_voice_id_and_a_preview():
    payload = voice_design.build_request_payload(DESIGN_PROMPT, A_VOICE_ID)
    assert payload == {
        "prompt": DESIGN_PROMPT,
        "voice_id": A_VOICE_ID,
        "preview_text": voice_design.DEFAULT_PREVIEW_TEXT,
    }


def test_payload_keeps_a_caller_supplied_preview_text():
    payload = voice_design.build_request_payload(DESIGN_PROMPT, A_VOICE_ID, "Hello from the studio.")
    assert payload["preview_text"] == "Hello from the studio."


def test_blank_design_prompt_is_rejected():
    with pytest.raises(ValueError, match="design_prompt is required"):
        voice_design.build_request_payload("   ", A_VOICE_ID)


def test_overlong_design_prompt_is_rejected():
    prompt = "a" * (voice_design.DESIGN_PROMPT_MAX_CHARS + 1)
    with pytest.raises(ValueError, match="design_prompt must be at most"):
        voice_design.build_request_payload(prompt, A_VOICE_ID)


def test_overlong_preview_text_is_rejected():
    preview = "a" * (voice_design.PREVIEW_TEXT_MAX_CHARS + 1)
    with pytest.raises(ValueError, match="preview_text must be at most"):
        voice_design.build_request_payload(DESIGN_PROMPT, A_VOICE_ID, preview)


# ── Response envelope ────────────────────────────────────────────────


def test_designed_voice_id_is_read_from_the_response():
    body = {"voice_id": A_VOICE_ID, "base_resp": {"status_code": 0, "status_msg": "success"}}
    assert voice_design.extract_voice_id(body, A_VOICE_ID) == A_VOICE_ID


def test_provider_error_in_base_resp_is_surfaced():
    body = {"base_resp": {"status_code": 1004, "status_msg": "authentication failed"}}
    with pytest.raises(RuntimeError, match="1004"):
        voice_design.extract_voice_id(body, A_VOICE_ID)


def test_response_without_a_voice_id_is_rejected():
    with pytest.raises(RuntimeError, match="did not contain a voice_id"):
        voice_design.extract_voice_id({"base_resp": {"status_code": 0}}, A_VOICE_ID)


# ── The design call ──────────────────────────────────────────────────


@pytest.fixture
def captured_request(monkeypatch):
    """Replace the network call with a capturing stub and provide a key."""
    captured = {}

    async def fake_post_voice_design(url, api_key, payload):
        captured["url"] = url
        captured["api_key"] = api_key
        captured["payload"] = payload
        return {"voice_id": payload["voice_id"], "base_resp": {"status_code": 0}}

    monkeypatch.setattr(voice_design, "post_voice_design", fake_post_voice_design)
    monkeypatch.setenv(voice_design.API_KEY_ENV_VAR, "test-key")
    return captured


@pytest.mark.asyncio
async def test_design_voice_sends_the_prompt_and_a_generated_voice_id(captured_request):
    voice_id = await voice_design.design_voice(DESIGN_PROMPT)

    payload = captured_request["payload"]
    assert payload["prompt"] == DESIGN_PROMPT
    assert payload["voice_id"] == voice_id
    assert voice_id.startswith(voice_design.VOICE_ID_PREFIX)
    assert captured_request["api_key"] == "test-key"
    assert captured_request["url"] == voice_design.MINIMAX_VOICE_DESIGN_ENDPOINTS["global_en"]


@pytest.mark.asyncio
async def test_design_voice_targets_the_requested_region(captured_request):
    await voice_design.design_voice(DESIGN_PROMPT, region="cn_zh")
    assert captured_request["url"] == voice_design.MINIMAX_VOICE_DESIGN_ENDPOINTS["cn_zh"]


@pytest.mark.asyncio
async def test_design_voice_requires_an_api_key(monkeypatch):
    async def unreachable(*args, **kwargs):
        raise AssertionError("the endpoint must not be called without a key")

    monkeypatch.setattr(voice_design, "post_voice_design", unreachable)

    with pytest.raises(RuntimeError, match=voice_design.API_KEY_ENV_VAR):
        await voice_design.design_voice(DESIGN_PROMPT)


# ── Persistence and reuse for synthesis ──────────────────────────────


@pytest.mark.asyncio
async def test_design_profile_voice_persists_the_returned_voice_id(test_db, monkeypatch):
    _add_designed_profile(test_db)
    seen = {}

    async def fake_design_voice(design_prompt, voice_id=None, preview_text=None, region=None):
        seen["design_prompt"] = design_prompt
        seen["preview_text"] = preview_text
        seen["region"] = region
        return "voicebox_design_persisted01"

    monkeypatch.setattr(voice_design, "design_voice", fake_design_voice)

    response = await profiles.design_profile_voice(
        "profile-designed", test_db, preview_text="Reading a line.", region="cn_zh"
    )

    assert seen == {
        "design_prompt": DESIGN_PROMPT,
        "preview_text": "Reading a line.",
        "region": "cn_zh",
    }
    assert response.designed_voice_id == "voicebox_design_persisted01"

    stored = test_db.query(DBVoiceProfile).filter_by(id="profile-designed").first()
    assert stored.designed_voice_id == "voicebox_design_persisted01"


@pytest.mark.asyncio
async def test_voice_prompt_references_the_designed_voice(test_db):
    profile = _add_designed_profile(test_db)
    profile.designed_voice_id = "voicebox_design_persisted01"
    test_db.commit()

    voice_prompt = await profiles.create_voice_prompt_for_profile("profile-designed", test_db)

    assert voice_prompt["voice_type"] == "designed"
    assert voice_prompt["design_prompt"] == DESIGN_PROMPT
    assert voice_prompt["voice_id"] == "voicebox_design_persisted01"


@pytest.mark.asyncio
async def test_voice_prompt_omits_the_voice_id_until_the_voice_is_designed(test_db):
    _add_designed_profile(test_db)

    voice_prompt = await profiles.create_voice_prompt_for_profile("profile-designed", test_db)

    assert voice_prompt["design_prompt"] == DESIGN_PROMPT
    assert "voice_id" not in voice_prompt


@pytest.mark.asyncio
async def test_designing_a_cloned_profile_is_rejected(test_db):
    test_db.add(DBVoiceProfile(id="profile-cloned", name="Cloned voice", language="en", voice_type="cloned"))
    test_db.commit()

    with pytest.raises(ValueError, match="not a designed profile"):
        await profiles.design_profile_voice("profile-cloned", test_db)


@pytest.mark.asyncio
async def test_designing_a_profile_without_a_prompt_is_rejected(test_db):
    test_db.add(DBVoiceProfile(id="profile-empty", name="Empty design", language="en", voice_type="designed"))
    test_db.commit()

    with pytest.raises(ValueError, match="missing design_prompt"):
        await profiles.design_profile_voice("profile-empty", test_db)


@pytest.mark.asyncio
async def test_designing_a_missing_profile_is_rejected(test_db):
    with pytest.raises(ValueError, match="Profile not found"):
        await profiles.design_profile_voice("does-not-exist", test_db)
