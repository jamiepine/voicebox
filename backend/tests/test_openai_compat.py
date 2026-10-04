"""Tests for the OpenAI-compatible ``/v1/audio/speech`` and ``/v1/models`` routes.

The routes are mounted on a bare FastAPI app and driven through the official
``openai`` Python SDK over an in-process ASGI transport, so the tests prove a
stock OpenAI client works against Voicebox with only the base URL changed.
Synthesis is stubbed with a sine wave; model loading and cache checks are
no-ops.
"""

import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from openai import AsyncOpenAI, BadRequestError, NotFoundError

import backend.backends as backends
import backend.mcp_server.resolve as resolve_mod
import backend.services.generation as generation_svc
import backend.services.profiles as profiles_svc
import backend.utils.chunked_tts as chunked_tts
from backend.database import get_db
from backend.routes import openai_compat

SAMPLE_RATE = 24000
DURATION_S = 1.0


class _FakeProfile:
    def __init__(self, name="Morgan", voice_type="cloned", default_engine=None, effects_chain=None):
        self.id = "profile-1"
        self.name = name
        self.language = "en"
        self.voice_type = voice_type
        self.default_engine = default_engine
        self.preset_engine = None
        self.preset_voice_id = None
        self.effects_chain = effects_chain


class _FakeBackend:
    device = "cpu"


@pytest.fixture
def app(monkeypatch):
    """App with only the OpenAI-compatible router and a stubbed pipeline."""
    calls = {}

    def fake_resolve_profile(explicit, client_id, db):
        calls["voice"] = explicit
        calls["client_id"] = client_id
        if explicit is None:
            return _FakeProfile(name="Default") if client_id == "bound-agent" else None
        if explicit.lower() in ("morgan", "profile-1"):
            return _FakeProfile()
        return None

    async def fake_ensure_cached(engine, model_size="default"):
        calls["ensure"] = (engine, model_size)

    async def fake_load(engine, model_size="default"):
        calls["load"] = (engine, model_size)

    async def fake_voice_prompt(profile_id, db, use_cache=True, engine="qwen"):
        calls["voice_prompt_engine"] = engine
        return {"fake": True}

    async def fake_generate_chunked(backend, text, voice_prompt, **kwargs):
        calls["text"] = text
        calls["generate_kwargs"] = kwargs
        t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SAMPLE_RATE

    monkeypatch.setattr(resolve_mod, "resolve_profile", fake_resolve_profile)
    monkeypatch.setattr(backends, "ensure_model_cached_or_raise", fake_ensure_cached)
    monkeypatch.setattr(backends, "load_engine_model", fake_load)
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: _FakeBackend())
    monkeypatch.setattr(profiles_svc, "create_voice_prompt_for_profile", fake_voice_prompt)
    monkeypatch.setattr(chunked_tts, "generate_chunked", fake_generate_chunked)
    monkeypatch.setattr(generation_svc, "release_generation_memory", lambda tts_model: None)

    application = FastAPI()
    application.include_router(openai_compat.router)
    application.dependency_overrides[get_db] = lambda: object()
    application.state.calls = calls
    return application


@pytest.fixture
def client(app):
    """Official openai SDK client pointed at the in-process app."""
    transport = httpx.ASGITransport(app=app)
    return AsyncOpenAI(
        base_url="http://voicebox.test/v1",
        api_key="not-used",
        http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.fixture
def raw_client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://voicebox.test")


# ─── /v1/models ───────────────────────────────────────────────────────────


async def test_models_list_includes_aliases_engines_and_variants(client):
    models = await client.models.list()
    ids = {m.id for m in models.data}
    assert {"tts-1", "tts-1-hd", "gpt-4o-mini-tts"} <= ids
    assert {"qwen", "chatterbox", "kokoro"} <= ids
    assert "qwen-tts-0.6B" in ids
    assert all(m.object == "model" and m.owned_by == "voicebox" for m in models.data)


async def test_models_retrieve_known_and_unknown(client):
    model = await client.models.retrieve("qwen")
    assert model.id == "qwen"
    with pytest.raises(NotFoundError) as exc_info:
        await client.models.retrieve("eleven-labs-v9")
    assert exc_info.value.code == "model_not_found"


# ─── /v1/audio/speech: happy paths ────────────────────────────────────────


async def test_speech_wav_default_with_openai_alias_uses_profile_engine(client, app):
    response = await client.audio.speech.create(model="tts-1", voice="Morgan", input="Hello there")
    assert response.response.headers["content-type"] == "audio/wav"
    audio, sr = sf.read(io.BytesIO(response.content))
    assert sr == SAMPLE_RATE
    assert abs(len(audio) / sr - DURATION_S) < 0.01

    calls = app.state.calls
    assert calls["voice"] == "Morgan"
    assert calls["text"] == "Hello there"
    # A cloned profile without a configured engine falls back to qwen at its default size.
    assert calls["load"] == ("qwen", "1.7B")
    assert calls["voice_prompt_engine"] == "qwen"


async def test_speech_model_is_a_voicebox_engine(client, app):
    await client.audio.speech.create(model="chatterbox", voice="Morgan", input="Hi")
    assert app.state.calls["load"] == ("chatterbox", "default")


async def test_speech_model_variant_selects_size(client, app):
    await client.audio.speech.create(model="qwen-tts-0.6B", voice="Morgan", input="Hi")
    assert app.state.calls["load"] == ("qwen", "0.6B")


@pytest.mark.parametrize(
    ("response_format", "media_type", "magic"),
    [
        ("mp3", "audio/mpeg", None),
        ("flac", "audio/flac", b"fLaC"),
        ("opus", "audio/ogg", b"OggS"),
    ],
)
async def test_speech_encodes_compressed_formats(client, response_format, media_type, magic):
    response = await client.audio.speech.create(
        model="tts-1", voice="Morgan", input="Hi", response_format=response_format
    )
    assert response.response.headers["content-type"] == media_type
    assert response.response.headers["content-disposition"].endswith(f'speech.{response_format}"')
    if magic:
        assert response.content[:4] == magic
    # Whatever the container, libsndfile must be able to read our output back.
    audio, _sr = sf.read(io.BytesIO(response.content))
    assert len(audio) > 0


async def test_speech_pcm_is_raw_int16_mono(client):
    response = await client.audio.speech.create(model="tts-1", voice="Morgan", input="Hi", response_format="pcm")
    assert response.response.headers["content-type"] == "audio/pcm"
    samples = np.frombuffer(response.content, dtype="<i2")
    assert len(samples) == int(SAMPLE_RATE * DURATION_S)
    assert samples.max() > 1000  # non-silent after loudness normalization


async def test_speech_speed_time_stretches(client):
    response = await client.audio.speech.create(model="tts-1", voice="Morgan", input="Hi", speed=2.0)
    audio, sr = sf.read(io.BytesIO(response.content))
    assert abs(len(audio) / sr - DURATION_S / 2) < 0.05


async def test_speech_instructions_forwarded_as_instruct(client, app):
    await client.audio.speech.create(
        model="gpt-4o-mini-tts", voice="Morgan", input="Hi", instructions="Whisper, slowly."
    )
    assert app.state.calls["generate_kwargs"]["instruct"] == "Whisper, slowly."


async def test_speech_without_voice_uses_client_binding(raw_client, app):
    response = await raw_client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "Hi"},
        headers={"X-Voicebox-Client-Id": "bound-agent"},
    )
    assert response.status_code == 200
    assert app.state.calls["voice"] is None
    assert app.state.calls["client_id"] == "bound-agent"


# ─── /v1/audio/speech: errors in OpenAI's shape ───────────────────────────


async def test_unknown_voice_is_openai_error(client):
    with pytest.raises(BadRequestError) as exc_info:
        await client.audio.speech.create(model="tts-1", voice="alloy", input="Hi")
    err = exc_info.value
    assert err.code == "voice_not_found"
    assert err.param == "voice"
    assert "alloy" in err.message


async def test_missing_voice_without_default_is_openai_error(raw_client):
    response = await raw_client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hi"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "voice_required"


async def test_unknown_model_is_openai_404(client):
    with pytest.raises(NotFoundError) as exc_info:
        await client.audio.speech.create(model="tts-9000", voice="Morgan", input="Hi")
    assert exc_info.value.code == "model_not_found"
    assert exc_info.value.param == "model"


async def test_unknown_response_format_is_openai_error(raw_client):
    response = await raw_client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "voice": "Morgan", "input": "Hi", "response_format": "wma"},
    )
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "unsupported_format"
    assert body["param"] == "response_format"


async def test_validation_errors_are_400_not_422(raw_client):
    response = await raw_client.post("/v1/audio/speech", json={"model": "tts-1", "voice": "Morgan"})
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["param"] == "input"
    assert body["type"] == "invalid_request_error"

    response = await raw_client.post(
        "/v1/audio/speech", json={"model": "tts-1", "voice": "Morgan", "input": "Hi", "speed": 9}
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "speed"

    response = await raw_client.post(
        "/v1/audio/speech", content=b"not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "JSON" in response.json()["error"]["message"]


async def test_model_not_downloaded_http_exception_is_rewrapped(raw_client, monkeypatch):
    from fastapi import HTTPException

    async def not_cached(engine, model_size="default"):
        raise HTTPException(status_code=400, detail="Model 1.7B is not downloaded yet.")

    monkeypatch.setattr(backends, "ensure_model_cached_or_raise", not_cached)
    response = await raw_client.post("/v1/audio/speech", json={"model": "tts-1", "voice": "Morgan", "input": "Hi"})
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "model_not_downloaded"
    assert "not downloaded" in body["message"]
    assert "/models/load" in body["message"]


async def test_preset_profile_rejects_other_engine(raw_client, monkeypatch):
    def preset_profile(explicit, client_id, db):
        p = _FakeProfile(name="Heart", voice_type="preset")
        p.preset_engine = "kokoro"
        p.preset_voice_id = "af_heart"
        return p

    monkeypatch.setattr(resolve_mod, "resolve_profile", preset_profile)
    response = await raw_client.post("/v1/audio/speech", json={"model": "qwen", "voice": "Heart", "input": "Hi"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "engine_not_compatible"

    # The alias picks the preset's own engine, so the same profile works with tts-1.
    response = await raw_client.post("/v1/audio/speech", json={"model": "tts-1", "voice": "Heart", "input": "Hi"})
    assert response.status_code == 200


# ─── encoder unit coverage ────────────────────────────────────────────────


def test_encode_audio_wav_roundtrip():
    audio = np.zeros(2400, dtype=np.float32)
    content, media_type = openai_compat.encode_audio(audio, 24000, "wav")
    assert media_type == "audio/wav"
    data, sr = sf.read(io.BytesIO(content))
    assert sr == 24000
    assert len(data) == 2400


def test_aac_without_ffmpeg_is_unsupported(monkeypatch):
    monkeypatch.setattr(openai_compat.shutil, "which", lambda name: None)
    with pytest.raises(openai_compat.OpenAIError) as exc_info:
        openai_compat.encode_audio(np.zeros(100, dtype=np.float32), 24000, "aac")
    assert exc_info.value.code == "unsupported_format"
    assert json.loads(exc_info.value.response().body)["error"]["param"] == "response_format"
