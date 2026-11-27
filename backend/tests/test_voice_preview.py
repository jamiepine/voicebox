"""Voice previews: sample phrases, the on-disk cache, and the preview routes.

The renderer is stubbed, so none of this loads a model.
"""

import io
import wave

import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

from backend import config
from backend.routes import profiles as profiles_routes
from backend.utils import voice_preview


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)
    return buf.getvalue()


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_cache_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def render_calls(monkeypatch):
    calls = []

    async def fake_render(engine, language, make_voice_prompt):
        calls.append((engine, language, await make_voice_prompt()))
        return _wav_bytes()

    monkeypatch.setattr(voice_preview, "render_preview", fake_render)
    return calls


@pytest.fixture
def client(cache_dir):
    app = FastAPI()
    app.include_router(profiles_routes.router)
    return TestClient(app)


@pytest.mark.parametrize("engine", ["kokoro", "qwen_custom_voice"])
def test_every_built_in_voice_language_has_its_own_sample_phrase(engine):
    languages = {lang for _vid, _name, _gender, lang in profiles_routes._preset_catalog(engine)}
    assert languages, f"no voices for {engine}"
    assert languages <= set(voice_preview.SAMPLE_PHRASES)


def test_unsupported_language_falls_back_to_english():
    assert voice_preview.sample_phrase_for("xx") == voice_preview.SAMPLE_PHRASES["en"]


def test_cache_path_is_stable_and_distinct_per_voice(cache_dir):
    a = voice_preview.preview_cache_path("kokoro", "am_echo", "en")
    assert a == voice_preview.preview_cache_path("kokoro", "am_echo", "en")
    assert a != voice_preview.preview_cache_path("kokoro", "af_heart", "en")
    assert a != voice_preview.preview_cache_path("kokoro", "am_echo", "es")
    assert a.parent == cache_dir / "voice_previews"


def test_cache_path_never_contains_the_voice_id(cache_dir):
    path = voice_preview.preview_cache_path("kokoro", "../../etc/passwd", "en")
    assert path.parent == cache_dir / "voice_previews"
    assert ".." not in path.name


def test_preset_preview_renders_once_then_serves_from_cache(client, render_calls):
    first = client.get("/profiles/presets/kokoro/am_echo/preview")
    second = client.get("/profiles/presets/kokoro/am_echo/preview")

    assert first.status_code == second.status_code == 200
    assert first.headers["content-type"] == "audio/wav"
    assert first.content == second.content == _wav_bytes()
    assert render_calls == [
        ("kokoro", "en", {"voice_type": "preset", "preset_engine": "kokoro", "preset_voice_id": "am_echo"})
    ]


def test_preset_preview_uses_the_voices_own_language(client, render_calls):
    client.get("/profiles/presets/kokoro/zf_xiaobei/preview")
    assert render_calls[0][1] == "zh"


@pytest.mark.parametrize(
    "path",
    [
        "/profiles/presets/kokoro/not_a_voice/preview",
        "/profiles/presets/not_an_engine/am_echo/preview",
    ],
)
def test_unknown_preset_voice_is_404(client, render_calls, path):
    assert client.get(path).status_code == 404
    assert render_calls == []


@pytest.mark.asyncio
async def test_missing_model_is_reported_in_ui_terms(monkeypatch):
    import backend.backends as backends

    async def not_cached(engine, model_size="default"):
        raise HTTPException(status_code=400, detail="Use /generate to trigger a download.")

    monkeypatch.setattr(backends, "ensure_model_cached_or_raise", not_cached)

    async def prompt():
        return {}

    with pytest.raises(HTTPException) as exc:
        await voice_preview.render_preview("kokoro", "en", prompt)

    assert exc.value.status_code == 409
    assert "Models tab" in exc.value.detail
    assert "/generate" not in exc.value.detail
