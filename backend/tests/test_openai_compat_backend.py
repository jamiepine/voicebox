"""
Tests for the custom OpenAI-compatible LLM endpoint against a fake server.

Unlike ``test_openai_compat_integration.py`` (which needs a real model
server and skips otherwise), these boot a tiny in-process HTTP server that
speaks just enough of ``/v1/chat/completions`` to check the request shape
Voicebox sends, the response shapes it accepts, how failures surface, and
how the settings layer switches dispatch between the remote backend and
the built-in Qwen path.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from backend import models
from backend.backends import (
    _OPENAI_COMPAT_CACHE_KEY,
    _llm_backends,
    get_llm_backend,
    reset_backends,
    set_llm_config,
)
from backend.backends.openai_compat_backend import OpenAICompatLLMBackend


class FakeOpenAIServer:
    """Minimal OpenAI-compatible chat server that records every request.

    ``script`` is a callable ``(path, headers, body) -> (status, payload)``;
    ``payload`` is JSON-encoded unless it is a ``bytes`` object, which is
    sent verbatim (to simulate non-JSON bodies).
    """

    def __init__(self, script):
        self.script = script
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                body = json.loads(raw) if raw else None
                server.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                status, payload = server.script(self.path, self.headers, body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                if status in (301, 302, 307, 308):
                    self.send_header("Location", payload["location"])
                    data = b""
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def _chat_reply(text: str):
    return 200, {"choices": [{"message": {"role": "assistant", "content": text}}]}


@pytest.fixture(autouse=True)
def _clean_backends():
    reset_backends()
    yield
    reset_backends()


@pytest.mark.asyncio
async def test_generate_sends_openai_chat_shape_with_bearer_and_examples():
    with FakeOpenAIServer(lambda *_: _chat_reply("  Refined text.  ")) as server:
        backend = OpenAICompatLLMBackend(
            endpoint=server.base_url + "/",
            model="my-model",
            api_key="sk-test",
        )
        out = await backend.generate(
            prompt="um hello there",
            system="You clean transcripts.",
            max_tokens=64,
            temperature=0.2,
            model_size="0.6B",
            examples=[("um hi", "Hi.")],
        )

    assert out == "Refined text."
    (req,) = server.requests
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test"
    assert req["body"]["model"] == "my-model"
    assert req["body"]["stream"] is False
    assert req["body"]["max_tokens"] == 64
    assert req["body"]["temperature"] == 0.2
    assert req["body"]["messages"] == [
        {"role": "system", "content": "You clean transcripts."},
        {"role": "user", "content": "um hi"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "um hello there"},
    ]


@pytest.mark.asyncio
async def test_generate_without_api_key_sends_no_authorization_header():
    with FakeOpenAIServer(lambda *_: _chat_reply("ok")) as server:
        backend = OpenAICompatLLMBackend(endpoint=server.base_url, model="m", api_key="")
        assert await backend.generate(prompt="x") == "ok"
    assert "Authorization" not in server.requests[0]["headers"]


@pytest.mark.asyncio
async def test_generate_accepts_legacy_text_completion_shape():
    with FakeOpenAIServer(lambda *_: (200, {"choices": [{"text": " legacy "}]})) as server:
        backend = OpenAICompatLLMBackend(endpoint=server.base_url, model="m")
        assert await backend.generate(prompt="x") == "legacy"


@pytest.mark.asyncio
async def test_http_error_surfaces_as_runtime_error_naming_the_endpoint():
    with FakeOpenAIServer(lambda *_: (401, {"error": {"message": "bad key"}})) as server:
        backend = OpenAICompatLLMBackend(endpoint=server.base_url, model="m", api_key="nope")
        with pytest.raises(RuntimeError) as excinfo:
            await backend.generate(prompt="x")
    message = str(excinfo.value)
    assert "HTTP 401" in message
    assert server.base_url in message
    assert "bad key" in message


@pytest.mark.asyncio
async def test_unreachable_endpoint_surfaces_as_runtime_error():
    with FakeOpenAIServer(lambda *_: _chat_reply("ok")) as server:
        dead_url = server.base_url
    backend = OpenAICompatLLMBackend(endpoint=dead_url, model="m", timeout=2.0)
    with pytest.raises(RuntimeError) as excinfo:
        await backend.generate(prompt="x")
    assert "unreachable" in str(excinfo.value)


@pytest.mark.asyncio
async def test_non_json_body_surfaces_as_runtime_error():
    with FakeOpenAIServer(lambda *_: (200, b"<html>proxy page</html>")) as server:
        backend = OpenAICompatLLMBackend(endpoint=server.base_url, model="m")
        with pytest.raises(RuntimeError) as excinfo:
            await backend.generate(prompt="x")
    assert "non-JSON" in str(excinfo.value)


@pytest.mark.asyncio
async def test_empty_choices_raises_value_error():
    with FakeOpenAIServer(lambda *_: (200, {"choices": []})) as server:
        backend = OpenAICompatLLMBackend(endpoint=server.base_url, model="m")
        with pytest.raises(ValueError, match="No choices"):
            await backend.generate(prompt="x")


def test_dispatch_switches_between_remote_and_builtin():
    set_llm_config("http://localhost:1/v1", "m", None)
    remote = get_llm_backend()
    assert isinstance(remote, OpenAICompatLLMBackend)
    assert get_llm_backend() is remote, "same config must reuse the cached instance"

    set_llm_config("http://localhost:2/v1", "m", "key")
    rotated = get_llm_backend()
    assert rotated is not remote
    assert rotated.endpoint == "http://localhost:2/v1"
    assert rotated.api_key == "key"

    set_llm_config("", "", "")
    assert _OPENAI_COMPAT_CACHE_KEY not in _llm_backends
    assert not isinstance(get_llm_backend(), OpenAICompatLLMBackend)


def test_dispatch_requires_both_endpoint_and_model():
    set_llm_config("http://localhost:1/v1", None, None)
    assert not isinstance(get_llm_backend(), OpenAICompatLLMBackend)
    set_llm_config(None, "m", None)
    assert not isinstance(get_llm_backend(), OpenAICompatLLMBackend)


def test_capture_settings_response_never_echoes_the_api_key():
    stored = models.CaptureSettingsResponse(
        custom_llm_endpoint="http://localhost:1/v1",
        custom_llm_model="m",
        custom_llm_api_key="sk-secret",
    )
    assert stored.custom_llm_api_key_configured is True
    assert "sk-secret" not in stored.model_dump_json()

    empty = models.CaptureSettingsResponse(custom_llm_api_key="")
    assert empty.custom_llm_api_key_configured is False


def test_settings_write_syncs_dispatch_and_readiness(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.database.models import Base
    from backend.services import settings as settings_service

    engine = create_engine(f"sqlite:///{tmp_path / 'settings.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        row = settings_service.update_capture_settings(
            db,
            {"custom_llm_endpoint": "http://localhost:1/v1", "custom_llm_model": "m", "custom_llm_api_key": "k"},
        )
        assert isinstance(get_llm_backend(), OpenAICompatLLMBackend)

        from backend.backends import get_llm_model_configs
        from backend.routes.captures import _llm_readiness

        llm_cfg = next(c for c in get_llm_model_configs() if c.model_size == row.llm_model)
        readiness = _llm_readiness(row, llm_cfg)
        assert readiness.ready is True
        assert readiness.display_name == "m"

        reset_backends()
        settings_service.bootstrap_llm_backend_config(db)
        assert isinstance(get_llm_backend(), OpenAICompatLLMBackend)

        row = settings_service.update_capture_settings(db, {"custom_llm_endpoint": None})
        assert not isinstance(get_llm_backend(), OpenAICompatLLMBackend)
        assert _llm_readiness(row, llm_cfg).display_name == llm_cfg.display_name
    finally:
        db.close()


def test_routes_round_trip_through_fake_server(tmp_path):
    """PUT /settings/captures → GET masks the key → /llm/generate and
    /capture/readiness honour the remote endpoint → clearing it reverts."""
    from fastapi import FastAPI
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from starlette.testclient import TestClient

    from backend.database import get_db
    from backend.database.models import Base
    from backend.routes import captures as captures_routes, llm as llm_routes, settings as settings_routes

    engine = create_engine(
        f"sqlite:///{tmp_path / 'routes.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    def _db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(settings_routes.router)
    app.include_router(llm_routes.router)
    app.include_router(captures_routes.router)
    app.dependency_overrides[get_db] = _db
    client = TestClient(app)

    with FakeOpenAIServer(lambda *_: _chat_reply("Hello from remote.")) as server:
        res = client.put(
            "/settings/captures",
            json={
                "custom_llm_endpoint": server.base_url,
                "custom_llm_model": "remote-model",
                "custom_llm_api_key": "sk-secret",
            },
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["custom_llm_endpoint"] == server.base_url
        assert body["custom_llm_api_key_configured"] is True
        assert "sk-secret" not in res.text

        assert "sk-secret" not in client.get("/settings/captures").text

        readiness = client.get("/capture/readiness").json()
        assert readiness["llm"]["ready"] is True
        assert readiness["llm"]["display_name"] == "remote-model"

        res = client.post("/llm/generate", json={"prompt": "um hi"})
        assert res.status_code == 200, res.text
        assert res.json() == {"text": "Hello from remote.", "model_size": "remote-model"}
        assert server.requests[-1]["headers"]["Authorization"] == "Bearer sk-secret"

        res = client.put("/settings/captures", json={"custom_llm_api_key": None})
        assert res.json()["custom_llm_api_key_configured"] is False
        client.post("/llm/generate", json={"prompt": "again"})
        assert "Authorization" not in server.requests[-1]["headers"]

    # Server is gone: the failure is a clean 502 naming the endpoint, not a hang.
    res = client.post("/llm/generate", json={"prompt": "um hi"})
    assert res.status_code == 502
    assert "unreachable" in res.text

    client.put("/settings/captures", json={"custom_llm_endpoint": None, "custom_llm_model": None})
    assert not isinstance(get_llm_backend(), OpenAICompatLLMBackend)
    readiness = client.get("/capture/readiness").json()
    assert readiness["llm"]["display_name"] != "remote-model"


@pytest.mark.asyncio
async def test_generate_follows_redirects():
    def script(path, _headers, _body):
        if path == "/v1/chat/completions":
            return 307, {"location": "/v2/chat/completions"}
        return _chat_reply("moved")

    with FakeOpenAIServer(script) as server:
        backend = OpenAICompatLLMBackend(endpoint=server.base_url, model="m")
        assert await backend.generate(prompt="x") == "moved"
    assert [r["path"] for r in server.requests] == ["/v1/chat/completions", "/v2/chat/completions"]


@pytest.mark.asyncio
async def test_refinement_and_personality_record_remote_model_label():
    from backend.services import personality, refinement

    with FakeOpenAIServer(lambda *_: _chat_reply("Refined.")) as server:
        set_llm_config(server.base_url, "remote-model", None)
        text, size = await refinement.refine_transcript(
            "um hello", refinement.RefinementFlags(), model_size="0.6B"
        )
        assert (text, size) == ("Refined.", "remote-model")
        result = await personality.rewrite_as_profile("A pirate.", "hello", model_size="0.6B")
        assert result.model_size == "remote-model"
