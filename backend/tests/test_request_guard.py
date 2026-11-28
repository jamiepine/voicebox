"""Tests for the Host / Origin guard and the nosniff header.

The guard is installed with ``configure_request_security``, the same call
``backend/app.py`` makes, so the tests exercise the real origin list and
middleware order.
"""

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.utils.request_guard import DEFAULT_CORS_ORIGINS, configure_request_security


def _build_app() -> FastAPI:
    app = FastAPI()
    configure_request_security(app)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(payload: dict):
        return payload

    @app.get("/stream")
    async def stream():
        async def chunks():
            yield b"data: one\n\n"
            yield b"data: two\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    @app.get("/file")
    async def file():
        return FileResponse(__file__, media_type="text/plain")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    return app


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VOICEBOX_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("VOICEBOX_CORS_ORIGINS", raising=False)


def _client(base_url: str = "http://127.0.0.1:17493") -> TestClient:
    return TestClient(_build_app(), base_url=base_url)


class TestHost:
    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost",
            "http://localhost:17493",
            "http://127.0.0.1",
            "http://127.0.0.1:17493",
            "http://[::1]",
            "http://[::1]:17493",
        ],
    )
    def test_loopback_hosts_allowed(self, base_url):
        assert TestClient(_build_app(), base_url=base_url).get("/health").status_code == 200

    @pytest.mark.parametrize("host", ["LOCALHOST:17493", "LocalHost", "[::1]:5173"])
    def test_host_check_is_case_insensitive_and_ignores_port(self, host):
        assert _client().get("/health", headers={"Host": host}).status_code == 200

    @pytest.mark.parametrize(
        "host",
        [
            "attacker.example",
            "attacker.example:17493",
            "localhost.attacker.example",
            "127.0.0.1.nip.io:17493",
            "",
        ],
    )
    def test_rebinding_hosts_rejected_on_loopback(self, host):
        response = _client().get("/health", headers={"Host": host})
        assert response.status_code == 403
        assert "is not allowed" in response.json()["detail"]

    def test_testserver_default_host_rejected(self):
        assert TestClient(_build_app()).get("/health").status_code == 403

    @pytest.mark.parametrize("host", ["192.168.1.20:17493", "voicebox.lan:17493", "gpu-box"])
    def test_any_host_allowed_on_non_loopback_address(self, host):
        client = _client("http://192.168.1.20:17493")
        assert client.get("/health", headers={"Host": host}).status_code == 200

    def test_ipv4_mapped_loopback_address_is_checked(self):
        client = _client("http://[::ffff:127.0.0.1]:17493")
        assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 403

    def test_allowed_hosts_extends_loopback_list(self, monkeypatch):
        monkeypatch.setenv("VOICEBOX_ALLOWED_HOSTS", "voicebox.lan, Proxy.Example:443")
        client = _client()
        assert client.get("/health", headers={"Host": "voicebox.lan:17493"}).status_code == 200
        assert client.get("/health", headers={"Host": "proxy.example"}).status_code == 200
        assert client.get("/health", headers={"Host": "localhost:17493"}).status_code == 200
        assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 403

    def test_allowed_hosts_applies_on_every_address(self, monkeypatch):
        monkeypatch.setenv("VOICEBOX_ALLOWED_HOSTS", "voicebox.lan")
        client = _client("http://192.168.1.20:17493")
        assert client.get("/health", headers={"Host": "voicebox.lan"}).status_code == 200
        assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 403

    def test_allowed_hosts_wildcard_disables_check(self, monkeypatch):
        monkeypatch.setenv("VOICEBOX_ALLOWED_HOSTS", "*")
        assert _client().get("/health", headers={"Host": "anything.example"}).status_code == 200


class TestOrigin:
    @pytest.mark.parametrize("origin", DEFAULT_CORS_ORIGINS)
    def test_app_origins_allowed(self, origin):
        response = _client().post("/generate", json={"a": 1}, headers={"Origin": origin})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin

    @pytest.mark.parametrize(
        "origin",
        [
            "http://evil.com",
            "http://localhost:9999",
            "https://localhost:17493",
            "http://127.0.0.1:17493.evil.com",
            "null",
        ],
    )
    def test_foreign_origins_rejected(self, origin):
        response = _client().post("/generate", json={"a": 1}, headers={"Origin": origin})
        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers

    def test_no_origin_allowed(self):
        assert _client().post("/generate", json={"a": 1}).json() == {"a": 1}

    def test_same_origin_allowed(self):
        client = _client("http://localhost:17600")
        response = client.post("/generate", json={}, headers={"Origin": "http://localhost:17600"})
        assert response.status_code == 200

    def test_same_origin_allowed_on_lan(self):
        client = _client("http://192.168.1.20:17493")
        response = client.post("/generate", json={}, headers={"Origin": "http://192.168.1.20:17493"})
        assert response.status_code == 200

    def test_lan_still_rejects_foreign_origin(self):
        client = _client("http://192.168.1.20:17493")
        response = client.post("/generate", json={}, headers={"Origin": "http://evil.com"})
        assert response.status_code == 403

    def test_cors_origins_env_extends_list(self, monkeypatch):
        monkeypatch.setenv("VOICEBOX_CORS_ORIGINS", "https://custom.example.com/")
        client = _client()
        response = client.post("/generate", json={}, headers={"Origin": "https://custom.example.com"})
        assert response.status_code == 200
        assert client.post("/generate", json={}, headers={"Origin": "http://evil.com"}).status_code == 403

    @pytest.mark.parametrize("origin", ["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"])
    def test_desktop_preflight(self, origin):
        response = _client().options(
            "/generate",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_foreign_preflight_rejected(self):
        response = _client().options(
            "/generate",
            headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "POST"},
        )
        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers


class TestNosniff:
    @pytest.mark.parametrize(
        ("method", "path", "kwargs", "status"),
        [
            ("get", "/health", {}, 200),
            ("get", "/missing", {}, 404),
            ("post", "/generate", {"content": b"not json"}, 422),
            ("get", "/stream", {}, 200),
            ("get", "/file", {}, 200),
            ("get", "/health", {"headers": {"Origin": "http://evil.com"}}, 403),
            ("get", "/health", {"headers": {"Host": "attacker.example"}}, 403),
        ],
    )
    def test_header_on_every_response(self, method, path, kwargs, status):
        response = getattr(_client(), method)(path, **kwargs)
        assert response.status_code == status
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_streaming_body_intact(self):
        assert _client().get("/stream").text == "data: one\n\ndata: two\n\n"


WS_URL = "ws://127.0.0.1:17493/ws"


class TestWebSocket:
    def test_allowed(self):
        with _client().websocket_connect(WS_URL, headers={"Origin": "tauri://localhost"}) as ws:
            assert ws.receive_text() == "hello"

    @pytest.mark.parametrize("headers", [{"Origin": "http://evil.com"}, {"Host": "attacker.example"}])
    def test_rejected(self, headers):
        with pytest.raises(WebSocketDisconnect) as exc_info, _client().websocket_connect(WS_URL, headers=headers):
            pass
        assert exc_info.value.code == 1008


class TestMcp:
    def test_no_origin_client_reaches_mcp(self):
        from backend.mcp_server.server import build_mcp_server

        mcp_app = build_mcp_server().http_app(path="/", transport="http")
        app = FastAPI(lifespan=mcp_app.router.lifespan_context)
        configure_request_security(app)
        app.mount("/mcp", mcp_app)

        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        }
        headers = {"Accept": "application/json, text/event-stream"}
        with TestClient(app, base_url="http://127.0.0.1:17493") as client:
            response = client.post("/mcp/", json=initialize, headers=headers)
            assert response.status_code == 200
            assert "serverInfo" in response.text
            assert response.headers["x-content-type-options"] == "nosniff"

            rejected = client.post("/mcp/", json=initialize, headers={**headers, "Origin": "http://evil.com"})
            assert rejected.status_code == 403
