import os
import pytest
from unittest.mock import patch
from fastapi import FastAPI, Request
from starlette.testclient import TestClient
from starlette.responses import JSONResponse

from backend.utils.security import SecurityMiddleware, is_loopback


from fastapi.middleware.cors import CORSMiddleware

def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(SecurityMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/shutdown")
    async def shutdown():
        return {"message": "shutting down"}

    @app.post("/models/unload")
    async def unload_model():
        return {"message": "unloaded"}

    @app.delete("/profiles/{profile_id}")
    async def delete_profile(profile_id: str):
        return {"message": f"deleted {profile_id}"}

    return app


@pytest.fixture()
def client():
    return TestClient(_build_app())


def test_is_loopback_helper():
    assert is_loopback("127.0.0.1") is True
    assert is_loopback("::1") is True
    assert is_loopback("localhost") is True
    assert is_loopback("testclient") is True
    assert is_loopback(None) is True
    
    assert is_loopback("192.168.1.1") is False
    assert is_loopback("10.0.0.5") is False
    assert is_loopback("8.8.8.8") is False


def test_loopback_caller_unrestricted(client):
    # Loopback callers should bypass all security gates
    with patch("backend.utils.security.is_loopback", return_value=True):
        # Admin / Destructive endpoints
        assert client.post("/shutdown").status_code == 200
        assert client.post("/models/unload").status_code == 200
        assert client.delete("/profiles/123").status_code == 200
        # Safe endpoints
        assert client.get("/health").status_code == 200


def test_remote_caller_no_api_key_blocks_destructive(client):
    # Remote callers with no VOICEBOX_API_KEY environment variable set
    with patch("backend.utils.security.is_loopback", return_value=False):
        with patch.dict(os.environ, {}, clear=True):
            # Safe endpoints should be allowed
            assert client.get("/health").status_code == 200
            
            # Administrative POST endpoints should be blocked (403)
            res_shutdown = client.post("/shutdown")
            assert res_shutdown.status_code == 403
            assert "restricted to loopback callers" in res_shutdown.json()["detail"]
            
            res_unload = client.post("/models/unload")
            assert res_unload.status_code == 403
            assert "restricted to loopback callers" in res_unload.json()["detail"]
            
            # Destructive DELETE endpoints should be blocked (403)
            res_delete = client.delete("/profiles/123")
            assert res_delete.status_code == 403
            assert "restricted to loopback callers" in res_delete.json()["detail"]


def test_remote_caller_with_api_key_requires_auth(client):
    # Remote callers with VOICEBOX_API_KEY environment variable set
    with patch("backend.utils.security.is_loopback", return_value=False):
        with patch.dict(os.environ, {"VOICEBOX_API_KEY": "secret_token"}):
            # Accessing any endpoint without credentials should fail (401)
            assert client.get("/health").status_code == 401
            assert client.post("/shutdown").status_code == 401
            assert client.delete("/profiles/123").status_code == 401
            
            # Accessing with invalid token should fail (401)
            headers = {"Authorization": "Bearer wrong_token"}
            assert client.get("/health", headers=headers).status_code == 401
            
            # Accessing with valid token should succeed (200)
            headers_valid = {"Authorization": "Bearer secret_token"}
            assert client.get("/health", headers=headers_valid).status_code == 200
            assert client.post("/shutdown", headers=headers_valid).status_code == 200
            assert client.delete("/profiles/123", headers=headers_valid).status_code == 200


def test_options_preflight_is_always_allowed(client):
    # OPTIONS requests (CORS preflight) must bypass authentication check
    with patch("backend.utils.security.is_loopback", return_value=False):
        with patch.dict(os.environ, {"VOICEBOX_API_KEY": "secret_token"}):
            headers = {
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            }
            assert client.options("/health", headers=headers).status_code == 200
            assert client.options("/shutdown", headers=headers).status_code == 200
