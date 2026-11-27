"""
Tests for CORS origin restrictions.

Validates that the CORS middleware only allows known local origins
and respects the VOICEBOX_CORS_ORIGINS environment variable.

Uses a minimal FastAPI app configured by the same
``configure_request_security`` call as backend/app.py, so tests run without
heavy ML dependencies.

Usage:
    pip install httpx pytest fastapi starlette
    python -m pytest backend/tests/test_cors.py -v
"""

import os
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from starlette.testclient import TestClient

from backend.utils.request_guard import configure_request_security

BASE_URL = "http://localhost:17493"


def _build_app(env_origins: str = "") -> FastAPI:
    """Build a minimal FastAPI app with the same CORS setup as backend/app.py."""
    app = FastAPI()
    with patch.dict(os.environ, {"VOICEBOX_CORS_ORIGINS": env_origins}):
        configure_request_security(app)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


@pytest.fixture()
def client():
    return TestClient(_build_app(), base_url=BASE_URL)


@pytest.fixture()
def client_with_custom_origins():
    return TestClient(_build_app("https://custom.example.com,https://other.example.com"), base_url=BASE_URL)


def _get_with_origin(client: TestClient, origin: str) -> dict:
    """Send a GET with Origin header, return response headers."""
    response = client.get("/health", headers={"Origin": origin})
    return dict(response.headers)


def _preflight(client: TestClient, origin: str) -> dict:
    """Send CORS preflight OPTIONS request, return response headers."""
    response = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    return dict(response.headers)


class TestCORSDefaultOrigins:
    """CORS should allow known local origins and block everything else."""

    @pytest.mark.parametrize("origin", [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:17493",
        "http://127.0.0.1:17493",
        "tauri://localhost",
        "https://tauri.localhost",
    ])
    def test_allowed_origins(self, client, origin):
        headers = _get_with_origin(client, origin)
        assert headers.get("access-control-allow-origin") == origin

    @pytest.mark.parametrize("origin", [
        "http://evil.com",
        "http://localhost:9999",
        "https://attacker.example.com",
        "null",
    ])
    def test_blocked_origins(self, client, origin):
        headers = _get_with_origin(client, origin)
        assert "access-control-allow-origin" not in headers

    def test_preflight_allowed(self, client):
        headers = _preflight(client, "http://localhost:5173")
        assert headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_preflight_blocked(self, client):
        headers = _preflight(client, "http://evil.com")
        assert "access-control-allow-origin" not in headers

    def test_credentials_header_present(self, client):
        headers = _get_with_origin(client, "http://localhost:5173")
        assert headers.get("access-control-allow-credentials") == "true"


class TestCORSCustomOrigins:
    """VOICEBOX_CORS_ORIGINS env var should extend the allowlist."""

    def test_custom_origin_allowed(self, client_with_custom_origins):
        headers = _get_with_origin(client_with_custom_origins, "https://custom.example.com")
        assert headers.get("access-control-allow-origin") == "https://custom.example.com"

    def test_other_custom_origin_allowed(self, client_with_custom_origins):
        headers = _get_with_origin(client_with_custom_origins, "https://other.example.com")
        assert headers.get("access-control-allow-origin") == "https://other.example.com"

    def test_default_origins_still_work(self, client_with_custom_origins):
        headers = _get_with_origin(client_with_custom_origins, "http://localhost:5173")
        assert headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_unlisted_origin_still_blocked(self, client_with_custom_origins):
        headers = _get_with_origin(client_with_custom_origins, "http://evil.com")
        assert "access-control-allow-origin" not in headers


class TestCORSEnvVarParsing:
    """Edge cases for VOICEBOX_CORS_ORIGINS parsing."""

    def test_empty_env_var(self):
        app = _build_app("")
        client = TestClient(app, base_url=BASE_URL)
        headers = _get_with_origin(client, "http://evil.com")
        assert "access-control-allow-origin" not in headers

    def test_whitespace_trimmed(self):
        app = _build_app("  https://spaced.example.com  ")
        client = TestClient(app, base_url=BASE_URL)
        headers = _get_with_origin(client, "https://spaced.example.com")
        assert headers.get("access-control-allow-origin") == "https://spaced.example.com"

    def test_trailing_comma_ignored(self):
        app = _build_app("https://one.example.com,")
        client = TestClient(app, base_url=BASE_URL)
        headers = _get_with_origin(client, "https://one.example.com")
        assert headers.get("access-control-allow-origin") == "https://one.example.com"
