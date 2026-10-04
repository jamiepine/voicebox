"""
Regression tests for the VOICEBOX_CLOUD_ENABLED gate.

Voicebox Cloud ships off by default: the /cloud routes stay mounted but answer
404 and never call out to voicebox.sh, and /health reports ``cloud_enabled`` so
the app can hide the Settings → General section. Setting the variable turns
both back on without a rebuild.

Runs without torch: the cloud router is mounted on a bare FastAPI app with the
DB dependency overridden, and the cloud service is stubbed so no network or
browser is touched.

Usage:
    python -m pytest backend/tests/test_cloud_gate.py -v
"""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from backend import config
from backend.database import get_db
from backend.routes import cloud as cloud_routes

CLOUD_ENV_VAR = "VOICEBOX_CLOUD_ENABLED"


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(cloud_routes.router)
    app.dependency_overrides[get_db] = lambda: object()

    calls: list[str] = []

    def _start_login(callback_url, device_name):
        calls.append("start_login")
        return "https://example.invalid/connect"

    async def _handle_callback(db, code, state):
        calls.append("handle_callback")
        return True, "Connected"

    def _get_status(db):
        calls.append("get_status")
        return {
            "connected": False,
            "device_name": None,
            "account_user_id": None,
            "key_prefix": None,
            "connected_at": None,
            "dashboard_url": "https://example.invalid/account",
        }

    monkeypatch.setattr(cloud_routes.cloud_service, "start_login", _start_login)
    monkeypatch.setattr(cloud_routes.cloud_service, "handle_callback", _handle_callback)
    monkeypatch.setattr(cloud_routes.cloud_service, "get_status", _get_status)
    monkeypatch.setattr(cloud_routes.cloud_service, "disconnect", lambda db: calls.append("disconnect"))

    test_client = TestClient(app)
    test_client.service_calls = calls
    return test_client


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "maybe"])
def test_cloud_disabled_by_default_and_for_falsy_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(CLOUD_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(CLOUD_ENV_VAR, value)
    assert config.is_cloud_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_cloud_enabled_for_truthy_values(monkeypatch, value):
    monkeypatch.setenv(CLOUD_ENV_VAR, value)
    assert config.is_cloud_enabled() is True


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/cloud/login/start"),
        ("get", "/cloud/callback?code=abc&state=xyz"),
        ("get", "/cloud/status"),
        ("post", "/cloud/disconnect"),
    ],
)
def test_cloud_routes_return_404_when_disabled(client, monkeypatch, method, path):
    monkeypatch.delenv(CLOUD_ENV_VAR, raising=False)

    response = getattr(client, method)(path)

    assert response.status_code == 404
    assert response.json()["detail"] == cloud_routes.CLOUD_DISABLED_DETAIL
    assert client.service_calls == [], "a disabled backend must never touch the cloud service"


def test_cloud_routes_work_when_enabled(client, monkeypatch):
    monkeypatch.setenv(CLOUD_ENV_VAR, "1")

    assert client.get("/cloud/status").status_code == 200
    assert client.post("/cloud/login/start").json()["authorize_url"] == "https://example.invalid/connect"
    assert client.get("/cloud/callback?code=abc&state=xyz").status_code == 200
    assert client.post("/cloud/disconnect").status_code == 200
    assert client.service_calls == [
        "get_status",
        "start_login",
        "handle_callback",
        "disconnect",
        "get_status",
    ]


def test_flag_is_read_per_request_not_at_import(client, monkeypatch):
    monkeypatch.delenv(CLOUD_ENV_VAR, raising=False)
    assert client.get("/cloud/status").status_code == 404

    monkeypatch.setenv(CLOUD_ENV_VAR, "1")
    assert client.get("/cloud/status").status_code == 200


def test_health_response_reports_cloud_flag():
    from backend import models

    assert models.HealthResponse(status="healthy", model_loaded=False, gpu_available=False).cloud_enabled is False
    assert (
        models.HealthResponse(
            status="healthy", model_loaded=False, gpu_available=False, cloud_enabled=True
        ).cloud_enabled
        is True
    )
