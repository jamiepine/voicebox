"""OmniVoice is refused on Intel macOS.

PyTorch's last x86_64 macOS wheel is 2.2.2, and the vendored Higgs Audio V2
codec needs torch.nn.Buffer (torch 2.5+), so OmniVoice cannot load there.
Generation, model load and model download must answer with a 400 and a short
reason, and /models/status must mark the model unavailable so the UI can
disable it.

Usage:
    python -m pytest backend/tests/test_omnivoice_intel_mac.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import backends
from backend.database import get_db
from backend.routes.generations import router as generations_router
from backend.routes.models import router as models_router
from backend.services import profiles
from backend.utils.tasks import get_task_manager

INTEL_MAC = ("Darwin", "x86_64")
APPLE_SILICON = ("Darwin", "arm64")
LINUX = ("Linux", "x86_64")


def _set_platform(monkeypatch, system, machine):
    monkeypatch.setattr("backend.utils.platform_detect.platform.system", lambda: system)
    monkeypatch.setattr("backend.utils.platform_detect.platform.machine", lambda: machine)


@pytest.fixture
def intel_mac(monkeypatch):
    _set_platform(monkeypatch, *INTEL_MAC)


@pytest.fixture
def fake_loads(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    calls = []

    def fake_load_func(config):
        async def _load():
            calls.append(config.model_name)

        return _load

    monkeypatch.setattr(backends, "get_model_load_func", fake_load_func)
    monkeypatch.setattr(backends, "check_model_loaded", lambda config: False)
    return calls


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(models_router)
    app.include_router(generations_router)
    app.dependency_overrides[get_db] = lambda: None
    with TestClient(app) as test_client:
        yield test_client


def _cloned_profile():
    return SimpleNamespace(id="p1", voice_type="cloned", default_engine="omnivoice", preset_engine=None)


@pytest.mark.parametrize(
    ("plat", "expected"),
    [
        (INTEL_MAC, backends.OMNIVOICE_INTEL_MAC_REASON),
        (APPLE_SILICON, None),
        (LINUX, None),
    ],
)
def test_engine_unavailable_reason(monkeypatch, plat, expected):
    _set_platform(monkeypatch, *plat)
    assert backends.engine_unavailable_reason("omnivoice") == expected
    assert backends.engine_unavailable_reason("qwen") is None


def test_reason_names_the_torch_floor():
    assert "PyTorch 2.5" in backends.OMNIVOICE_INTEL_MAC_REASON
    assert "Intel Mac" in backends.OMNIVOICE_INTEL_MAC_REASON


@pytest.mark.parametrize("plat", [APPLE_SILICON, LINUX])
def test_validate_profile_engine_allows_omnivoice_elsewhere(monkeypatch, plat):
    _set_platform(monkeypatch, *plat)
    profiles.validate_profile_engine(_cloned_profile(), "omnivoice")


def test_validate_profile_engine_refuses_omnivoice_on_intel_mac(intel_mac):
    with pytest.raises(ValueError, match="Intel Macs"):
        profiles.validate_profile_engine(_cloned_profile(), "omnivoice")
    profiles.validate_profile_engine(_cloned_profile(), "qwen")


@pytest.mark.parametrize("path", ["/generate", "/generate/stream"])
def test_generate_refuses_omnivoice_on_intel_mac(intel_mac, monkeypatch, client, path):
    async def fake_get_profile(profile_id, db):
        return _cloned_profile()

    monkeypatch.setattr(profiles, "get_profile", fake_get_profile)

    response = client.post(path, json={"profile_id": "p1", "text": "hello", "engine": "omnivoice"})

    assert response.status_code == 400
    assert response.json()["detail"] == backends.OMNIVOICE_INTEL_MAC_REASON


def test_load_refuses_omnivoice_on_intel_mac(intel_mac, fake_loads, client):
    response = client.post("/models/load", json={"model_name": "omnivoice"})

    assert response.status_code == 400
    assert response.json()["detail"] == backends.OMNIVOICE_INTEL_MAC_REASON
    assert fake_loads == []


def test_load_other_models_still_works_on_intel_mac(intel_mac, fake_loads, client):
    response = client.post("/models/load", json={"model_name": "kokoro"})

    assert response.status_code == 200
    assert fake_loads == ["kokoro"]


def test_download_refuses_omnivoice_on_intel_mac(intel_mac, fake_loads, client):
    response = client.post("/models/download", json={"model_name": "omnivoice"})

    assert response.status_code == 400
    assert response.json()["detail"] == backends.OMNIVOICE_INTEL_MAC_REASON
    assert fake_loads == []
    assert "omnivoice" not in {t.model_name for t in get_task_manager().get_active_downloads()}


@pytest.mark.parametrize("plat", [APPLE_SILICON, LINUX])
def test_download_omnivoice_allowed_elsewhere(monkeypatch, fake_loads, client, plat):
    _set_platform(monkeypatch, *plat)

    async def fake_track(config, load, *, force=False):
        await load()

    monkeypatch.setattr(backends, "track_model_download", fake_track)

    response = client.post("/models/download", json={"model_name": "omnivoice"})

    assert response.status_code == 200
    get_task_manager().cancel_download("omnivoice")


@pytest.mark.parametrize(("plat", "available"), [(INTEL_MAC, False), (APPLE_SILICON, True), (LINUX, True)])
def test_model_status_reports_omnivoice_availability(monkeypatch, fake_loads, client, plat, available):
    _set_platform(monkeypatch, *plat)

    response = client.get("/models/status")

    assert response.status_code == 200
    statuses = {m["model_name"]: m for m in response.json()["models"]}
    omnivoice = statuses["omnivoice"]
    assert omnivoice["engine"] == "omnivoice"
    assert omnivoice["available"] is available
    assert omnivoice["unavailable_reason"] == (None if available else backends.OMNIVOICE_INTEL_MAC_REASON)
    others = [m for name, m in statuses.items() if name != "omnivoice"]
    assert others
    assert all(m["available"] for m in others)
    assert all(m["unavailable_reason"] is None for m in others)
