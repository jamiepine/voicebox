"""A download must leave GPU memory as it found it.

``/models/download`` works by loading the model, which used to leave it
resident. Queueing several downloads filled VRAM until later ones failed with
"CUDA out of memory" even though their files had downloaded fine.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import backend.backends as backends_pkg
from backend import models as api_models
from backend.routes import models as models_route
from backend.utils.tasks import get_task_manager


@pytest.fixture
def harness(monkeypatch):
    state = {"loaded": False, "unloaded": 0}
    config = SimpleNamespace(model_name="fake-model", hf_repo_id="org/fake", engine="fake")

    def load():
        state["loaded"] = True

    def unload(_cfg):
        state["unloaded"] += 1
        state["loaded"] = False
        return True

    monkeypatch.setattr(backends_pkg, "get_model_config", lambda name: config)
    monkeypatch.setattr(backends_pkg, "get_model_load_func", lambda cfg: load)
    monkeypatch.setattr(backends_pkg, "check_model_loaded", lambda cfg: state["loaded"])
    monkeypatch.setattr(backends_pkg, "unload_model_by_config", unload)

    scheduled = []
    monkeypatch.setattr(models_route, "create_background_task", lambda coro: scheduled.append(coro))

    tm = get_task_manager()
    tm.clear_all()
    yield state, scheduled, tm
    tm.clear_all()


def _download(scheduled):
    async def run():
        await models_route.trigger_model_download(api_models.ModelDownloadRequest(model_name="fake-model"))
        for coro in scheduled:
            await coro

    asyncio.run(run())


def test_model_loaded_only_for_download_is_unloaded(harness):
    state, scheduled, tm = harness
    _download(scheduled)
    assert state["unloaded"] == 1
    assert state["loaded"] is False
    assert not tm.is_download_active("fake-model")


def test_already_loaded_model_stays_loaded(harness):
    state, scheduled, _ = harness
    state["loaded"] = True
    _download(scheduled)
    assert state["unloaded"] == 0
    assert state["loaded"] is True


def test_model_kept_while_a_generation_is_running(harness):
    state, scheduled, tm = harness
    tm.start_generation("gen-1", "profile", "hello")
    _download(scheduled)
    assert state["unloaded"] == 0
    assert state["loaded"] is True
