"""Kokoro fetches each voice's style vector lazily; make sure that fetch can reach the Hub.

No network and no model: huggingface_hub is stubbed.
"""

import huggingface_hub
import huggingface_hub.utils
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from backend.backends.kokoro_backend import KokoroTTSBackend
from backend.utils import hf_offline_patch


@pytest.fixture
def hub(monkeypatch):
    """Stub the Hub: `cached` is the set of files already on disk; `events` is the call log."""
    state = {"cached": set(), "events": []}

    def fake_download(repo_id, filename, local_files_only=False, **_kw):
        if local_files_only:
            if filename not in state["cached"]:
                raise LocalEntryNotFoundError("not cached")
            return f"/cache/{filename}"
        # A real download is only allowed while an uncached (network) window is open.
        state["events"].append(("download", filename, hf_offline_patch._uncached_active))
        state["cached"].add(filename)
        return f"/cache/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        huggingface_hub.utils, "reset_sessions", lambda: state["events"].append(("reset",))
    )
    return state


def test_cached_voice_makes_no_network_call(hub):
    hub["cached"].add("voices/am_echo.pt")

    KokoroTTSBackend._ensure_voice_cached("am_echo")

    assert hub["events"] == []


def test_new_voice_is_downloaded_in_a_network_window_with_a_fresh_session(hub):
    KokoroTTSBackend._ensure_voice_cached("af_bella")

    # Sessions are reset first, so one built during an offline window is not reused,
    # and the download runs while an uncached window is open.
    assert hub["events"] == [("reset",), ("download", "voices/af_bella.pt", 1)]
    assert hf_offline_patch._uncached_active == 0


def test_voice_is_only_downloaded_once(hub):
    KokoroTTSBackend._ensure_voice_cached("af_bella")
    KokoroTTSBackend._ensure_voice_cached("af_bella")

    assert [e for e in hub["events"] if e[0] == "download"] == [("download", "voices/af_bella.pt", 1)]
