"""Offline mode must not outlive its window on reused threads.

huggingface_hub caches one HTTP session per thread; a session created while
HF_HUB_OFFLINE was forced refuses every request. Model loads run on reused
worker threads, so a download that later landed on a thread where a cached
model had loaded failed with "outgoing traffic has been disabled".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from huggingface_hub.utils import get_session
from huggingface_hub.utils._http import OfflineAdapter

from backend.utils.hf_offline_patch import force_offline_if_cached


def _is_offline_session(session) -> bool:
    return isinstance(session.get_adapter("https://huggingface.co"), OfflineAdapter)


def test_session_is_online_again_after_offline_window():
    assert not _is_offline_session(get_session())
    with force_offline_if_cached(True, "cached-model"):
        assert _is_offline_session(get_session())
    # Same thread, window closed: an uncached download must reach the network.
    with force_offline_if_cached(False, "uncached-model"):
        assert not _is_offline_session(get_session())


def test_tada_3b_sharded_weights_count_as_cached(tmp_path, monkeypatch):
    from huggingface_hub import constants

    from backend.backends.hume_backend import TADA_3B_ML_REPO, _TADA_MODEL_WEIGHT_FILES
    from backend.backends.base import is_model_cached

    snap = tmp_path / ("models--" + TADA_3B_ML_REPO.replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    for name in ("config.json", "model.safetensors.index.json",
                 "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (snap / name).write_bytes(b"x")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

    assert is_model_cached(TADA_3B_ML_REPO, required_files=_TADA_MODEL_WEIGHT_FILES)
    assert is_model_cached(TADA_3B_ML_REPO)
