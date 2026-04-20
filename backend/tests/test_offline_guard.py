"""
Unit tests for the ``force_offline_if_cached`` helper.

Verifies that the helper mutates the cached module constants in
``huggingface_hub.constants`` and ``transformers.utils.hub`` — not just
``os.environ`` — and that concurrent users are refcount-coordinated so
one thread's exit can't strip another thread's offline protection.

NOTE: These tests mutate process-global state in ``huggingface_hub.constants``
and ``transformers.utils.hub``. They are not safe under cross-process
parallelism (e.g. ``pytest-xdist`` with ``--dist=loadfile``/``loadscope``);
run this file serially.
"""

import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hf_offline_patch import force_offline_if_cached  # noqa: E402


def _hf_const():
    import huggingface_hub.constants as hf_const

    return hf_const


def _tf_hub():
    import transformers.utils.hub as tf_hub

    return tf_hub


def test_mutates_cached_huggingface_hub_constant():
    original = _hf_const().HF_HUB_OFFLINE
    with force_offline_if_cached(True, "t"):
        assert _hf_const().HF_HUB_OFFLINE is True
    assert original == _hf_const().HF_HUB_OFFLINE


def test_mutates_cached_transformers_constant():
    original = _tf_hub()._is_offline_mode
    with force_offline_if_cached(True, "t"):
        assert _tf_hub()._is_offline_mode is True
    assert original == _tf_hub()._is_offline_mode


def test_sets_env_variable():
    original = os.environ.get("HF_HUB_OFFLINE")
    with force_offline_if_cached(True, "t"):
        assert "1" == os.environ.get("HF_HUB_OFFLINE")
    assert original == os.environ.get("HF_HUB_OFFLINE")


def test_noop_when_not_cached():
    before = _hf_const().HF_HUB_OFFLINE
    with force_offline_if_cached(False, "t"):
        assert before == _hf_const().HF_HUB_OFFLINE


def test_nested_contexts_respect_refcount():
    original = _hf_const().HF_HUB_OFFLINE
    with force_offline_if_cached(True, "outer"):
        assert _hf_const().HF_HUB_OFFLINE is True
        with force_offline_if_cached(True, "inner"):
            assert _hf_const().HF_HUB_OFFLINE is True
        # inner exit must not restore while outer is still active
        assert _hf_const().HF_HUB_OFFLINE is True
    assert original == _hf_const().HF_HUB_OFFLINE


def test_concurrent_threads_share_offline_window():
    """A slow thread must keep seeing offline mode even if a peer exits first."""
    original = _hf_const().HF_HUB_OFFLINE
    observations: list[bool] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(2)
    fast_exited = threading.Event()

    def slow():
        try:
            with force_offline_if_cached(True, "slow"):
                barrier.wait(timeout=5)
                assert fast_exited.wait(timeout=5), "fast thread did not exit"
                observations.append(_hf_const().HF_HUB_OFFLINE)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def fast():
        try:
            with force_offline_if_cached(True, "fast"):
                barrier.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            fast_exited.set()

    t_slow = threading.Thread(target=slow)
    t_fast = threading.Thread(target=fast)
    t_slow.start()
    t_fast.start()
    t_slow.join(timeout=5)
    t_fast.join(timeout=5)

    assert not t_slow.is_alive(), "slow thread did not finish"
    assert not t_fast.is_alive(), "fast thread did not finish"
    assert not errors, errors
    assert observations == [True], "slow thread lost offline protection"
    assert original == _hf_const().HF_HUB_OFFLINE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
