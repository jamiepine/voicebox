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

import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hf_offline_patch import force_offline_if_cached  # noqa: E402


def _nest_opposite_modes_in_subprocess(queue):
    """Module-level so it's picklable for multiprocessing's spawn start method.

    Runs in a fresh child process rather than a thread of the test process,
    so a real deadlock here (a regression of the guard this test exists for)
    can't leave the shared _offline_cv state corrupted for every other test
    in this run: the child either exits cleanly (guard raised) or gets
    terminated by the parent after the timeout, and either way the test
    process's own state was never touched.
    """
    try:
        with force_offline_if_cached(False, "outer-uncached"), force_offline_if_cached(
            True, "inner-cached"
        ):
            pass
    except Exception as exc:
        queue.put(("exc", type(exc).__name__, str(exc)))
    else:
        queue.put(("ok", None, None))


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


def test_uncached_load_never_observes_offline_flag_from_concurrent_cached_load():
    """An uncached (network-needing) load must never inherit the forced
    offline mode of a concurrent, unrelated cached load — even when both
    start at nearly the same time.
    """
    original = _hf_const().HF_HUB_OFFLINE
    observations: list[bool] = []
    errors: list[Exception] = []
    cached_entered = threading.Event()

    def cached_load():
        try:
            with force_offline_if_cached(True, "cached"):
                cached_entered.set()
                time.sleep(0.2)
        except Exception as exc:
            errors.append(exc)

    def uncached_load():
        try:
            assert cached_entered.wait(timeout=5), "cached thread never entered"
            with force_offline_if_cached(False, "uncached"):
                observations.append(_hf_const().HF_HUB_OFFLINE)
        except Exception as exc:
            errors.append(exc)

    t_cached = threading.Thread(target=cached_load)
    t_uncached = threading.Thread(target=uncached_load)
    t_cached.start()
    t_uncached.start()
    t_cached.join(timeout=5)
    t_uncached.join(timeout=5)

    assert not t_cached.is_alive(), "cached thread did not finish"
    assert not t_uncached.is_alive(), "uncached thread did not finish"
    assert not errors, errors
    assert observations == [False], "uncached load observed offline mode forced by a concurrent cached load"
    assert original == _hf_const().HF_HUB_OFFLINE


def test_nesting_opposite_mode_on_same_thread_raises_instead_of_deadlocking():
    """Nesting is_cached=True inside is_cached=False (or vice versa) on the
    same thread must raise immediately, not hang: the inner call's wait
    condition can only be cleared by the outer call's own exit, which can
    never run because it's blocked inside the inner call waiting for it.

    Runs in a spawned child process with a bounded join, terminated if it's
    still alive after the timeout, so a regression fails this test instead of
    hanging the suite or leaving _offline_cv's shared state corrupted for
    every other test in this run.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_nest_opposite_modes_in_subprocess, args=(queue,))
    proc.start()
    proc.join(timeout=5)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        pytest.fail(
            "nesting the opposite mode on the same thread hung instead of raising, "
            "this is the deadlock the per-thread mode-stack guard exists to prevent"
        )

    kind, exc_type, exc_msg = queue.get(timeout=2)
    assert kind == "exc", (kind, exc_type, exc_msg)
    assert exc_type == "RuntimeError", (kind, exc_type, exc_msg)


def test_offline_session_does_not_leak_after_exit():
    """Per-thread cached HF sessions must be cleared when exiting offline mode."""
    from huggingface_hub.utils._http import OfflineAdapter, get_session

    with force_offline_if_cached(True, "cached"):
        offline_session = get_session()
        assert isinstance(offline_session.adapters.get("https://"), OfflineAdapter)

    with force_offline_if_cached(False, "uncached"):
        online_session = get_session()
        assert not isinstance(online_session.adapters.get("https://"), OfflineAdapter)
        assert not isinstance(online_session.adapters.get("http://"), OfflineAdapter)


def test_online_session_does_not_leak_into_offline_window():
    """Pre-existing online sessions must be cleared when entering offline mode."""
    from huggingface_hub.utils._http import OfflineAdapter, get_session

    with force_offline_if_cached(False, "uncached"):
        online_session = get_session()
        assert not isinstance(online_session.adapters.get("https://"), OfflineAdapter)

    with force_offline_if_cached(True, "cached"):
        offline_session = get_session()
        assert isinstance(offline_session.adapters.get("https://"), OfflineAdapter)
        assert isinstance(offline_session.adapters.get("http://"), OfflineAdapter)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
