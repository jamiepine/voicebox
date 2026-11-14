"""Concurrent model loads must not overlap.

Starting a second download while another model was loading (e.g. Whisper
while Qwen TTS was downloading) produced two failures:

* transformers/accelerate build models inside ``init_empty_weights()``, which
  patches ``torch.nn.Module.register_parameter`` for every thread, so the
  other model's layers came out on ``meta`` and its load failed with
  "Cannot copy out of meta tensor; no data!" -- until it was retried.
* both downloads patched the shared tqdm classes, so each model's progress
  bar received the other's bytes and jumped back and forth (issue #708).

``model_load_progress`` now serializes loads process-wide.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.backends import base
from backend.backends.base import WAITING_FOR_LOAD_MESSAGE, model_load_progress
from backend.utils.progress import get_progress_manager


def test_concurrent_loads_do_not_overlap():
    active = 0
    max_active = 0
    lock = threading.Lock()
    first_inside = threading.Event()

    def load(name):
        nonlocal active, max_active
        with model_load_progress(name, is_cached=False):
            with lock:
                active += 1
                max_active = max(max_active, active)
            first_inside.set()
            time.sleep(0.3)
            with lock:
                active -= 1

    a = threading.Thread(target=load, args=("serial-test-a",))
    a.start()
    assert first_inside.wait(5)
    b = threading.Thread(target=load, args=("serial-test-b",))
    b.start()

    # While A holds the load, B must be reported as waiting, not stuck.
    deadline = time.time() + 2
    waiting = None
    while time.time() < deadline:
        waiting = get_progress_manager().get_progress("serial-test-b")
        if waiting and waiting.get("filename") == WAITING_FOR_LOAD_MESSAGE:
            break
        time.sleep(0.02)
    assert waiting and waiting.get("filename") == WAITING_FOR_LOAD_MESSAGE

    a.join(5)
    b.join(5)
    assert max_active == 1


def test_lock_released_after_failed_load():
    try:
        with model_load_progress("serial-test-fail", is_cached=False):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    acquired = base._MODEL_LOAD_LOCK.acquire(timeout=1)
    assert acquired
    base._MODEL_LOAD_LOCK.release()


def test_nested_load_on_same_thread_does_not_deadlock():
    with model_load_progress("serial-test-outer", is_cached=False):
        with model_load_progress("serial-test-inner", is_cached=False):
            pass
