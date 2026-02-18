#!/usr/bin/env python3
"""
Test whether our tqdm monkey-patch actually intercepts HuggingFace Hub downloads.

Run from the voicebox/backend directory:
    python -m tests.test_tqdm_patch
"""

import sys
import os
import logging
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add parent to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_1_hf_tqdm_class_identity():
    """Verify what class huggingface_hub uses for tqdm."""
    from huggingface_hub.utils import tqdm as hf_tqdm
    from huggingface_hub._snapshot_download import hf_tqdm as snapshot_hf_tqdm
    import tqdm as tqdm_module

    print(f"\n=== Test 1: Class identity ===")
    print(f"  hf_tqdm class: {hf_tqdm}")
    print(f"  snapshot hf_tqdm: {snapshot_hf_tqdm}")
    print(f"  same object: {hf_tqdm is snapshot_hf_tqdm}")
    print(f"  MRO: {[c.__name__ for c in hf_tqdm.__mro__]}")
    print(f"  'update' in hf_tqdm.__dict__: {'update' in hf_tqdm.__dict__}")
    print(f"  'update' defined on: {[c.__name__ for c in hf_tqdm.__mro__ if 'update' in c.__dict__]}")
    print(f"  tqdm.tqdm is base: {tqdm_module.tqdm}")


def test_2_patch_intercepts_update():
    """Verify our class-level patch intercepts .update() calls."""
    from huggingface_hub.utils import tqdm as hf_tqdm_class

    print(f"\n=== Test 2: Patch intercepts update ===")

    original_update = hf_tqdm_class.update
    calls = []

    def patched(self, n=1):
        calls.append({"n": n, "self_n": getattr(self, "n", "?"), "total": getattr(self, "total", "?")})
        return original_update(self, n)

    hf_tqdm_class.update = patched

    try:
        # Create instance AFTER patching (like snapshot_download would)
        bar = hf_tqdm_class(total=1000, disable=False)
        bar.update(100)
        bar.update(200)
        bar.close()

        print(f"  calls: {calls}")
        print(f"  bar.n: {bar.n}")
        print(f"  PASS: {'update intercepted' if len(calls) == 2 else 'FAIL - not intercepted'}")
    finally:
        hf_tqdm_class.update = original_update


def test_3_patch_with_our_tracker():
    """Test our actual HFProgressTracker patch_download context manager."""
    from utils.hf_progress import HFProgressTracker, create_hf_progress_callback
    from utils.progress import ProgressManager

    print(f"\n=== Test 3: Our HFProgressTracker ===")

    pm = ProgressManager()
    callback = create_hf_progress_callback("test-model", pm)
    tracker = HFProgressTracker(callback, filter_non_downloads=False)

    with tracker.patch_download():
        # Check if hf_tqdm_class.update was patched
        from huggingface_hub.utils import tqdm as hf_tqdm_class
        print(f"  'update' in hf_tqdm_class.__dict__ after patch: {'update' in hf_tqdm_class.__dict__}")
        print(f"  update is patched: {hf_tqdm_class.update is not tracker._hf_tqdm_original_update}")

        # Simulate what snapshot_download does
        bar = hf_tqdm_class(total=50_000_000, desc="Downloading test...", disable=False)
        bar.update(10_000_000)
        bar.update(20_000_000)
        bar.close()

    progress = pm.get_progress("test-model")
    print(f"  progress: {progress}")
    print(f"  PASS: {'progress tracked' if progress and progress.get('current', 0) > 0 else 'FAIL - no progress'}")


def test_4_real_small_download():
    """Test with a real (tiny) HuggingFace download to see if progress fires."""
    from utils.hf_progress import HFProgressTracker, create_hf_progress_callback
    from utils.progress import ProgressManager

    print(f"\n=== Test 4: Real HF download (tiny model config) ===")

    updates = []

    def capture_callback(downloaded, total, filename):
        updates.append({"downloaded": downloaded, "total": total, "filename": filename})
        if len(updates) <= 3 or len(updates) % 10 == 0:
            print(f"  progress: {downloaded}/{total} ({filename})")

    pm = ProgressManager()
    tracker = HFProgressTracker(capture_callback, filter_non_downloads=False)

    with tracker.patch_download():
        from huggingface_hub import snapshot_download
        import tempfile

        # Download a tiny model (just config files)
        print("  Starting snapshot_download of a tiny repo...")
        try:
            result = snapshot_download(
                "hf-internal-testing/tiny-random-gpt2",
                cache_dir=tempfile.mkdtemp(),
                allow_patterns=["*.json"],
            )
            print(f"  Downloaded to: {result}")
        except Exception as e:
            print(f"  Download error (may be expected): {e}")

    print(f"  Total updates captured: {len(updates)}")
    print(f"  PASS: {'progress captured' if len(updates) > 0 else 'FAIL - no progress updates'}")


def test_5_check_thread_safety():
    """The download runs in a thread via asyncio.to_thread — test that patch works across threads."""
    from huggingface_hub.utils import tqdm as hf_tqdm_class

    print(f"\n=== Test 5: Thread safety ===")

    original_update = hf_tqdm_class.update
    calls = []

    def patched(self, n=1):
        calls.append(threading.current_thread().name)
        return original_update(self, n)

    hf_tqdm_class.update = patched

    try:
        def run_in_thread():
            bar = hf_tqdm_class(total=1000, disable=False)
            bar.update(100)
            bar.close()

        # Patch in main thread
        t = threading.Thread(target=run_in_thread, name="worker-thread")
        t.start()
        t.join()

        print(f"  calls from threads: {calls}")
        print(f"  PASS: {'cross-thread works' if len(calls) > 0 else 'FAIL - not called from thread'}")
    finally:
        hf_tqdm_class.update = original_update


if __name__ == "__main__":
    test_1_hf_tqdm_class_identity()
    test_2_patch_intercepts_update()
    test_3_patch_with_our_tracker()
    test_4_real_small_download()
    test_5_check_thread_safety()
    print("\n=== All tests complete ===")
