"""
Test script to debug model download progress tracking.
"""

import asyncio
import json
import logging
import time

import pytest

from backend.utils.hf_progress import MIN_TOTAL_BYTES, HFProgressTracker, create_hf_progress_callback
from backend.utils.progress import ProgressManager, get_progress_manager

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def test_progress_manager_basic():
    """Test 1: Basic ProgressManager functionality."""
    print("\n" + "=" * 60)
    print("Test 1: ProgressManager Basic Operations")
    print("=" * 60)

    pm = ProgressManager()

    # Test update_progress
    pm.update_progress(model_name="test-model", current=50, total=100, filename="test.bin", status="downloading")

    # Test get_progress
    progress = pm.get_progress("test-model")
    print(f"✓ Progress stored: {progress}")
    assert progress is not None
    assert progress["progress"] == 50.0
    assert progress["filename"] == "test.bin"
    assert progress["status"] == "downloading"

    # Test mark_complete
    pm.mark_complete("test-model")
    progress = pm.get_progress("test-model")
    print(f"✓ Marked complete: {progress}")
    assert progress["status"] == "complete"
    assert progress["progress"] == 100.0

    print("✓ Test 1 PASSED\n")


async def test_progress_manager_sse():
    """Test 2: ProgressManager SSE streaming."""
    print("\n" + "=" * 60)
    print("Test 2: ProgressManager SSE Streaming")
    print("=" * 60)

    pm = ProgressManager()
    collected_events: list[dict] = []

    # Simulate SSE client
    async def sse_client():
        """Simulates a frontend SSE connection."""
        print("  SSE client: Subscribing to test-model-sse...")
        async for event in pm.subscribe("test-model-sse"):
            # Parse SSE event
            if event.startswith("data: "):
                data = json.loads(event[6:])
                print(f"  SSE client: Received event: {data['status']} - {data.get('progress', 0):.1f}%")
                collected_events.append(data)

                # Stop when complete
                if data.get("status") in ("complete", "error"):
                    break
            elif event.startswith(": heartbeat"):
                print("  SSE client: Received heartbeat")

    # Simulate download progress updates (from backend thread)
    async def simulate_download():
        """Simulates backend sending progress updates."""
        print("  Backend: Starting simulated download...")
        await asyncio.sleep(0.2)  # Let SSE client subscribe first

        # Send progress updates
        for i in range(0, 101, 20):
            print(f"  Backend: Updating progress to {i}%")
            pm.update_progress(
                model_name="test-model-sse",
                current=i,
                total=100,
                filename=f"file_{i}.bin",
                status="downloading",
            )
            await asyncio.sleep(0.1)

        # Mark complete
        print("  Backend: Marking download complete")
        pm.mark_complete("test-model-sse")

    # Run SSE client and download simulation concurrently
    await asyncio.gather(sse_client(), simulate_download())

    # Verify we got events
    print(f"\n  Collected {len(collected_events)} events")
    assert len(collected_events) > 0, "Should have received at least one event"
    assert collected_events[-1]["status"] == "complete", "Last event should be 'complete'"

    print("✓ Test 2 PASSED\n")


def test_hf_progress_tracker():
    """Test 3: HFProgressTracker tqdm patching."""
    print("\n" + "=" * 60)
    print("Test 3: HFProgressTracker tqdm Patching")
    print("=" * 60)

    captured_progress: list[tuple] = []

    def progress_callback(downloaded: int, total: int, filename: str):
        """Capture progress updates."""
        captured_progress.append((downloaded, total, filename))
        print(f"  Progress callback: {downloaded}/{total} bytes ({filename})")

    tracker = HFProgressTracker(progress_callback)

    # The tracker works by subclassing tqdm, so there is nothing to exercise
    # without it. Skip rather than swallow the ImportError: this test asserts
    # nothing on that path, so catching it turned a missing dependency into a
    # silent pass.
    tqdm_module = pytest.importorskip("tqdm")

    # Simulate a download with tqdm
    with tracker.patch_download():
        # Resolved inside the block on purpose: patch_download() rebinds
        # tqdm.tqdm to the tracking subclass, so a reference taken before it
        # is the untracked original and nothing reaches the callback.
        tqdm = tqdm_module.tqdm

        # Simulate downloading a file
        print("  Simulating download with tqdm...")
        # Above MIN_TOTAL_BYTES: the tracker drops progress under that, so a
        # smaller transfer reports nothing and the assertion below can never
        # hold. Derived from the constant so the two cannot drift apart.
        total_size = 2 * MIN_TOTAL_BYTES
        chunk_size = total_size // 10
        with tqdm(total=total_size, desc="model.bin", unit="B", unit_scale=True) as pbar:
            for _chunk in range(0, total_size, chunk_size):
                pbar.update(chunk_size)
                time.sleep(0.01)

        print(f"  Captured {len(captured_progress)} progress updates")
        assert len(captured_progress) > 0, "Should have captured progress updates"

        # Verify progress increases
        last_downloaded = 0
        for downloaded, total, _filename in captured_progress:
            assert downloaded >= last_downloaded, "Downloaded bytes should increase"
            assert total == total_size, "Total should be consistent"
            last_downloaded = downloaded

        print("✓ Test 3 PASSED\n")


async def test_full_integration():
    """Test 4: Full integration test."""
    print("\n" + "=" * 60)
    print("Test 4: Full Integration (ProgressManager + HFProgressTracker)")
    print("=" * 60)

    # Same as test 3 -- without tqdm there is no download to track. The old
    # handler marked the transfer "error" and then asserted "complete", so a
    # missing dependency surfaced as a failed assertion rather than a skip.
    tqdm_module = pytest.importorskip("tqdm")

    pm = get_progress_manager()
    collected_events: list[dict] = []

    # SSE client
    async def sse_client():
        print("  SSE client: Subscribing...")
        async for event in pm.subscribe("integration-test"):
            if event.startswith("data: "):
                data = json.loads(event[6:])
                print(f"  SSE client: {data['status']} - {data.get('progress', 0):.1f}% - {data.get('filename', '')}")
                collected_events.append(data)
                if data.get("status") in ("complete", "error"):
                    break

    # Simulate backend download with HFProgressTracker
    async def simulate_real_download():
        await asyncio.sleep(0.2)  # Let SSE subscribe

        print("  Backend: Starting download with HFProgressTracker...")

        # Set up tracking (like the real backend does)
        progress_callback = create_hf_progress_callback("integration-test", pm)
        tracker = HFProgressTracker(progress_callback)

        # Initialize progress
        pm.update_progress(model_name="integration-test", current=0, total=1, filename="", status="downloading")

        # Simulate download with tqdm patching
        with tracker.patch_download():
            # Resolved inside the block -- see test 3.
            tqdm = tqdm_module.tqdm

            # Simulate multi-file download (like HuggingFace does)
            # Sized above MIN_TOTAL_BYTES for the same reason as test 3 --
            # below it the tracker stays silent and this exercises only
            # mark_complete(), not the integration it is named for. Chunk
            # counts are unchanged, so the runtime is too.
            files = [
                ("model.safetensors", 10 * MIN_TOTAL_BYTES),
                ("config.json", 2 * MIN_TOTAL_BYTES),
                ("tokenizer.json", MIN_TOTAL_BYTES),
            ]
            step = MIN_TOTAL_BYTES

            for filename, size in files:
                print(f"  Backend: Downloading {filename}...")
                with tqdm(total=size, desc=filename, unit="B") as pbar:
                    for chunk in range(0, size, step):
                        pbar.update(min(step, size - chunk))
                        await asyncio.sleep(0.05)

            # Mark complete
            print("  Backend: Download complete")
            pm.mark_complete("integration-test")

    # Run both
    await asyncio.gather(sse_client(), simulate_real_download())

    # Verify
    print(f"\n  Collected {len(collected_events)} events")
    # Asserted, not branched on: the old "no events" path printed FAILED and
    # returned False, which pytest reports as a pass.
    assert collected_events, "Should have received at least one event"
    print(f"  First event: {collected_events[0]}")
    print(f"  Last event: {collected_events[-1]}")
    assert collected_events[-1]["status"] == "complete", "Should end with 'complete'"
    print("✓ Test 4 PASSED\n")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Voicebox Progress Tracking Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Basic operations
    try:
        results.append(("Basic Operations", test_progress_manager_basic()))
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}\n")
        results.append(("Basic Operations", False))

    # Test 2: SSE streaming
    try:
        results.append(("SSE Streaming", await test_progress_manager_sse()))
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}\n")
        results.append(("SSE Streaming", False))

    # Test 3: tqdm patching
    try:
        results.append(("tqdm Patching", test_hf_progress_tracker()))
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}\n")
        results.append(("tqdm Patching", False))

    # Test 4: Full integration
    try:
        results.append(("Full Integration", await test_full_integration()))
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}\n")
        results.append(("Full Integration", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else ("⊘ SKIP" if result is None else "✗ FAIL")
        print(f"  {status:8} {name}")

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    print()
    print(f"  Total: {len(results)} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
