"""
Test script to debug model download progress tracking.
"""

import asyncio
import json
import time
from typing import List, Dict
import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from utils.progress import ProgressManager, get_progress_manager
from utils.hf_progress import HFProgressTracker, create_hf_progress_callback


def test_progress_manager_basic():
    """Test 1: Basic ProgressManager functionality."""
    print("\n" + "=" * 60)
    print("Test 1: ProgressManager Basic Operations")
    print("=" * 60)

    pm = ProgressManager()

    # Test update_progress
    pm.update_progress(
        model_name="test-model",
        current=50,
        total=100,
        filename="test.bin",
        status="downloading"
    )

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
    return True


async def test_progress_manager_sse():
    """Test 2: ProgressManager SSE streaming."""
    print("\n" + "=" * 60)
    print("Test 2: ProgressManager SSE Streaming")
    print("=" * 60)

    pm = ProgressManager()
    collected_events: List[Dict] = []

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
                status="downloading" if i < 100 else "downloading"
            )
            await asyncio.sleep(0.1)

        # Mark complete
        print("  Backend: Marking download complete")
        pm.mark_complete("test-model-sse")

    # Run SSE client and download simulation concurrently
    await asyncio.gather(
        sse_client(),
        simulate_download()
    )

    # Verify we got events
    print(f"\n  Collected {len(collected_events)} events")
    assert len(collected_events) > 0, "Should have received at least one event"
    assert collected_events[-1]["status"] == "complete", "Last event should be 'complete'"

    print("✓ Test 2 PASSED\n")
    return True


def test_hf_progress_tracker():
    """Test 3: HFProgressTracker tqdm patching."""
    print("\n" + "=" * 60)
    print("Test 3: HFProgressTracker tqdm Patching")
    print("=" * 60)

    captured_progress: List[tuple] = []

    def progress_callback(downloaded: int, total: int, filename: str):
        """Capture progress updates."""
        captured_progress.append((downloaded, total, filename))
        print(f"  Progress callback: {downloaded}/{total} bytes ({filename})")

    tracker = HFProgressTracker(progress_callback)

    # Simulate a download with tqdm
    with tracker.patch_download():
        try:
            from tqdm import tqdm

            # Simulate downloading a file
            print("  Simulating download with tqdm...")
            total_size = 1000
            with tqdm(total=total_size, desc="model.bin", unit="B", unit_scale=True) as pbar:
                for chunk in range(0, total_size, 100):
                    pbar.update(100)
                    time.sleep(0.01)

            print(f"  Captured {len(captured_progress)} progress updates")
            assert len(captured_progress) > 0, "Should have captured progress updates"

            # Verify progress increases
            last_downloaded = 0
            for downloaded, total, filename in captured_progress:
                assert downloaded >= last_downloaded, "Downloaded bytes should increase"
                assert total == total_size, "Total should be consistent"
                last_downloaded = downloaded

            print("✓ Test 3 PASSED\n")
            return True

        except ImportError:
            print("✗ tqdm not available, skipping test\n")
            return None


async def test_full_integration():
    """Test 4: Full integration test."""
    print("\n" + "=" * 60)
    print("Test 4: Full Integration (ProgressManager + HFProgressTracker)")
    print("=" * 60)

    pm = get_progress_manager()
    collected_events: List[Dict] = []

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
        pm.update_progress(
            model_name="integration-test",
            current=0,
            total=1,
            filename="",
            status="downloading"
        )

        # Simulate download with tqdm patching
        with tracker.patch_download():
            try:
                from tqdm import tqdm

                # Simulate multi-file download (like HuggingFace does)
                files = [
                    ("model.safetensors", 5000),
                    ("config.json", 1000),
                    ("tokenizer.json", 500),
                ]

                for filename, size in files:
                    print(f"  Backend: Downloading {filename}...")
                    with tqdm(total=size, desc=filename, unit="B") as pbar:
                        for chunk in range(0, size, 500):
                            chunk_size = min(500, size - chunk)
                            pbar.update(chunk_size)
                            await asyncio.sleep(0.05)

                # Mark complete
                print("  Backend: Download complete")
                pm.mark_complete("integration-test")

            except ImportError:
                print("  ✗ tqdm not available")
                pm.mark_error("integration-test", "tqdm not available")

    # Run both
    await asyncio.gather(
        sse_client(),
        simulate_real_download()
    )

    # Verify
    print(f"\n  Collected {len(collected_events)} events")
    if len(collected_events) > 0:
        print(f"  First event: {collected_events[0]}")
        print(f"  Last event: {collected_events[-1]}")
        assert collected_events[-1]["status"] == "complete", "Should end with 'complete'"
        print("✓ Test 4 PASSED\n")
        return True
    else:
        print("✗ Test 4 FAILED - No events received\n")
        return False


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
