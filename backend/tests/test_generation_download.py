"""
Test TTS generation with SSE progress monitoring.
This test captures the exact SSE events triggered during generation
to identify UX issues where users see download progress even when
the model is already cached.
"""

import asyncio
import json
import httpx
from typing import List, Dict, Optional
from datetime import datetime


async def monitor_sse_stream(model_name: str, timeout: int = 120):
    """Monitor SSE stream for a model during generation."""
    events: List[Dict] = []
    url = f"http://localhost:8000/models/progress/{model_name}"

    print(f"[{_timestamp()}] Connecting to SSE endpoint: {url}")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url) as response:
                print(f"[{_timestamp()}] SSE connected, status: {response.status_code}")

                if response.status_code != 200:
                    print(f"[{_timestamp()}] Error: SSE endpoint returned {response.status_code}")
                    return events

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    timestamp = _timestamp()

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            print(f"[{timestamp}] → SSE Event: {data['status']:12} {data.get('progress', 0):6.1f}% {data.get('filename', '')}")
                            events.append({
                                **data,
                                "_timestamp": timestamp
                            })

                            # Stop if complete or error
                            if data.get("status") in ("complete", "error"):
                                print(f"[{timestamp}] → Model {data['status']}!")
                                break

                        except json.JSONDecodeError as e:
                            print(f"[{timestamp}] Error parsing JSON: {e}")
                            print(f"  Line was: {line}")

                    elif line.startswith(": heartbeat"):
                        print(f"[{timestamp}] ♥ heartbeat")

    except asyncio.TimeoutError:
        print(f"[{_timestamp()}] SSE monitoring timed out")
    except Exception as e:
        print(f"[{_timestamp()}] SSE error: {e}")

    return events


async def trigger_generation(profile_id: str, text: str, model_size: str = "1.7B"):
    """Trigger TTS generation via the API."""
    url = "http://localhost:8000/generate"

    print(f"\n[{_timestamp()}] Triggering generation...")
    print(f"  Profile: {profile_id}")
    print(f"  Text: {text[:50]}...")
    print(f"  Model: {model_size}")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, json={
                "profile_id": profile_id,
                "text": text,
                "language": "en",
                "model_size": model_size,
            })

            print(f"[{_timestamp()}] Response: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"[{_timestamp()}] ✓ Generation successful!")
                print(f"  Generation ID: {result.get('id')}")
                print(f"  Duration: {result.get('duration', 0):.2f}s")
                return True, result
            elif response.status_code == 202:
                # Model is being downloaded
                result = response.json()
                print(f"[{_timestamp()}] → Model download in progress")
                print(f"  Detail: {result}")
                return False, result
            else:
                print(f"[{_timestamp()}] ✗ Error: {response.text}")
                return False, None

    except Exception as e:
        print(f"[{_timestamp()}] ✗ Exception: {e}")
        return False, None


async def get_first_profile():
    """Get the first available voice profile."""
    url = "http://localhost:8000/profiles"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            if response.status_code == 200:
                profiles = response.json()
                if profiles:
                    return profiles[0]["id"]
    except Exception as e:
        print(f"Error getting profiles: {e}")

    return None


async def check_server():
    """Check if the server is running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code == 200
    except Exception as e:
        print(f"Server not running: {e}")
        return False


def _timestamp():
    """Get current timestamp for logging."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


async def test_generation_with_cached_model():
    """
    Test Case 1: Generation when model is already cached.
    
    This should NOT show any download progress events.
    If it does, that's the UX bug we're trying to fix.
    """
    print("\n" + "=" * 80)
    print("TEST CASE 1: Generation with Cached Model")
    print("=" * 80)
    print("Expected: No download progress events (or minimal/instant completion)")
    print("Actual UX Issue: Users see 'started' and 'finished' events even for cached models")
    print("=" * 80)

    model_size = "1.7B"
    model_name = f"qwen-tts-{model_size}"

    # Get a profile
    profile_id = await get_first_profile()
    if not profile_id:
        print("✗ No voice profiles found. Please create a profile first.")
        return False

    print(f"\nUsing profile: {profile_id}")

    # Start SSE monitor BEFORE triggering generation
    monitor_task = asyncio.create_task(monitor_sse_stream(model_name, timeout=30))

    # Wait for SSE to connect
    await asyncio.sleep(1)

    # Trigger generation
    test_text = "Hello, this is a test of the voice generation system."
    success, result = await trigger_generation(profile_id, test_text, model_size)

    if not success and result and result.get("downloading"):
        print("\n⚠ Model is being downloaded. Waiting for download to complete...")
        # Wait for SSE monitor to capture download events
        events = await monitor_task
        return events

    # Wait a bit more to catch any progress events
    await asyncio.sleep(3)

    # Cancel SSE monitor
    monitor_task.cancel()
    try:
        events = await monitor_task
    except asyncio.CancelledError:
        events = []

    return events


async def test_generation_with_fresh_download():
    """
    Test Case 2: Generation when model needs to be downloaded.
    
    This SHOULD show download progress events.
    """
    print("\n" + "=" * 80)
    print("TEST CASE 2: Generation with Model Download")
    print("=" * 80)
    print("Expected: Download progress events from 0% to 100%")
    print("=" * 80)

    # Use a different model size to force download
    model_size = "0.6B"  # Smaller model for faster testing
    model_name = f"qwen-tts-{model_size}"

    # Get a profile
    profile_id = await get_first_profile()
    if not profile_id:
        print("✗ No voice profiles found. Please create a profile first.")
        return False

    print(f"\nUsing profile: {profile_id}")
    print("Note: This will download the model if not cached")

    # Start SSE monitor BEFORE triggering generation
    monitor_task = asyncio.create_task(monitor_sse_stream(model_name, timeout=300))

    # Wait for SSE to connect
    await asyncio.sleep(1)

    # Trigger generation
    test_text = "This should trigger a model download if the model is not cached."
    success, result = await trigger_generation(profile_id, test_text, model_size)

    if not success and result and result.get("downloading"):
        print("\n→ Model download initiated. Monitoring progress...")
        # Wait for download to complete
        events = await monitor_task

        # Try generation again
        print(f"\n[{_timestamp()}] Retrying generation after download...")
        await asyncio.sleep(2)
        success, result = await trigger_generation(profile_id, test_text, model_size)

        if success:
            print("✓ Generation successful after download")

        return events

    # If model was already cached
    await asyncio.sleep(3)
    monitor_task.cancel()
    try:
        events = await monitor_task
    except asyncio.CancelledError:
        events = []

    return events


async def main():
    print("=" * 80)
    print("TTS Generation Progress Test")
    print("=" * 80)
    print("Purpose: Capture exact SSE events during generation to identify UX issues")
    print("=" * 80)

    # Check if server is running
    print(f"\n[{_timestamp()}] Checking if server is running...")
    if not await check_server():
        print("✗ Server is not running on http://localhost:8000")
        print("\nPlease start the server first:")
        print("  cd backend && python main.py")
        return False

    print("✓ Server is running")

    # Test Case 1: Cached model
    print("\n" + "🧪 " * 20)
    events_cached = await test_generation_with_cached_model()

    # Results for Test Case 1
    print("\n" + "=" * 80)
    print("TEST CASE 1 RESULTS: Generation with Cached Model")
    print("=" * 80)

    if not events_cached:
        print("✓ GOOD: No SSE progress events received")
        print("  This is the expected behavior for a cached model.")
    else:
        print(f"⚠ ISSUE FOUND: Received {len(events_cached)} SSE events:")
        print("\nEvent Timeline:")
        for i, event in enumerate(events_cached, 1):
            timestamp = event.pop("_timestamp", "??:??:??.???")
            print(f"  {i}. [{timestamp}] {event}")

        print("\n⚠ This explains the UX issue!")
        print("  Users see progress events even when the model is already cached,")
        print("  making them think the model is downloading again.")

    # Test Case 2: Fresh download (optional, commented out by default)
    # Uncomment if you want to test download progress
    # print("\n" + "🧪 " * 20)
    # events_download = await test_generation_with_fresh_download()
    #
    # print("\n" + "=" * 80)
    # print("TEST CASE 2 RESULTS: Generation with Model Download")
    # print("=" * 80)
    #
    # if not events_download:
    #     print("ℹ Model was already cached, no download occurred")
    # else:
    #     print(f"✓ Received {len(events_download)} download progress events")
    #     print("\nDownload Timeline:")
    #     for i, event in enumerate(events_download, 1):
    #         timestamp = event.pop("_timestamp", "??:??:??.???")
    #         print(f"  {i}. [{timestamp}] {event}")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    asyncio.run(main())
