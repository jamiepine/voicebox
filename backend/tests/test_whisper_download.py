"""
Test real model download with SSE progress monitoring.
"""

import asyncio
import json
import httpx
import time
from typing import List, Dict

async def monitor_sse_stream(model_name: str, timeout: int = 300):
    """Monitor SSE stream for a model download."""
    events: List[Dict] = []
    url = f"http://localhost:8000/models/progress/{model_name}"

    print(f"Connecting to SSE endpoint: {url}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url) as response:
            print(f"SSE connected, status: {response.status_code}")

            if response.status_code != 200:
                print(f"Error: SSE endpoint returned {response.status_code}")
                return events

            async for line in response.aiter_lines():
                if not line:
                    continue

                print(f"  Raw SSE: {line[:100]}...")  # Print first 100 chars

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        print(f"  → {data['status']:12} {data.get('progress', 0):6.1f}% {data.get('filename', '')}")
                        events.append(data)

                        # Stop if complete or error
                        if data.get("status") in ("complete", "error"):
                            print(f"  Download {data['status']}!")
                            break

                    except json.JSONDecodeError as e:
                        print(f"  Error parsing JSON: {e}")
                        print(f"  Line was: {line}")

                elif line.startswith(": heartbeat"):
                    print("  ♥ heartbeat")

    return events


async def trigger_download(model_name: str):
    """Trigger a model download via the API."""
    url = "http://localhost:8000/models/download"

    print(f"\nTriggering download for: {model_name}")

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(url, json={"model_name": model_name})
        print(f"Response: {response.status_code} - {response.json()}")
        return response.status_code == 200


async def check_server():
    """Check if the server is running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code == 200
    except Exception as e:
        print(f"Server not running: {e}")
        return False


async def main():
    print("=" * 60)
    print("Real Model Download Progress Test")
    print("=" * 60)

    # Check if server is running
    print("\nChecking if server is running...")
    if not await check_server():
        print("✗ Server is not running on http://localhost:8000")
        print("\nPlease start the server first:")
        print("  cd backend && python main.py")
        return False

    print("✓ Server is running")

    # Choose a small model for testing
    model_name = "whisper-base"  # ~150MB, faster to download
    print(f"\nUsing model: {model_name}")

    # Option to delete model first if it exists
    print("\nDo you want to delete the model first to force a fresh download? (y/n)")
    # For automated testing, skip deletion prompt
    # delete_first = input().strip().lower() == 'y'
    delete_first = False

    if delete_first:
        print(f"Deleting {model_name}...")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(f"http://localhost:8000/models/{model_name}")
            print(f"Delete response: {response.status_code}")

    print("\n" + "=" * 60)
    print("Starting Test")
    print("=" * 60)

    # Start monitoring SSE stream BEFORE triggering download
    async def run_test():
        # Start SSE monitor in background
        monitor_task = asyncio.create_task(monitor_sse_stream(model_name))

        # Wait a bit to ensure SSE is connected
        await asyncio.sleep(1)

        # Trigger download
        success = await trigger_download(model_name)

        if not success:
            print("✗ Failed to trigger download")
            monitor_task.cancel()
            return False

        # Wait for SSE monitor to complete
        events = await monitor_task

        return events

    events = await run_test()

    # Results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    if not events:
        print("✗ FAILED - No SSE events received!")
        print("\nPossible causes:")
        print("  1. SSE endpoint not working")
        print("  2. Progress updates not being sent")
        print("  3. Model already downloaded (no progress to report)")
        print("\nTry deleting the model first to force a fresh download:")
        print(f"  curl -X DELETE http://localhost:8000/models/{model_name}")
        return False

    print(f"✓ Received {len(events)} SSE events")
    print(f"\nFirst event: {events[0]}")
    print(f"Last event: {events[-1]}")

    # Check if we got meaningful progress
    has_progress = any(e.get('progress', 0) > 0 for e in events)
    has_complete = any(e.get('status') == 'complete' for e in events)

    if has_progress:
        print("✓ Progress updates received")
    else:
        print("✗ No progress updates (might be already downloaded)")

    if has_complete:
        print("✓ Download completed successfully")
    else:
        print("✗ Download did not complete")

    success = has_progress and has_complete

    if success:
        print("\n✓ TEST PASSED - Progress tracking works!")
    else:
        print("\n⊘ TEST INCONCLUSIVE - Try with a fresh download")

    return success


if __name__ == "__main__":
    asyncio.run(main())
