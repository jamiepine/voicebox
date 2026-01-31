"""
Test Qwen TTS model download with SSE progress monitoring.

This specifically tests the MLX TTS backend download progress tracking,
which requires tqdm to be patched BEFORE mlx_audio is imported.

Usage:
    cd backend && python -m tests.test_qwen_download

Prerequisites:
    - Server must be running: cd backend && python main.py
    - Delete model first for fresh download test:
      curl -X DELETE http://localhost:8000/models/qwen-tts-0.6B
"""

import asyncio
import json
import httpx
import time
from typing import List, Dict, Optional


async def monitor_sse_stream(model_name: str, timeout: int = 600) -> List[Dict]:
    """
    Monitor SSE stream for a model download.
    
    Args:
        model_name: Name of the model to monitor
        timeout: Maximum time to wait for download (seconds)
        
    Returns:
        List of SSE events received
    """
    events: List[Dict] = []
    url = f"http://localhost:8000/models/progress/{model_name}"
    last_progress = -1
    
    print(f"\n📡 Connecting to SSE endpoint: {url}")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url) as response:
                print(f"   SSE connected, status: {response.status_code}")

                if response.status_code != 200:
                    print(f"   ❌ Error: SSE endpoint returned {response.status_code}")
                    return events

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            events.append(data)
                            
                            # Print progress (only when it changes significantly)
                            progress = data.get('progress', 0)
                            status = data.get('status', 'unknown')
                            filename = data.get('filename', '')
                            current = data.get('current', 0)
                            total = data.get('total', 0)
                            
                            # Print every 5% change or status change
                            if abs(progress - last_progress) >= 5 or status in ('complete', 'error'):
                                current_mb = current / (1024 * 1024)
                                total_mb = total / (1024 * 1024)
                                print(f"   📊 {status:12} {progress:6.1f}% ({current_mb:.1f}MB / {total_mb:.1f}MB) {filename[:50]}")
                                last_progress = progress

                            # Stop if complete or error
                            if status in ("complete", "error"):
                                if status == "complete":
                                    print(f"   ✅ Download complete!")
                                else:
                                    print(f"   ❌ Download error: {data.get('error', 'unknown')}")
                                break

                        except json.JSONDecodeError as e:
                            print(f"   ⚠️  Error parsing JSON: {e}")

                    elif line.startswith(": heartbeat"):
                        # Heartbeat every 1 second, don't spam
                        pass

    except asyncio.CancelledError:
        print("   ⏹️  SSE monitor cancelled")
    except Exception as e:
        print(f"   ❌ SSE error: {e}")

    return events


async def trigger_download(model_name: str) -> bool:
    """Trigger a model download via the API."""
    url = "http://localhost:8000/models/download"

    print(f"\n🚀 Triggering download for: {model_name}")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json={"model_name": model_name})
            result = response.json()
            print(f"   Response: {response.status_code} - {result}")
            return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error triggering download: {e}")
        return False


async def delete_model(model_name: str) -> bool:
    """Delete a model from cache."""
    url = f"http://localhost:8000/models/{model_name}"

    print(f"\n🗑️  Deleting model: {model_name}")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(url)
            if response.status_code == 200:
                print(f"   ✅ Model deleted")
                return True
            elif response.status_code == 404:
                print(f"   ℹ️  Model not found (already deleted)")
                return True
            else:
                print(f"   ⚠️  Delete response: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        print(f"   ❌ Error deleting model: {e}")
        return False


async def check_model_status(model_name: str) -> Optional[Dict]:
    """Check the status of a model."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("http://localhost:8000/models/status")
            if response.status_code == 200:
                data = response.json()
                for model in data.get("models", []):
                    if model["model_name"] == model_name:
                        return model
    except Exception as e:
        print(f"   ⚠️  Error checking model status: {e}")
    return None


async def check_server() -> bool:
    """Check if the server is running."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code == 200
    except Exception:
        return False


async def main():
    print("=" * 70)
    print("🧪 Qwen TTS Model Download Progress Test")
    print("=" * 70)
    print("\nThis test verifies that MLX TTS download progress tracking works.")
    print("It specifically tests the tqdm patching for mlx_audio.tts imports.")

    # Check if server is running
    print("\n📡 Checking if server is running...")
    if not await check_server():
        print("   ❌ Server is not running on http://localhost:8000")
        print("\n   Please start the server first:")
        print("   cd backend && python main.py")
        return False

    print("   ✅ Server is running")

    # Test model
    model_name = "qwen-tts-0.6B"  # Note: 0.6B currently maps to 1.7B on MLX
    
    # Check current status
    print(f"\n📊 Checking status of {model_name}...")
    status = await check_model_status(model_name)
    if status:
        print(f"   Downloaded: {status.get('downloaded', False)}")
        print(f"   Downloading: {status.get('downloading', False)}")
        print(f"   Loaded: {status.get('loaded', False)}")
        if status.get('size_mb'):
            print(f"   Size: {status['size_mb']:.1f} MB")
    else:
        print("   ⚠️  Could not get model status")

    # Ask if user wants to delete first
    print("\n" + "-" * 70)
    if status and status.get('downloaded'):
        print("⚠️  Model is already downloaded. Delete it for a fresh download test?")
        print("   [y] Yes, delete and download fresh")
        print("   [n] No, just test SSE connection")
        print("   [q] Quit")
        
        choice = input("\nChoice [y/n/q]: ").strip().lower()
        
        if choice == 'q':
            print("Exiting...")
            return True
        
        if choice == 'y':
            if not await delete_model(model_name):
                print("Failed to delete model. Continue anyway? [y/n]")
                if input().strip().lower() != 'y':
                    return False
    else:
        print("Model not downloaded. Will perform fresh download test.")
        input("Press Enter to continue...")

    # Run the test
    print("\n" + "=" * 70)
    print("🏃 Starting Download Test")
    print("=" * 70)

    async def run_test():
        # Start SSE monitor in background FIRST
        monitor_task = asyncio.create_task(monitor_sse_stream(model_name, timeout=600))

        # Wait for SSE to connect
        await asyncio.sleep(1)

        # Trigger download
        success = await trigger_download(model_name)

        if not success:
            print("   ❌ Failed to trigger download")
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            return []

        # Wait for SSE monitor to complete
        print("\n⏳ Waiting for download to complete (this may take several minutes)...")
        events = await monitor_task

        return events

    start_time = time.time()
    events = await run_test()
    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("📋 Test Results")
    print("=" * 70)

    print(f"\n⏱️  Elapsed time: {elapsed:.1f} seconds")
    print(f"📨 Total SSE events received: {len(events)}")

    if not events:
        print("\n❌ FAILED - No SSE events received!")
        print("\nPossible causes:")
        print("  1. SSE endpoint not working")
        print("  2. tqdm not patched before mlx_audio import")
        print("  3. Progress callbacks not firing")
        print("  4. Model already fully downloaded")
        print("\nDebug steps:")
        print("  1. Check server logs for [DEBUG] messages")
        print("  2. Look for 'tqdm patched' before 'mlx_audio.tts import'")
        print(f"  3. Delete model: curl -X DELETE http://localhost:8000/models/{model_name}")
        return False

    # Analyze events
    first_event = events[0]
    last_event = events[-1]
    
    print(f"\n📊 First event:")
    print(f"   Status: {first_event.get('status')}")
    print(f"   Progress: {first_event.get('progress', 0):.1f}%")
    
    print(f"\n📊 Last event:")
    print(f"   Status: {last_event.get('status')}")
    print(f"   Progress: {last_event.get('progress', 0):.1f}%")

    # Check for expected behaviors
    has_progress_updates = len(events) > 2
    has_increasing_progress = False
    has_complete = any(e.get('status') == 'complete' for e in events)
    has_100_percent = any(e.get('progress', 0) >= 100 for e in events)
    
    # Check if progress increased over time
    if len(events) >= 2:
        progress_values = [e.get('progress', 0) for e in events]
        has_increasing_progress = progress_values[-1] > progress_values[0]

    print("\n📋 Checks:")
    print(f"   {'✅' if has_progress_updates else '❌'} Multiple progress updates received ({len(events)} events)")
    print(f"   {'✅' if has_increasing_progress else '❌'} Progress increased over time")
    print(f"   {'✅' if has_100_percent else '❌'} Reached 100% progress")
    print(f"   {'✅' if has_complete else '❌'} Received 'complete' status")

    # Overall result
    success = has_progress_updates and has_complete
    
    if success:
        print("\n" + "=" * 70)
        print("✅ TEST PASSED - Qwen TTS download progress tracking works!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED - Progress tracking has issues")
        print("=" * 70)
        print("\nCheck the server logs for debug output.")

    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
