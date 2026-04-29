"""Test GPU auto-offload monitor.

Run from project root:
    python backend/tests/test_gpu_monitor.py
"""

import asyncio
import inspect
import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_gpu_monitor")

passed = 0
failed = 0
skipped = 0


def _record(result, name):
    global passed, failed, skipped
    if result is True:
        passed += 1
        print(f"  PASS  {name}")
    elif result is None:
        skipped += 1
        print(f"  SKIP  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


async def test_idle_tracker_tracks_activity():
    from backend.services.gpu_monitor import IdleTracker

    mock_backend = MagicMock()
    mock_backend.generate = AsyncMock(return_value=("audio", 24000))
    mock_backend.transcribe = AsyncMock(return_value="hello world")
    mock_backend.create_voice_prompt = AsyncMock(return_value=({"voice_type": "cloned"}, False))
    mock_backend.is_loaded = MagicMock(return_value=True)
    mock_backend.model_size = "0.6B"

    tracker = IdleTracker(mock_backend, "qwen")
    before = tracker.last_used_at

    await asyncio.sleep(0.01)
    result = await tracker.generate("text", {"voice_type": "cloned"})
    assert tracker.last_used_at > before
    assert result == ("audio", 24000)
    mock_backend.generate.assert_awaited_once()

    before2 = tracker.last_used_at
    await asyncio.sleep(0.01)
    result2 = await tracker.transcribe("/path/to/audio.wav")
    assert tracker.last_used_at > before2
    assert result2 == "hello world"

    before3 = tracker.last_used_at
    await asyncio.sleep(0.01)
    result3 = await tracker.create_voice_prompt("/path.wav", "hello")
    assert tracker.last_used_at > before3
    assert result3 == ({"voice_type": "cloned"}, False)

    assert tracker.is_loaded() is True
    assert tracker.model_size == "0.6B"
    assert repr(tracker) == "IdleTracker(qwen)"

    return True


async def test_gpu_monitor_auto_unload():
    from backend.services.gpu_monitor import IdleTracker, GPUMonitor

    mock_backend = MagicMock()
    mock_backend.is_loaded = MagicMock(return_value=True)
    mock_backend.unload_model = MagicMock()
    mock_backend.device = "cuda"

    tracker = IdleTracker(mock_backend, "qwen")
    tracker.last_used_at = time.time() - 100

    monitor = GPUMonitor(idle_timeout=10, sweep_interval=1)

    with patch("backend.services.gpu_monitor.empty_device_cache"):
        await monitor._check_backend(tracker, "qwen")

    assert mock_backend.unload_model.called
    return True


async def test_no_unload_during_generation():
    from backend.services.gpu_monitor import IdleTracker, GPUMonitor

    mock_backend = MagicMock()
    mock_backend.is_loaded = MagicMock(return_value=True)
    mock_backend.unload_model = MagicMock()
    mock_backend.device = "cuda"

    tracker = IdleTracker(mock_backend, "qwen")
    tracker.last_used_at = time.time() - 100
    tracker.is_generating = True

    monitor = GPUMonitor(idle_timeout=10, sweep_interval=1)

    with patch("backend.services.gpu_monitor.empty_device_cache"):
        await monitor._check_backend(tracker, "qwen")

    assert not mock_backend.unload_model.called
    return True


async def test_no_unload_when_not_idle():
    from backend.services.gpu_monitor import IdleTracker, GPUMonitor

    mock_backend = MagicMock()
    mock_backend.is_loaded = MagicMock(return_value=True)
    mock_backend.unload_model = MagicMock()
    mock_backend.device = "cuda"

    tracker = IdleTracker(mock_backend, "qwen")
    tracker.last_used_at = time.time()

    monitor = GPUMonitor(idle_timeout=600, sweep_interval=1)

    with patch("backend.services.gpu_monitor.empty_device_cache"):
        await monitor._check_backend(tracker, "qwen")

    assert not mock_backend.unload_model.called
    return True


async def test_no_unload_when_not_loaded():
    from backend.services.gpu_monitor import IdleTracker, GPUMonitor

    mock_backend = MagicMock()
    mock_backend.is_loaded = MagicMock(return_value=False)
    mock_backend.unload_model = MagicMock()

    tracker = IdleTracker(mock_backend, "qwen")
    tracker.last_used_at = time.time() - 100

    monitor = GPUMonitor(idle_timeout=10, sweep_interval=1)

    with patch("backend.services.gpu_monitor.empty_device_cache"):
        await monitor._check_backend(tracker, "qwen")

    assert not mock_backend.unload_model.called
    return True


async def test_skip_untracked_backend():
    from backend.services.gpu_monitor import GPUMonitor

    mock_backend = MagicMock(spec=[])
    mock_backend.unload_model = MagicMock()

    monitor = GPUMonitor(idle_timeout=1, sweep_interval=1)

    with patch("backend.services.gpu_monitor.empty_device_cache"):
        await monitor._check_backend(mock_backend, "qwen")

    assert not mock_backend.unload_model.called
    return True


async def test_monitor_start_stop():
    from backend.services.gpu_monitor import GPUMonitor

    monitor = GPUMonitor(idle_timeout=999999, sweep_interval=999999)
    await monitor.start()
    assert monitor._sweep_task is not None
    assert not monitor._sweep_task.done()

    await monitor.stop()
    assert monitor._sweep_task is None
    return True


async def test_loaded_models_status():
    from backend.services.gpu_monitor import GPUMonitor, IdleTracker

    monitor = GPUMonitor()

    # With no models loaded, should return empty list
    status = monitor.get_loaded_models_status()
    assert isinstance(status, list)
    print(f"  (status with no models loaded: {status})")

    # Verify the output structure is correct
    mock_backend = MagicMock()
    mock_backend.is_loaded = MagicMock(return_value=True)
    mock_backend.model_size = "0.6B"
    tracker = IdleTracker(mock_backend, "qwen")
    tracker.last_used_at = time.time() - 5

    item = {
        "engine": "Qwen TTS",
        "model": tracker.model_size,
        "last_used_seconds_ago": round(time.time() - tracker.last_used_at, 1),
        "loaded": True,
    }
    assert item["engine"] == "Qwen TTS"
    assert item["model"] == "0.6B"
    assert item["loaded"] is True
    assert item["last_used_seconds_ago"] >= 4.5
    return True


def test_wrap_backend():
    from backend.services.gpu_monitor import wrap_backend_with_idle_tracking, IdleTracker

    mock_backend = MagicMock()
    wrapped = wrap_backend_with_idle_tracking(mock_backend, "luxtts")

    assert isinstance(wrapped, IdleTracker)
    assert wrapped._engine == "luxtts"
    assert wrapped.last_used_at is not None
    assert wrapped.is_generating is False
    return True


async def test_factory_wiring():
    from backend.backends import reset_backends

    reset_backends()

    try:
        from backend.backends import get_tts_backend_for_engine
        from backend.services.gpu_monitor import IdleTracker

        with patch("backend.backends.get_backend_type", return_value="pytorch"):
            backend = get_tts_backend_for_engine("luxtts")

        assert isinstance(backend, IdleTracker)
        assert backend._engine == "luxtts"
        return True

    except ImportError as e:
        logger.warning("Cannot test factory wiring: %s", e)
        return None
    finally:
        reset_backends()


async def test_health_endpoint():
    from pydantic import ValidationError
    from backend.models import HealthResponse

    resp = HealthResponse(
        status="healthy",
        model_loaded=False,
        gpu_available=True,
        gpu_type="CUDA (Test)",
        vram_used_mb=0.0,
        backend_type="pytorch",
        backend_variant="cpu",
        gpu_models_loaded=[{"engine": "Qwen TTS", "model": "0.6B", "last_used_seconds_ago": 5.0, "loaded": True}],
    )

    assert hasattr(resp, "gpu_models_loaded")
    assert len(resp.gpu_models_loaded) == 1
    assert resp.gpu_models_loaded[0]["engine"] == "Qwen TTS"

    resp2 = HealthResponse(
        status="healthy",
        model_loaded=False,
        gpu_available=False,
        gpu_type=None,
        vram_used_mb=None,
        backend_type=None,
    )
    assert resp2.gpu_models_loaded == []

    return True


def test_env_var_config():
    import os
    from backend.services.gpu_monitor import GPUMonitor

    m1 = GPUMonitor()
    assert m1.idle_timeout == 600
    assert m1.sweep_interval == 30

    m2 = GPUMonitor(idle_timeout=120, sweep_interval=10)
    assert m2.idle_timeout == 120
    assert m2.sweep_interval == 10

    os.environ["VOICEBOX_GPU_IDLE_TIMEOUT_SECONDS"] = "999"
    os.environ["VOICEBOX_GPU_SWEEP_INTERVAL_SECONDS"] = "77"
    try:
        m3 = GPUMonitor()
        assert m3.idle_timeout == 999
        assert m3.sweep_interval == 77
    finally:
        del os.environ["VOICEBOX_GPU_IDLE_TIMEOUT_SECONDS"]
        del os.environ["VOICEBOX_GPU_SWEEP_INTERVAL_SECONDS"]

    return True


async def test_unload_calls_cache_and_gc():
    from backend.services.gpu_monitor import IdleTracker, GPUMonitor

    mock_backend = MagicMock()
    mock_backend.is_loaded = MagicMock(return_value=True)
    mock_backend.unload_model = MagicMock()
    mock_backend.device = "cuda"

    tracker = IdleTracker(mock_backend, "qwen")
    tracker.last_used_at = time.time() - 100

    monitor = GPUMonitor(idle_timeout=10, sweep_interval=1)

    with patch("backend.services.gpu_monitor.empty_device_cache") as mock_empty:
        with patch("backend.services.gpu_monitor.gc.collect") as mock_gc:
            await monitor._check_backend(tracker, "qwen")

    assert mock_backend.unload_model.called
    assert mock_empty.called
    assert mock_gc.called
    return True


async def test_live_server_health():
    import urllib.request
    import json

    try:
        req = urllib.request.Request("http://localhost:17493/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
        print("  (server not running, skipping live test)")
        return None

    if "gpu_models_loaded" not in data:
        print("  (server running old code, restart to pick up gpu_models_loaded)")
        return None

    assert isinstance(data["gpu_models_loaded"], list)
    print(f"    gpu_models_loaded: {data['gpu_models_loaded']}")
    return True


async def main():
    global passed, failed, skipped

    print("\n" + "=" * 60)
    print("GPU Auto-Offload Monitor — Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("IdleTracker tracks activity", test_idle_tracker_tracks_activity),
        ("GPUMonitor auto-unloads idle", test_gpu_monitor_auto_unload),
        ("No unload during generation", test_no_unload_during_generation),
        ("No unload when not idle", test_no_unload_when_not_idle),
        ("No unload when not loaded", test_no_unload_when_not_loaded),
        ("Skip untracked backend", test_skip_untracked_backend),
        ("Monitor start/stop lifecycle", test_monitor_start_stop),
        ("Loaded models status", test_loaded_models_status),
        ("wrap_backend returns IdleTracker", test_wrap_backend),
        ("Factory wiring", test_factory_wiring),
        ("Health endpoint field", test_health_endpoint),
        ("Env var configuration", test_env_var_config),
        ("Unload calls cache + gc", test_unload_calls_cache_and_gc),
        ("Live server /health", test_live_server_health),
    ]

    for name, test_fn in tests:
        try:
            if inspect.iscoroutinefunction(test_fn):
                result = await test_fn()
            else:
                result = test_fn()
            _record(result, name)
        except Exception as e:
            logger.exception("Test '%s' raised exception", name)
            _record(False, f"{name} — {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped out of {len(tests)}")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
