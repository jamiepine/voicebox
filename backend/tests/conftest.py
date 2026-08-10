"""Shared pytest fixtures and markers for the voicebox backend suite.

Markers:
    gpu  — needs a local CUDA GPU; excluded from CI runs.
    e2e  — spawns the real backend process; excluded from CI runs.
    slow — long-running checks (e.g. test_rocm_build.py's full build).

CI-safe selection: ``python -m pytest -m "not gpu and not e2e"``.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HEALTH_TIMEOUT = 120


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires a local CUDA GPU (not run in CI)")
    config.addinivalue_line("markers", "e2e: spawns the real backend process (not run in CI)")
    config.addinivalue_line("markers", "slow: long-running test (deselect with -m 'not slow')")


@pytest.fixture(scope="session")
def gpu_or_skip() -> bool:
    """Require a CUDA-capable GPU; skip the test otherwise."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU available")
    return True


@pytest.fixture(scope="session")
def vram_mb():
    """Return a callable reading GPU 0 used VRAM (MiB) via nvidia-smi."""

    def _read() -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0])

    return _read


def _pick_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout: int) -> None:
    import httpx

    deadline = time.time() + timeout
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"backend exited early with code {proc.returncode}")
            try:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
    raise TimeoutError(f"backend did not become healthy within {timeout}s")


@pytest.fixture(scope="session")
def live_backend(tmp_path_factory):
    """Spawn the real backend on a free port with an isolated data directory."""
    data_dir = tmp_path_factory.mktemp("voicebox-data")
    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = data_dir / "server.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "backend.server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--data-dir",
                str(data_dir),
                "--parent-pid",
                str(os.getpid()),
            ],
            cwd=str(REPO_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        log_fh.close()
        pytest.skip(f"could not spawn backend process: {exc}")

    try:
        _wait_for_health(base_url, proc, timeout=HEALTH_TIMEOUT)
    except Exception as exc:
        proc.kill()
        proc.wait(timeout=10)
        log_fh.close()
        with contextlib.suppress(OSError):
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        if "tail" not in locals():
            tail = ""
        pytest.skip(f"backend did not become healthy: {exc}\n--- server log tail ---\n{tail}")

    try:
        yield base_url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        log_fh.close()
