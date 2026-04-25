"""
Tests for the ROCm backend download service.

Mocks httpx to verify download, extraction, and progress reporting
without hitting the network.
"""

import json
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services import rocm
from backend.utils.progress import get_progress_manager


@pytest.fixture(autouse=True)
def reset_progress_manager():
    """Reset the global progress manager before each test."""
    import backend.utils.progress
    backend.utils.progress._progress_manager = None
    yield
    backend.utils.progress._progress_manager = None


@pytest.fixture
def mock_backends_dir(tmp_path: Path, monkeypatch):
    """Patch get_data_dir so downloads land in a temp directory."""
    monkeypatch.setattr(rocm, "get_backends_dir", lambda: tmp_path / "backends")
    return tmp_path / "backends"


@pytest.fixture
def fake_tar_gz():
    """Create an in-memory .tar.gz archive containing a dummy file."""
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"fake binary content"
        info = tarfile.TarInfo(name="voicebox-server-rocm.exe")
        info.size = len(data)
        tar.addfile(info, BytesIO(data))
    buf.seek(0)
    return buf.read()


@pytest.fixture
def fake_sha256():
    """Return a dummy SHA-256 hex string."""
    return "a" * 64


class FakeResponse:
    """Minimal fake for httpx.Response."""

    def __init__(self, content: bytes = b"", status_code: int = 200, headers: dict | None = None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size: int = 1024):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    async def aiter_bytes(self, chunk_size: int = 1024):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    @property
    def text(self):
        return self.content.decode()


class FakeHttpxClient:
    """Minimal fake for httpx.AsyncClient."""

    def __init__(self, responses: dict[str, FakeResponse]):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def head(self, url: str):
        return self._responses.get(url, FakeResponse(status_code=404))

    async def get(self, url: str):
        return self._responses.get(url, FakeResponse(status_code=404))

    def stream(self, method: str, url: str):
        resp = self._responses.get(url, FakeResponse(status_code=404))
        resp.raise_for_status()

        class _Streamer:
            async def __aenter__(self):
                return resp

            async def __aexit__(self, *args):
                return False

            async def aiter_bytes(self, chunk_size: int = 1024):
                for i in range(0, len(resp.content), chunk_size):
                    yield resp.content[i : i + chunk_size]

        return _Streamer()


@pytest.mark.asyncio
async def test_get_rocm_status_not_installed(mock_backends_dir):
    status = rocm.get_rocm_status()
    assert status["available"] is False
    assert status["active"] is False
    assert status["binary_path"] is None
    assert status["downloading"] is False


@pytest.mark.asyncio
async def test_download_rocm_binary_progress_reporting(mock_backends_dir, fake_tar_gz, fake_sha256):
    """
    Verify that download_rocm_binary():
    1. Downloads the server archive and ROCm libs archive.
    2. Extracts them into the backends/rocm directory.
    3. Reports progress via the progress_manager.
    """
    import hashlib

    server_sha = hashlib.sha256(fake_tar_gz).hexdigest()
    libs_sha = hashlib.sha256(fake_tar_gz).hexdigest()

    responses = {
        "https://github.com/jamiepine/voicebox/releases/download/v0.2.3/voicebox-server-rocm.tar.gz": FakeResponse(
            content=fake_tar_gz,
            headers={"content-length": str(len(fake_tar_gz))},
        ),
        "https://github.com/jamiepine/voicebox/releases/download/v0.2.3/voicebox-server-rocm.tar.gz.sha256": FakeResponse(
            content=f"{server_sha}  voicebox-server-rocm.tar.gz\n".encode(),
        ),
        f"https://github.com/jamiepine/voicebox/releases/download/v0.2.3/rocm-libs-{rocm.ROCM_LIBS_VERSION}.tar.gz": FakeResponse(
            content=fake_tar_gz,
            headers={"content-length": str(len(fake_tar_gz))},
        ),
        f"https://github.com/jamiepine/voicebox/releases/download/v0.2.3/rocm-libs-{rocm.ROCM_LIBS_VERSION}.tar.gz.sha256": FakeResponse(
            content=f"{libs_sha}  rocm-libs.tar.gz\n".encode(),
        ),
    }

    fake_client = FakeHttpxClient(responses)

    with patch("httpx.AsyncClient", return_value=fake_client):
        await rocm.download_rocm_binary(version="v0.2.3")

    # Verify extraction
    rocm_dir = rocm.get_rocm_dir()
    assert (rocm_dir / "voicebox-server-rocm.exe").exists()

    # Verify manifest written
    manifest_path = rocm.get_rocm_libs_manifest_path()
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["version"] == rocm.ROCM_LIBS_VERSION

    # Verify progress was reported
    progress = get_progress_manager().get_progress("rocm-backend")
    assert progress is not None
    assert progress["status"] == "complete"
    assert progress["progress"] == 100.0


@pytest.mark.asyncio
async def test_is_rocm_active(mock_backends_dir, monkeypatch):
    monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", "rocm")
    assert rocm.is_rocm_active() is True

    monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", "cpu")
    assert rocm.is_rocm_active() is False

    monkeypatch.delenv("VOICEBOX_BACKEND_VARIANT", raising=False)
    assert rocm.is_rocm_active() is False


@pytest.mark.asyncio
async def test_delete_rocm_binary(mock_backends_dir, fake_tar_gz):
    """Test deleting the ROCm backend directory."""
    rocm_dir = rocm.get_rocm_dir()
    rocm_dir.mkdir(parents=True, exist_ok=True)
    (rocm_dir / "dummy.txt").write_text("hello")

    result = await rocm.delete_rocm_binary()
    assert result is True
    assert not rocm_dir.exists()

    # Deleting again should return False
    result = await rocm.delete_rocm_binary()
    assert result is False
