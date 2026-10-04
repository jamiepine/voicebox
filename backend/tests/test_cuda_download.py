"""
Platform gate for the downloadable CUDA backend.

CUDA server assets are published for linux-x86_64 and windows-x86_64 only.
Any other platform must be refused before a release download starts.
"""

import sys as py_sys
import types
from unittest.mock import patch

import pytest

from backend.services import cuda


def _platform(system: str, machine: str):
    return (
        patch("backend.utils.platform_detect.platform.system", return_value=system),
        patch("backend.utils.platform_detect.platform.machine", return_value=machine),
    )


@pytest.mark.parametrize(
    "system,machine",
    [("Darwin", "arm64"), ("Linux", "aarch64")],
)
def test_cuda_status_reports_unsupported_platform(monkeypatch, tmp_path, system, machine):
    monkeypatch.setattr(cuda, "get_data_dir", lambda: tmp_path)

    with _platform(system, machine)[0], _platform(system, machine)[1]:
        status = cuda.get_cuda_status()

    assert status["available"] is False
    assert status["download_supported"] is False
    assert status["unsupported_reason"] == cuda.CUDA_DOWNLOAD_UNSUPPORTED_REASON


@pytest.mark.parametrize(
    "system,machine",
    [("Linux", "x86_64"), ("Windows", "AMD64")],
)
def test_cuda_status_reports_supported_platform(monkeypatch, tmp_path, system, machine):
    monkeypatch.setattr(cuda, "get_data_dir", lambda: tmp_path)

    with _platform(system, machine)[0], _platform(system, machine)[1]:
        status = cuda.get_cuda_status()

    assert status["download_supported"] is True
    assert status["unsupported_reason"] is None


@pytest.mark.asyncio
async def test_cuda_download_rejects_macos_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda, "get_data_dir", lambda: tmp_path)

    class UnexpectedClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unsupported platforms should not start a release download")

    monkeypatch.setitem(py_sys.modules, "httpx", types.SimpleNamespace(AsyncClient=UnexpectedClient))

    with _platform("Darwin", "arm64")[0], _platform("Darwin", "arm64")[1]:
        with pytest.raises(RuntimeError, match="only published for"):
            await cuda._download_cuda_binary_locked("v0.5.0")


@pytest.mark.asyncio
async def test_cuda_download_requests_linux_x86_64_assets(monkeypatch, tmp_path):
    """On Linux x86_64 the download must fetch the Linux-qualified asset names."""
    monkeypatch.setattr(cuda, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(cuda, "_needs_server_download", lambda version: True)
    monkeypatch.setattr(cuda, "_needs_cuda_libs_download", lambda: False)

    requested: list[str] = []

    class StopDownload(Exception):
        pass

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def head(self, url, *args, **kwargs):
            return types.SimpleNamespace(headers={"content-length": "0"})

    async def fake_download(client, url, sha256_url, **kwargs):
        requested.append(url)
        requested.append(sha256_url)
        raise StopDownload()

    monkeypatch.setattr(cuda, "_download_and_extract_archive", fake_download)
    monkeypatch.setitem(py_sys.modules, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))

    with _platform("Linux", "x86_64")[0], _platform("Linux", "x86_64")[1]:
        with pytest.raises(StopDownload):
            await cuda._download_cuda_binary_locked("v0.7.0")

    asset = f"{cuda.GITHUB_RELEASES_URL}/v0.7.0/voicebox-server-cuda-linux-x86_64.tar.gz"
    assert requested == [asset, f"{asset}.sha256"]
