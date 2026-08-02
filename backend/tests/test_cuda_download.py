import sys as py_sys
import types

import pytest

from backend.services import cuda


def test_cuda_status_reports_unsupported_linux_download(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda.sys, "platform", "linux")
    monkeypatch.setattr(cuda, "get_data_dir", lambda: tmp_path)

    status = cuda.get_cuda_status()

    assert status["available"] is False
    assert status["download_supported"] is False
    assert status["unsupported_reason"] == cuda.CUDA_DOWNLOAD_UNSUPPORTED_REASON


@pytest.mark.asyncio
async def test_cuda_download_rejects_linux_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda.sys, "platform", "linux")
    monkeypatch.setattr(cuda, "get_data_dir", lambda: tmp_path)

    class UnexpectedClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unsupported platforms should not start a release download")

    monkeypatch.setitem(py_sys.modules, "httpx", types.SimpleNamespace(AsyncClient=UnexpectedClient))

    with pytest.raises(RuntimeError, match="currently only published for Windows"):
        await cuda._download_cuda_binary_locked("v0.5.0")
