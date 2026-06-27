import pytest
from unittest.mock import patch, MagicMock
from ..services import cuda

def test_is_cuda_download_supported_win32():
    with patch("sys.platform", "win32"):
        assert cuda.is_cuda_download_supported() is True
        assert cuda.get_cuda_download_unsupported_reason() is None

def test_is_cuda_download_supported_linux():
    with patch("sys.platform", "linux"):
        assert cuda.is_cuda_download_supported() is False
        assert cuda.get_cuda_download_unsupported_reason() == cuda.CUDA_DOWNLOAD_UNSUPPORTED_REASON

def test_ensure_cuda_download_supported_win32():
    with patch("sys.platform", "win32"):
        # Should not raise
        cuda.ensure_cuda_download_supported()

def test_ensure_cuda_download_supported_linux():
    with patch("sys.platform", "linux"):
        with pytest.raises(RuntimeError) as excinfo:
            cuda.ensure_cuda_download_supported()
        assert str(excinfo.value) == cuda.CUDA_DOWNLOAD_UNSUPPORTED_REASON

def test_get_cuda_status_fields():
    # Verify that the new fields are present in the status dictionary
    status = cuda.get_cuda_status()
    assert "download_supported" in status
    assert "unsupported_reason" in status
    assert isinstance(status["download_supported"], bool)
    assert status["unsupported_reason"] is None or isinstance(status["unsupported_reason"], str)
