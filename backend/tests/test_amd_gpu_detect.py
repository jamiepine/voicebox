"""
Phase 2.1 Test: AMD GPU detection on Windows.

Validates is_amd_gpu_windows() via mocked WMI and torch queries.

Usage:
    python -m pytest backend/tests/test_amd_gpu_detect.py -v
"""

from unittest.mock import MagicMock, patch

from backend.utils.platform_detect import is_amd_gpu_windows


class TestAmdGpuWindows:
    """Unit tests for is_amd_gpu_windows with mocks."""

    @patch("backend.utils.platform_detect.platform.system", return_value="Linux")
    def test_returns_false_on_linux(self, _mock_system):
        """Non-Windows platforms should always return False."""
        assert is_amd_gpu_windows() is False

    @patch("backend.utils.platform_detect.platform.system", return_value="Windows")
    @patch(
        "backend.utils.platform_detect.subprocess.run",
        return_value=MagicMock(stdout="1\n", returncode=0),
    )
    def test_detects_amd_via_wmi(self, _mock_run, _mock_system):
        """WMI reporting an AMD adapter should return True."""
        assert is_amd_gpu_windows() is True

    @patch("backend.utils.platform_detect.platform.system", return_value="Windows")
    @patch(
        "backend.utils.platform_detect.subprocess.run",
        return_value=MagicMock(stdout="0\n", returncode=0),
    )
    def test_no_amd_via_wmi(self, _mock_run, _mock_system):
        """WMI reporting zero AMD adapters should return False."""
        assert is_amd_gpu_windows() is False

    @patch("backend.utils.platform_detect.platform.system", return_value="Windows")
    @patch(
        "backend.utils.platform_detect.subprocess.run",
        side_effect=Exception("WMI not available"),
    )
    @patch("torch.cuda.is_available", return_value=True)
    @patch(
        "torch.cuda.get_device_name",
        return_value="AMD Radeon RX 7800 XT",
    )
    def test_fallback_to_torch_radeon(self, _mock_name, _mock_avail, _mock_run, _mock_system):
        """When WMI fails, torch.cuda.get_device_name('Radeon') should return True."""
        assert is_amd_gpu_windows() is True

    @patch("backend.utils.platform_detect.platform.system", return_value="Windows")
    @patch(
        "backend.utils.platform_detect.subprocess.run",
        side_effect=Exception("WMI not available"),
    )
    @patch("torch.cuda.is_available", return_value=True)
    @patch(
        "torch.cuda.get_device_name",
        return_value="NVIDIA GeForce RTX 4090",
    )
    def test_fallback_to_torch_nvidia(self, _mock_name, _mock_avail, _mock_run, _mock_system):
        """When WMI fails, torch.cuda.get_device_name('NVIDIA') should return False."""
        assert is_amd_gpu_windows() is False

    @patch("backend.utils.platform_detect.platform.system", return_value="Windows")
    @patch(
        "backend.utils.platform_detect.subprocess.run",
        side_effect=Exception("WMI not available"),
    )
    @patch("torch.cuda.is_available", return_value=False)
    def test_no_torch_cuda(self, _mock_avail, _mock_run, _mock_system):
        """When WMI fails and torch.cuda is unavailable, should return False."""
        assert is_amd_gpu_windows() is False

    @patch("backend.utils.platform_detect.platform.system", return_value="Windows")
    @patch(
        "backend.utils.platform_detect.subprocess.run",
        side_effect=Exception("WMI not available"),
    )
    def test_torch_not_installed(self, _mock_run, _mock_system):
        """When torch is not installed, should return False without crashing."""
        with patch.dict("sys.modules", {"torch": None}):
            assert is_amd_gpu_windows() is False
