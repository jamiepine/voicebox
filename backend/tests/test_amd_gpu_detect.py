"""
Phase 2.1 Test: AMD GPU detection on Windows.

Validates is_amd_gpu_windows() via mocked WMI and torch queries.

Usage:
    python -m pytest backend/tests/test_amd_gpu_detect.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.utils.platform_detect import (
    is_amd_gpu_windows,
    is_amd_rocm_capable,
    server_asset_platform,
)


class TestAmdGpuWindows:
    """Unit tests for is_amd_gpu_windows with mocks."""

    @pytest.fixture(autouse=True)
    def _clear_detection_cache(self):
        # is_amd_gpu_windows is memoized; reset between cases so each mock takes effect.
        is_amd_gpu_windows.cache_clear()
        yield
        is_amd_gpu_windows.cache_clear()

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
    @patch("torch.cuda.is_available", return_value=False)
    def test_no_amd_via_wmi(self, _mock_avail, _mock_run, _mock_system):
        """WMI reporting zero AMD adapters should return False.

        torch.cuda.is_available is pinned False so the fallback probe can't pick
        up real AMD hardware when this runs on an actual ROCm host.
        """
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


class TestServerAssetPlatform:
    """Pin the exact release-asset platform tokens the download URLs depend on."""

    @pytest.mark.parametrize(
        "system,machine,expected",
        [
            ("Linux", "x86_64", "linux-x86_64"),
            ("Windows", "AMD64", "windows-x86_64"),
            ("Linux", "aarch64", "linux-arm64"),
            ("Darwin", "arm64", "darwin-arm64"),
        ],
    )
    def test_tokens(self, system, machine, expected):
        with (
            patch("backend.utils.platform_detect.platform.system", return_value=system),
            patch("backend.utils.platform_detect.platform.machine", return_value=machine),
        ):
            assert server_asset_platform() == expected


class TestAmdRocmCapable:
    """Linux ROCm capability gate (drives the ROCm download UI)."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        is_amd_rocm_capable.cache_clear()
        yield
        is_amd_rocm_capable.cache_clear()

    @patch("backend.utils.platform_detect.platform.machine", return_value="x86_64")
    @patch("backend.utils.platform_detect.platform.system", return_value="Linux")
    @patch("backend.utils.platform_detect.os.path.exists", return_value=True)
    def test_linux_x86_with_kfd_is_capable(self, _mock_exists, _mock_system, _mock_machine):
        """/dev/kfd present => ROCm-capable, even before ROCm torch is installed."""
        assert is_amd_rocm_capable() is True

    @patch("backend.utils.platform_detect.platform.machine", return_value="x86_64")
    @patch("backend.utils.platform_detect.platform.system", return_value="Linux")
    @patch("backend.utils.platform_detect.os.path.exists", return_value=False)
    def test_linux_without_kfd_is_not_capable(self, _mock_exists, _mock_system, _mock_machine):
        assert is_amd_rocm_capable() is False

    @patch("backend.utils.platform_detect.platform.machine", return_value="aarch64")
    @patch("backend.utils.platform_detect.platform.system", return_value="Linux")
    @patch("backend.utils.platform_detect.os.path.exists", return_value=True)
    def test_linux_arm_is_not_offered_unbuilt_asset(self, _mock_exists, _mock_system, _mock_machine):
        """ARM has /dev/kfd but no published asset — must not be offered a 404."""
        assert is_amd_rocm_capable() is False

    @patch("backend.utils.platform_detect.platform.machine", return_value="arm64")
    @patch("backend.utils.platform_detect.platform.system", return_value="Darwin")
    def test_macos_is_not_capable(self, _mock_system, _mock_machine):
        assert is_amd_rocm_capable() is False
