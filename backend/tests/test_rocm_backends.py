"""
Phase 2.2 Test: Backend ROCm compatibility.

Validates that check_cuda_compatibility() and other backend utilities
behave correctly on ROCm/AMD hardware.

Usage:
    python -m pytest backend/tests/test_rocm_backends.py -v
"""

from unittest.mock import patch

import pytest


class TestCheckCudaCompatibility:
    """Unit tests for check_cuda_compatibility with ROCm awareness."""

    def test_no_gpu_returns_compatible(self):
        from backend.backends.base import check_cuda_compatibility

        with patch("torch.cuda.is_available", return_value=False):
            compatible, warning = check_cuda_compatibility()
            assert compatible is True
            assert warning is None

    def test_rocm_skips_compute_check(self):
        """On ROCm, the NVIDIA compute-capability check should be skipped."""
        from backend.backends.base import check_cuda_compatibility

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.version.hip", "6.2.41133"):
                compatible, warning = check_cuda_compatibility()
                assert compatible is True
                assert warning is None

    def test_cuda_compatible_arch(self):
        from backend.backends.base import check_cuda_compatibility

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.version.hip", None):
                with patch("torch.cuda.get_device_capability", return_value=(8, 6)):
                    with patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3060"):
                        with patch.object(
                            __import__("torch").cuda, "_get_arch_list",
                            return_value=["sm_80", "sm_86", "sm_89"],
                            create=True,
                        ):
                            compatible, warning = check_cuda_compatibility()
                            assert compatible is True
                            assert warning is None

    def test_cuda_incompatible_arch(self):
        from backend.backends.base import check_cuda_compatibility

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.version.hip", None):
                with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                    with patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 4090"):
                        with patch.object(
                            __import__("torch").cuda, "_get_arch_list",
                            return_value=["sm_80", "sm_86"],
                            create=True,
                        ):
                            compatible, warning = check_cuda_compatibility()
                            assert compatible is False
                            assert warning is not None
                            assert "not supported" in warning
