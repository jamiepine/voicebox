"""
Regression tests for the VOICEBOX_FORCE_CPU environment override.

The docs promise (docs/content/docs/developer/tts-generation.mdx) that
get_torch_device() layers "VOICEBOX_FORCE_CPU environment override" ahead of
CUDA/XPU/MPS detection, and gpu-acceleration.mdx tells users to set it to fall
back to CPU when the bundled PyTorch has no kernels for their GPU.

torch is stubbed through sys.modules so these run without a torch install and
without any GPU.

Usage:
    python -m pytest backend/tests/test_force_cpu_env.py -v
"""

import sys

import pytest

from backend.backends.base import get_torch_device

# The documented public name and value of the override. Pinned here independently
# of the production constants so a rename of either fails these tests.
FORCE_CPU_ENV_VAR = "VOICEBOX_FORCE_CPU"

# Sentinel for "the variable is not set at all".
UNSET = None


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True


class _FakeTorch:
    """Minimal stand-in for a CUDA-enabled torch install."""

    cuda = _FakeCuda


@pytest.fixture
def cuda_available(monkeypatch):
    """Make torch report a usable CUDA device without installing torch."""
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)


def _set_override(monkeypatch, value):
    if value is UNSET:
        monkeypatch.delenv(FORCE_CPU_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(FORCE_CPU_ENV_VAR, value)


@pytest.mark.parametrize("value", ["1", " 1 "])
def test_force_cpu_wins_over_available_cuda(monkeypatch, cuda_available, value):
    """The documented value must beat an otherwise usable CUDA device.

    Surrounding whitespace is tolerated: on Windows, where this override
    matters most, it is typically set through the GUI environment editor."""
    _set_override(monkeypatch, value)

    assert get_torch_device(allow_xpu=True, allow_directml=True, allow_mps=True) == "cpu"


@pytest.mark.parametrize("value", [UNSET, "", "0"])
def test_without_override_cuda_is_still_selected(monkeypatch, cuda_available, value):
    """Unset or disabled must not disturb normal device detection."""
    _set_override(monkeypatch, value)

    assert get_torch_device() == "cuda"


def test_force_cpu_does_not_need_torch(monkeypatch):
    """The override is honoured before torch is imported, so it works on a
    broken/incompatible torch install — which is the case it exists for."""
    _set_override(monkeypatch, "1")
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise

    assert get_torch_device() == "cpu"
