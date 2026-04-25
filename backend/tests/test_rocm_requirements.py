"""
Phase 1.1 Test: ROCm requirements installation.

Validates that requirements-rocm.txt correctly installs ROCm-enabled PyTorch
and that torch.cuda.is_available() returns True on AMD hardware.

Usage:
    python -m pytest backend/tests/test_rocm_requirements.py -v
"""

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _has_amd_hardware():
    """Check if AMD GPU hardware is present on Windows."""
    if platform.system() != "Windows":
        return False
    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-WmiObject Win32_VideoController | "
                "Where-Object {$_.AdapterCompatibility -like '*AMD*'} | "
                "Measure-Object | Select-Object -ExpandProperty Count",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip()) > 0
    except Exception:
        return False


@pytest.fixture()
def backend_dir():
    return Path(__file__).parent.parent


class TestRocmRequirements:
    """Validate requirements-rocm.txt content and installation."""

    def test_requirements_file_exists(self, backend_dir):
        req_file = backend_dir / "requirements-rocm.txt"
        assert req_file.exists(), "requirements-rocm.txt must exist"

    def test_requirements_file_content(self, backend_dir):
        import re
        req_file = backend_dir / "requirements-rocm.txt"
        content = req_file.read_text()
        assert "rocm7.2" in content, "Must point to ROCm 7.2 extra index"
        # Parse exact package names to avoid false positives from URL substrings
        package_names = re.findall(r"^([A-Za-z][A-Za-z0-9_-]*)", content, re.MULTILINE)
        assert "torch" in package_names, "Must include torch package"
        assert "torchaudio" in package_names, "Must include torchaudio package"
        assert "torchvision" in package_names, "Must include torchvision package"

    @pytest.mark.timeout(900)
    @pytest.mark.skipif(
        not os.environ.get("VOICEBOX_TEST_ROCM_INSTALL"),
        reason="Set VOICEBOX_TEST_ROCM_INSTALL=1 to run the heavy install test",
    )
    def test_rocm_torch_installs_and_detects_amd(self, backend_dir):
        """
        Create a temporary venv, install requirements-rocm.txt, and verify
        torch.cuda.is_available() returns True on AMD hardware.
        """
        req_file = backend_dir / "requirements-rocm.txt"
        has_amd = _has_amd_hardware()

        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "venv"
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
            )

            if sys.platform == "win32":
                venv_python = venv_dir / "Scripts" / "python.exe"
            else:
                venv_python = venv_dir / "bin" / "python"

            # Upgrade pip to avoid resolver issues
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
            )

            # Install ROCm requirements
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-r", str(req_file)],
                check=True,
            )

            # Verify torch imports and cuda availability
            result = subprocess.run(
                [
                    str(venv_python),
                    "-c",
                    "import torch; print(torch.__version__); print(torch.cuda.is_available())",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            lines = result.stdout.strip().splitlines()
            assert len(lines) >= 2, f"Unexpected output: {result.stdout}"
            torch_version = lines[0]
            cuda_available = lines[1] == "True"

            # The honest test: on AMD hardware ROCm torch should report cuda available
            if has_amd:
                assert cuda_available, (
                    f"AMD hardware detected but torch.cuda.is_available() returned False. "
                    f"torch version: {torch_version}, stderr: {result.stderr}"
                )
            else:
                assert not cuda_available, (
                    f"No AMD hardware detected but torch.cuda.is_available() returned True. "
                    f"torch version: {torch_version}"
                )
