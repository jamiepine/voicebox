"""
Phase 1.2 Test: ROCm build script configuration.

Validates that build_binary.py --rocm generates the correct PyInstaller
arguments and optionally performs a true E2E build.

Usage:
    python -m pytest backend/tests/test_rocm_build.py -v
    python -m pytest backend/tests/test_rocm_build.py -v -m "slow"    # include E2E
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from build_binary import build_server


class TestRocmBuildArgs:
    """Validate PyInstaller arguments for ROCm builds."""

    @pytest.fixture
    def captured_args(self):
        """Run build_server(rocm=True) with mocked PyInstaller and return args."""
        with (
            patch("build_binary.PyInstaller.__main__.run") as mock_run,
            patch("build_binary.platform.system", return_value="Linux"),
            patch("build_binary.os.chdir"),
        ):
            build_server(rocm=True)
            return mock_run.call_args[0][0]

    def test_binary_name(self, captured_args):
        idx = captured_args.index("--name")
        assert captured_args[idx + 1] == "voicebox-server-rocm"

    def test_pack_mode_is_onedir(self, captured_args):
        assert "--onedir" in captured_args
        assert "--onefile" not in captured_args

    def test_hidden_imports_cuda(self, captured_args):
        """ROCm builds must include torch.cuda hidden imports."""
        assert "torch.cuda" in captured_args

    def test_no_cudnn_hidden_import_for_rocm(self, captured_args):
        """ROCm builds must NOT include NVIDIA-specific cudnn hidden imports."""
        assert "torch.backends.cudnn" not in captured_args

    def test_nvidia_excludes_present(self, captured_args):
        """ROCm builds must exclude nvidia packages to avoid bundling ~3GB of bloat."""
        excludes = []
        for i, arg in enumerate(captured_args):
            if arg == "--exclude-module":
                excludes.append(captured_args[i + 1])
        assert "nvidia" in excludes
        assert "nvidia.cudnn" in excludes


class TestRocmBuildCli:
    """Validate CLI argument parsing for --rocm."""

    def test_rocm_flag_parses(self):
        build_script = Path(__file__).parent.parent / "build_binary.py"
        result = subprocess.run(
            [sys.executable, str(build_script), "--rocm", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--rocm" in result.stdout

    def test_cannot_combine_cuda_and_rocm(self):
        """Building with both CUDA and ROCm should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot build with both CUDA and ROCm"):
            build_server(cuda=True, rocm=True)


@pytest.mark.slow()
@pytest.mark.skipif(sys.platform != "win32", reason="ROCm build E2E only runs on Windows")
class TestRocmBuildE2E:
    """
    True end-to-end build test.
    Executes build_binary.py --rocm, verifies the binary exists, and runs it
    with --help to confirm it boots without import errors.
    """

    def test_rocm_binary_compiles_and_runs(self, tmp_path):
        backend_dir = Path(__file__).parent.parent
        build_script = backend_dir / "build_binary.py"
        dist_dir = backend_dir / "dist"
        binary_dir = dist_dir / "voicebox-server-rocm"
        binary_exe = binary_dir / "voicebox-server-rocm.exe"

        # Clean previous dist if it exists to ensure a fresh build
        if binary_dir.exists():
            import shutil
            shutil.rmtree(binary_dir)

        # Run the full build (this can take several minutes)
        result = subprocess.run(
            [sys.executable, str(build_script), "--rocm"],
            capture_output=True,
            text=True,
            cwd=str(backend_dir),
            timeout=900,
        )

        assert result.returncode == 0, (
            f"Build failed with stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert binary_exe.exists(), (
            f"Expected binary not found at {binary_exe}"
        )

        # Run the binary with --help to ensure it boots without import errors
        run_result = subprocess.run(
            [str(binary_exe), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # A frozen binary may not have argparse help, but it should not crash
        # with a ModuleNotFoundError or similar import error.
        assert "ModuleNotFoundError" not in run_result.stderr
        assert "ImportError" not in run_result.stderr
