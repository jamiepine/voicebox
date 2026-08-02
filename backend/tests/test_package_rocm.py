"""
Tests for scripts/package_rocm.py — the ROCm onedir → server + libs splitter.

The classifier can't be validated against a real AMD build on CI hardware, so
these tests pin the file-classification rules against a synthetic onedir layout
that mirrors the PyInstaller --rocm output (torch/lib HIP DLLs + bundled
rocm_sdk runtime packages).

Usage:
    python -m pytest backend/tests/test_package_rocm.py -v
"""

import importlib.util
import tarfile
from pathlib import Path

import pytest

_PACKAGE_ROCM = Path(__file__).resolve().parents[2] / "scripts" / "package_rocm.py"
_spec = importlib.util.spec_from_file_location("package_rocm", _PACKAGE_ROCM)
package_rocm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(package_rocm)


class TestIsRocmFile:
    """Classification of individual files into core vs ROCm libs."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "_internal/torch/lib/amdhip64.dll",
            "_internal/torch/lib/rocblas.dll",
            "_internal/torch/lib/hipblaslt.dll",
            "_internal/torch/lib/miopen.dll",
            "_internal/_rocm_sdk_core/amd_comgr.dll",
            "_internal/_rocm_sdk_libraries_custom/lib/rocblas/library/TensileLibrary.dat",
            "_internal/_rocm_sdk_libraries_custom/lib/miopen/db/kernels.kdb",
            # Windows path separators must be handled too.
            "_internal\\torch\\lib\\rccl.dll",
        ],
    )
    def test_runtime_files_are_rocm(self, rel_path):
        assert package_rocm.is_rocm_file(rel_path) is True

    @pytest.mark.parametrize(
        "rel_path",
        [
            "voicebox-server-rocm.exe",
            "_internal/python312.dll",
            "_internal/torch/lib/torch_cpu.dll",
            "_internal/torch/lib/c10.dll",
            # Pure-python rocm_sdk glue stays in the core, even under an SDK dir.
            "_internal/rocm_sdk/__init__.py",
            "_internal/_rocm_sdk_core/_dist_info.py",
            "_internal/torch/_inductor/codegen/something.py",
        ],
    )
    def test_core_files_are_not_rocm(self, rel_path):
        assert package_rocm.is_rocm_file(rel_path) is False


def _write(path: Path, content: bytes = b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


class TestPackage:
    """End-to-end split of a synthetic onedir into the two archives."""

    def test_split_and_manifest(self, tmp_path):
        onedir = tmp_path / "voicebox-server-rocm"
        _write(onedir / "voicebox-server-rocm.exe")
        _write(onedir / "_internal" / "python312.dll")
        _write(onedir / "_internal" / "rocm_sdk" / "__init__.py")
        _write(onedir / "_internal" / "torch" / "lib" / "torch_cpu.dll")
        _write(onedir / "_internal" / "torch" / "lib" / "amdhip64.dll")
        _write(onedir / "_internal" / "_rocm_sdk_core" / "miopen.dll")
        _write(
            onedir
            / "_internal"
            / "_rocm_sdk_libraries_custom"
            / "lib"
            / "rocblas"
            / "library"
            / "TensileLibrary.dat"
        )

        out = tmp_path / "release-assets"
        package_rocm.package(onedir, out, "rocm7.2-v1", ">=2.9.0,<2.10.0")

        server = out / "voicebox-server-rocm.tar.gz"
        libs = out / "rocm-libs-rocm7.2-v1.tar.gz"
        assert server.exists()
        assert libs.exists()
        assert (out / "voicebox-server-rocm.tar.gz.sha256").exists()
        assert (out / "rocm-libs-rocm7.2-v1.tar.gz.sha256").exists()

        with tarfile.open(libs) as tar:
            lib_names = set(tar.getnames())
        with tarfile.open(server) as tar:
            core_names = set(tar.getnames())

        assert "_internal/torch/lib/amdhip64.dll" in lib_names
        assert "_internal/_rocm_sdk_core/miopen.dll" in lib_names
        assert (
            "_internal/_rocm_sdk_libraries_custom/lib/rocblas/library/TensileLibrary.dat"
            in lib_names
        )
        assert "voicebox-server-rocm.exe" in core_names
        assert "_internal/torch/lib/torch_cpu.dll" in core_names
        assert "_internal/rocm_sdk/__init__.py" in core_names
        # Archives must be disjoint.
        assert lib_names.isdisjoint(core_names)

    def test_empty_rocm_set_exits(self, tmp_path):
        onedir = tmp_path / "voicebox-server-rocm"
        _write(onedir / "voicebox-server-rocm.exe")
        _write(onedir / "_internal" / "torch" / "lib" / "torch_cpu.dll")

        with pytest.raises(SystemExit):
            package_rocm.package(onedir, tmp_path / "out", "rocm7.2-v1", ">=2.9.0,<2.10.0")
