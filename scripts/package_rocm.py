"""
Package the PyInstaller --onedir ROCm build into two archives.

Takes the PyInstaller --onedir output directory and splits it into
(names carry a platform token, e.g. linux-x86_64, via --platform):
  1. voicebox-server-rocm-{platform}.tar.gz  — server core (exe + non-AMD deps)
  2. rocm-libs-{platform}-{version}.tar.gz   — AMD/ROCm runtime libraries only
  3. rocm-libs.json                          — version manifest for the ROCm libs

Mirrors scripts/package_cuda.py. The split lets the server core re-download on
every app update while the much larger ROCm runtime stays cached until the
toolkit version bumps.

Usage:
    python scripts/package_rocm.py backend/dist/voicebox-server-rocm/
    python scripts/package_rocm.py backend/dist/voicebox-server-rocm/ --output release-assets/
    python scripts/package_rocm.py backend/dist/voicebox-server-rocm/ --rocm-libs-version rocm7.2-v1
"""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

# DLL/.so name prefixes that identify AMD ROCm/HIP runtime libraries. They may
# sit in torch/lib/ (torch's bundled HIP runtime) or inside the bundled ROCm SDK
# packages. Matched case-insensitively against the file's base name.
ROCM_DLL_PREFIXES = (
    "amdhip",
    "amd_comgr",
    "amdocl",
    "hiprtc",
    "hipblaslt",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsolver",
    "hipsparse",
    "hip",
    "rocblas",
    "rocfft",
    "rocrand",
    "rocsolver",
    "rocsparse",
    "rocroller",
    "rocprofiler",
    "roctracer",
    "roctx",
    "rocm_smi",
    "miopen",
    "rccl",
    "hsa-runtime",
    "hsa",
    "aotriton",
)

# Directory markers for the bundled ROCm SDK runtime packages. Everything under
# these trees except Python sources (the pure-python rocm_sdk glue) is part of
# the runtime payload — this is where rocBLAS Tensile data and MIOpen kernel
# databases live, which dominate the download size.
#
# Windows: the ROCm runtime ships as separate rocm_sdk wheels (the _rocm_sdk_*
# trees). Linux: the download.pytorch.org/whl/rocm torch wheels bundle the
# runtime kernel data under torch/lib/<library>/ instead (rocBLAS/hipBLASLt
# Tensile data ~5 GB, aotriton kernel images ~0.9 GB). Any file under either
# tree is runtime payload and belongs in the cached libs archive.
ROCM_LIB_DIR_MARKERS = (
    "_rocm_sdk_core",
    "_rocm_sdk_libraries_custom",
    "rocm_sdk_core",
    "rocm_sdk_libraries_custom",
    "torch/lib/rocblas/",
    "torch/lib/hipblaslt/",
    "torch/lib/hipsparselt/",
    "torch/lib/aotriton.images/",
    "torch/share/miopen/",
)

# Python sources stay in the server core so the rocm_sdk import glue remains
# alongside the exe. (Both archives extract into backends/rocm/, so this only
# affects which archive carries the file, not runtime resolution.)
_PYTHON_EXTS = (".py", ".pyc", ".pyi")


def is_rocm_file(rel_path: str) -> bool:
    """Check if a relative path belongs to the AMD ROCm runtime libraries.

    Identifies large ROCm/HIP runtime DLLs and the SDK runtime payload
    (kernel databases, Tensile data) regardless of where PyInstaller placed
    them, while keeping pure-python glue in the server core.
    """
    rel_lower = rel_path.lower().replace("\\", "/")
    name = rel_lower.rsplit("/", 1)[-1]

    # Never split out Python sources / stubs.
    if name.endswith(_PYTHON_EXTS):
        return False

    # Everything under a ROCm runtime payload tree (Windows rocm_sdk wheels or
    # Linux torch/lib/<library>/ kernel data) is cached-libs content.
    if any(marker in rel_lower for marker in ROCM_LIB_DIR_MARKERS):
        return True

    # ROCm/HIP shared objects anywhere. Windows names them amdhip64.dll; Linux
    # names them libamdhip64.so / libaotriton_v2.so.0.11.2 — a leading "lib" and
    # a version suffix the simple ".dll"/".so" split would otherwise miss.
    if name.endswith(".dll") or ".so" in name:
        stem = name.split(".dll", 1)[0].split(".so", 1)[0].removeprefix("lib")
        for prefix in ROCM_DLL_PREFIXES:
            if stem.startswith(prefix):
                return True

    return False


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def platform_tag() -> str:
    """Platform token for release-asset names (mirrors backend server_asset_platform)."""
    import platform

    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    return f"{platform.system().lower()}-{arch}"


def package(
    onedir_path: Path,
    output_dir: Path,
    rocm_libs_version: str,
    torch_compat: str,
    plat: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all files in the onedir output, split into core vs rocm.
    core_files = []
    rocm_files = []

    for item in sorted(onedir_path.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(onedir_path)
        rel_str = str(rel)
        if is_rocm_file(rel_str):
            rocm_files.append((rel_str, item))
        else:
            core_files.append((rel_str, item))

    core_size = sum(f.stat().st_size for _, f in core_files)
    rocm_size = sum(f.stat().st_size for _, f in rocm_files)

    print(f"Input directory: {onedir_path}")
    print(f"Core files: {len(core_files)} ({core_size / (1024**2):.1f} MB)")
    print(f"ROCm files: {len(rocm_files)} ({rocm_size / (1024**2):.1f} MB)")

    if not rocm_files:
        print(
            f"ERROR: No ROCm files found in {onedir_path}. "
            "Refusing to create an empty ROCm libs archive.",
            file=sys.stderr,
        )
        print(
            "Make sure you built with --rocm and the ROCm SDK packages are present. "
            "If the layout differs, adjust ROCM_DLL_PREFIXES / ROCM_LIB_DIR_MARKERS.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create server core archive. Files are stored relative to the archive root
    # (no parent prefix) so extracting to backends/rocm/ lands at the right level.
    server_name = f"voicebox-server-rocm-{plat}.tar.gz"
    server_archive = output_dir / server_name
    print(f"\nCreating server core archive: {server_archive.name}")
    with tarfile.open(server_archive, "w:gz") as tar:
        for rel_str, full_path in core_files:
            tar.add(full_path, arcname=rel_str)
    server_sha = sha256_file(server_archive)
    (output_dir / f"{server_name}.sha256").write_text(
        f"{server_sha}  {server_name}\n"
    )
    print(f"  Size: {server_archive.stat().st_size / (1024**2):.1f} MB")
    print(f"  SHA-256: {server_sha[:16]}...")

    # Create ROCm libs archive.
    rocm_libs_name = f"rocm-libs-{plat}-{rocm_libs_version}.tar.gz"
    rocm_libs_archive = output_dir / rocm_libs_name
    print(f"\nCreating ROCm libs archive: {rocm_libs_archive.name}")
    with tarfile.open(rocm_libs_archive, "w:gz") as tar:
        for rel_str, full_path in rocm_files:
            tar.add(full_path, arcname=rel_str)
    rocm_sha = sha256_file(rocm_libs_archive)
    (output_dir / f"{rocm_libs_name}.sha256").write_text(
        f"{rocm_sha}  {rocm_libs_name}\n"
    )
    print(f"  Size: {rocm_libs_archive.stat().st_size / (1024**2):.1f} MB")
    print(f"  SHA-256: {rocm_sha[:16]}...")

    # Write rocm-libs.json manifest.
    manifest = {
        "version": rocm_libs_version,
        "torch_compat": torch_compat,
        "archive": rocm_libs_archive.name,
        "sha256": rocm_sha,
    }
    manifest_path = output_dir / "rocm-libs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest: {manifest_path.name}")
    print(json.dumps(manifest, indent=2))

    # Summary
    total_input = core_size + rocm_size
    total_output = server_archive.stat().st_size + rocm_libs_archive.stat().st_size
    print(f"\nTotal input:  {total_input / (1024**3):.2f} GB")
    print(f"Total output: {total_output / (1024**3):.2f} GB (compressed)")
    print(
        f"Server core:  {server_archive.stat().st_size / (1024**2):.1f} MB (redownloaded on app update)"
    )
    print(
        f"ROCm libs:    {rocm_libs_archive.stat().st_size / (1024**2):.1f} MB (cached until ROCm toolkit bump)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Package PyInstaller --onedir ROCm build into server + ROCm libs archives"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to PyInstaller --onedir output directory (e.g. backend/dist/voicebox-server-rocm/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for archives (default: same as input parent)",
    )
    parser.add_argument(
        "--rocm-libs-version",
        type=str,
        default="rocm7.2-v1",
        help="Version string for the ROCm libs archive (default: rocm7.2-v1)",
    )
    parser.add_argument(
        "--torch-compat",
        type=str,
        default=">=2.9.0,<2.10.0",
        help="Torch version compatibility range (default: >=2.9.0,<2.10.0)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Platform token for asset names, e.g. linux-x86_64 "
        "(default: auto-detect from the current host)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory", file=sys.stderr)
        print("Expected a PyInstaller --onedir output directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or args.input.parent
    plat = args.platform or platform_tag()
    package(args.input, output_dir, args.rocm_libs_version, args.torch_compat, plat)


if __name__ == "__main__":
    main()
