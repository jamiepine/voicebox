"""
Package the PyInstaller --onedir CUDA build into two archives.

Takes the PyInstaller --onedir output directory and splits it into:
  1. voicebox-server-cuda.tar.gz  — server core (exe + non-NVIDIA deps)
  2. cuda-libs-cu128.tar.gz       — NVIDIA runtime libraries only
  3. cuda-libs.json                — version manifest for the CUDA libs

Usage:
    python scripts/package_cuda.py backend/dist/voicebox-server-cuda/
    python scripts/package_cuda.py backend/dist/voicebox-server-cuda/ --output release-assets/
    python scripts/package_cuda.py backend/dist/voicebox-server-cuda/ --cuda-libs-version cu128-v1
"""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

# DLL name prefixes that identify NVIDIA CUDA runtime libraries.
# These DLLs may appear in different locations depending on the torch
# and PyInstaller version:
#   - nvidia/ subdirectories (older torch with separate nvidia-* packages)
#   - _internal/torch/lib/ (torch 2.10+ bundles NVIDIA DLLs directly)
#   - Top-level directory (some PyInstaller versions)
NVIDIA_DLL_PREFIXES = (
    "cublas",
    "cublaslt",
    "cudart",
    "cudnn",
    "cufft",
    "cufftw",
    "curand",
    "cusolver",
    "cusolvermg",
    "cusparse",
    "nvjitlink",
    "nvrtc",
    "nccl",
    "caffe2_nvrtc",
)

# Files to keep in the server core even if they match NVIDIA prefixes.
# These are small Python modules or stubs, not the large runtime DLLs.
NVIDIA_KEEP_IN_CORE = {
    "torch/cuda/nccl.py",
    "torch/_inductor/codegen/cuda/cutlass_lib_extensions/cutlass_mock_imports/cuda/cudart.py",
}


def is_nvidia_file(rel_path: str) -> bool:
    """Check if a relative path belongs to the NVIDIA CUDA libs.

    Identifies large NVIDIA runtime DLLs (.dll/.so) regardless of where
    PyInstaller placed them. Excludes small Python stubs that happen to
    share NVIDIA-related names.
    """
    rel_lower = rel_path.lower().replace("\\", "/")

    # Never split out Python source files or small stubs
    if rel_lower in NVIDIA_KEEP_IN_CORE:
        return False

    # Files under nvidia/ subdirectory tree (older torch layout)
    if rel_lower.startswith("nvidia/") or "/nvidia/" in rel_lower:
        # Only DLLs/shared objects — not .py, .dist-info, etc.
        if rel_lower.endswith((".dll", ".so")):
            return True
        # Include entire nvidia/ namespace package tree
        for part in rel_lower.split("/"):
            if part == "nvidia":
                return True

    # NVIDIA DLLs anywhere in the tree (e.g. _internal/torch/lib/cublas64_12.dll)
    name = rel_lower.rsplit("/", 1)[-1]
    if name.endswith(".dll") or name.endswith(".so"):
        name_no_ext = name.rsplit(".", 1)[0]
        for prefix in NVIDIA_DLL_PREFIXES:
            if name_no_ext.startswith(prefix):
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


def package(
    onedir_path: Path,
    output_dir: Path,
    cuda_libs_version: str,
    torch_compat: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all files in the onedir output, split into core vs nvidia
    core_files = []
    nvidia_files = []

    for item in sorted(onedir_path.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(onedir_path)
        rel_str = str(rel)
        if is_nvidia_file(rel_str):
            nvidia_files.append((rel_str, item))
        else:
            core_files.append((rel_str, item))

    core_size = sum(f.stat().st_size for _, f in core_files)
    nvidia_size = sum(f.stat().st_size for _, f in nvidia_files)

    print(f"Input directory: {onedir_path}")
    print(f"Core files:  {len(core_files)} ({core_size / (1024**2):.1f} MB)")
    print(f"NVIDIA files: {len(nvidia_files)} ({nvidia_size / (1024**2):.1f} MB)")

    if not nvidia_files:
        print(
            f"ERROR: No NVIDIA files found in {onedir_path}. "
            "Refusing to create an empty CUDA libs archive.",
            file=sys.stderr,
        )
        print(
            "Make sure you built with --cuda and the NVIDIA packages are present.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create server core archive
    # Files are stored relative to the archive root (no parent directory prefix)
    # so extracting to backends/cuda/ puts everything at the right level.
    server_archive = output_dir / "voicebox-server-cuda.tar.gz"
    print(f"\nCreating server core archive: {server_archive.name}")
    with tarfile.open(server_archive, "w:gz") as tar:
        for rel_str, full_path in core_files:
            tar.add(full_path, arcname=rel_str)
    server_sha = sha256_file(server_archive)
    (output_dir / "voicebox-server-cuda.tar.gz.sha256").write_text(
        f"{server_sha}  voicebox-server-cuda.tar.gz\n"
    )
    print(f"  Size: {server_archive.stat().st_size / (1024**2):.1f} MB")
    print(f"  SHA-256: {server_sha[:16]}...")

    # Create CUDA libs archive
    cuda_libs_archive = output_dir / f"cuda-libs-{cuda_libs_version}.tar.gz"
    print(f"\nCreating CUDA libs archive: {cuda_libs_archive.name}")
    with tarfile.open(cuda_libs_archive, "w:gz") as tar:
        for rel_str, full_path in nvidia_files:
            tar.add(full_path, arcname=rel_str)
    cuda_sha = sha256_file(cuda_libs_archive)
    (output_dir / f"cuda-libs-{cuda_libs_version}.tar.gz.sha256").write_text(
        f"{cuda_sha}  cuda-libs-{cuda_libs_version}.tar.gz\n"
    )
    print(f"  Size: {cuda_libs_archive.stat().st_size / (1024**2):.1f} MB")
    print(f"  SHA-256: {cuda_sha[:16]}...")

    # Write cuda-libs.json manifest
    manifest = {
        "version": cuda_libs_version,
        "torch_compat": torch_compat,
        "archive": cuda_libs_archive.name,
        "sha256": cuda_sha,
    }
    manifest_path = output_dir / "cuda-libs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest: {manifest_path.name}")
    print(json.dumps(manifest, indent=2))

    # Summary
    total_input = core_size + nvidia_size
    total_output = server_archive.stat().st_size + cuda_libs_archive.stat().st_size
    print(f"\nTotal input:  {total_input / (1024**3):.2f} GB")
    print(f"Total output: {total_output / (1024**3):.2f} GB (compressed)")
    print(
        f"Server core:  {server_archive.stat().st_size / (1024**2):.1f} MB (redownloaded on app update)"
    )
    print(
        f"CUDA libs:    {cuda_libs_archive.stat().st_size / (1024**2):.1f} MB (cached until CUDA toolkit bump)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Package PyInstaller --onedir CUDA build into server + CUDA libs archives"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to PyInstaller --onedir output directory (e.g. backend/dist/voicebox-server-cuda/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for archives (default: same as input parent)",
    )
    parser.add_argument(
        "--cuda-libs-version",
        type=str,
        default="cu128-v1",
        help="Version string for the CUDA libs archive (default: cu128-v1)",
    )
    parser.add_argument(
        "--torch-compat",
        type=str,
        default=">=2.7.0,<2.11.0",
        help="Torch version compatibility range (default: >=2.6.0,<2.11.0)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory", file=sys.stderr)
        print("Expected a PyInstaller --onedir output directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or args.input.parent
    package(args.input, output_dir, args.cuda_libs_version, args.torch_compat)


if __name__ == "__main__":
    main()
