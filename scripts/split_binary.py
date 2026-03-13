"""
Split a large binary into chunks for GitHub Releases (<2 GB each).

Usage:
    python scripts/split_binary.py backend/dist/voicebox-server-cuda.exe
    python scripts/split_binary.py backend/dist/voicebox-server-cuda.exe --chunk-size 1900000000
    python scripts/split_binary.py backend/dist/voicebox-server-cuda.exe --output release-assets/

The script produces:
    - voicebox-server-cuda.part00.exe, .part01.exe, ...  (binary chunks)
    - voicebox-server-cuda.sha256      (SHA-256 checksum of the complete file)
    - voicebox-server-cuda.manifest    (ordered list of part filenames)
"""

import argparse
import hashlib
import sys
from pathlib import Path


def split(input_path: Path, chunk_size: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = input_path.read_bytes()
    total_size = len(data)

    # Write SHA-256 of the complete file
    sha256 = hashlib.sha256(data).hexdigest()
    checksum_file = output_dir / f"{input_path.stem}.sha256"
    checksum_file.write_text(f"{sha256}  {input_path.name}\n")

    # Split into chunks
    parts = []
    for i in range(0, total_size, chunk_size):
        part_index = len(parts)
        part_name = f"{input_path.stem}.part{part_index:02d}{input_path.suffix}"
        part_path = output_dir / part_name
        part_path.write_bytes(data[i:i + chunk_size])
        parts.append(part_name)

    # Write manifest (ordered list of part filenames)
    manifest_file = output_dir / f"{input_path.stem}.manifest"
    manifest_file.write_text("\n".join(parts) + "\n")

    print(f"Input:    {input_path} ({total_size / (1024**3):.2f} GB)")
    print(f"Output:   {output_dir}/")
    print(f"Parts:    {len(parts)} (chunk size: {chunk_size / (1024**3):.2f} GB)")
    print(f"SHA-256:  {sha256}")
    print(f"Manifest: {manifest_file.name}")
    for p in parts:
        size = (output_dir / p).stat().st_size
        print(f"  {p}  ({size / (1024**3):.2f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Split a large binary into chunks for GitHub Releases"
    )
    parser.add_argument("input", type=Path, help="Path to the binary file to split")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_900_000_000,  # 1.9 GB — safely under 2 GB GitHub limit
        help="Maximum chunk size in bytes (default: 1.9 GB)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} does not exist", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or args.input.parent
    split(args.input, args.chunk_size, output_dir)


if __name__ == "__main__":
    main()
