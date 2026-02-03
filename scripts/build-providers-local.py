#!/usr/bin/env python3
"""
Build and install all TTS providers locally for development.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Build and install all providers."""
    project_root = Path(__file__).parent.parent
    providers_dir = project_root / "providers"

    providers = ["pytorch-cpu", "pytorch-cuda"]

    for provider in providers:
        provider_path = providers_dir / provider
        script_path = provider_path / "build_and_install.py"

        if not script_path.exists():
            print(f"⚠ Skipping {provider}: build_and_install.py not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Building and installing {provider}...")
        print(f"{'=' * 60}\n")

        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=provider_path,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to build {provider}: {e}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("✓ All providers built and installed successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
