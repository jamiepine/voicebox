"""
Build PyTorch CPU provider and install to local provider directory.
"""

import platform
import shutil
from pathlib import Path

from build import build_provider


def get_providers_dir() -> Path:
    """Get the directory where providers are stored."""
    system = platform.system()

    if system == "Windows":
        appdata = Path.home() / "AppData" / "Roaming"
    elif system == "Darwin":
        appdata = Path.home() / "Library" / "Application Support"
    else:  # Linux
        appdata = Path.home() / ".local" / "share"

    providers_dir = appdata / "voicebox" / "providers"
    providers_dir.mkdir(parents=True, exist_ok=True)
    return providers_dir


def main():
    """Build and install provider."""
    provider_dir = Path(__file__).parent

    # Build the provider
    print("Building PyTorch CPU provider...")
    build_provider()

    # Determine binary name
    binary_name = "tts-provider-pytorch-cpu"
    if platform.system() == "Windows":
        binary_name += ".exe"

    # Source and destination paths
    source = provider_dir / "dist" / binary_name
    destination = get_providers_dir() / binary_name

    # Copy to provider directory
    print(f"Installing to {destination}...")
    shutil.copy2(source, destination)

    # Make executable on Unix systems
    if platform.system() != "Windows":
        destination.chmod(0o755)

    print(f"✓ Provider installed successfully to {destination}")


if __name__ == "__main__":
    main()
