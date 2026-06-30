#!/bin/bash
# Build Python server binary for all platforms

set -e

# Determine platform
PLATFORM=$(rustc --print host-tuple 2>/dev/null || echo "unknown")

echo "Building Voicebox sidecars for platform: $PLATFORM"

# Build Python binary
# Resolve PATH to absolute paths before changing directory
export PATH="$(cd "$(dirname "$0")/.." && pwd)/backend/venv/bin:$PATH"
cd backend

# Check if PyInstaller is installed
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    python -m pip install pyinstaller
fi

# Create binaries directory if it doesn't exist
mkdir -p ../tauri/src-tauri/binaries

copy_sidecar() {
    local name="$1"

    if [ -f "dist/${name}" ]; then
        cp "dist/${name}" "../tauri/src-tauri/binaries/${name}-${PLATFORM}"
        chmod +x "../tauri/src-tauri/binaries/${name}-${PLATFORM}"
        echo "Built ${name}-${PLATFORM}"
    elif [ -f "dist/${name}.exe" ]; then
        cp "dist/${name}.exe" "../tauri/src-tauri/binaries/${name}-${PLATFORM}.exe"
        echo "Built ${name}-${PLATFORM}.exe"
    else
        echo "Error: ${name} binary not found in dist/"
        exit 1
    fi
}

python build_binary.py
copy_sidecar voicebox-server

python build_binary.py --shim
copy_sidecar voicebox-mcp

echo "Build complete!"
