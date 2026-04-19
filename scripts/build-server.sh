#!/bin/bash
# Build Python server binary for all platforms

set -e

# Determine platform
PLATFORM=$(rustc --print host-tuple 2>/dev/null || echo "unknown")

echo "Building voicebox-server for platform: $PLATFORM"

# Build Python binary
# Resolve PATH to absolute paths before changing directory
export PATH="$(cd "$(dirname "$0")/.." && pwd)/backend/venv/bin:$PATH"
cd backend

# Check if PyInstaller is installed
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    python -m pip install pyinstaller
fi

# Build binary
python build_binary.py

# Create binaries directory if it doesn't exist
mkdir -p ../tauri/src-tauri/binaries

# Copy binary with platform suffix
if [ -f dist/voicebox-server ]; then
    cp dist/voicebox-server ../tauri/src-tauri/binaries/voicebox-server-${PLATFORM}
    chmod +x ../tauri/src-tauri/binaries/voicebox-server-${PLATFORM}
    echo "Built voicebox-server-${PLATFORM}"
elif [ -f dist/voicebox-server.exe ]; then
    cp dist/voicebox-server.exe ../tauri/src-tauri/binaries/voicebox-server-${PLATFORM}.exe
    echo "Built voicebox-server-${PLATFORM}.exe"
else
    echo "Error: Binary not found in dist/"
    exit 1
fi

echo "Build complete!"
