#!/bin/bash
# Build Python server binary for all platforms

set -e

# Determine platform
PLATFORM=$(rustc --print host-tuple 2>/dev/null || echo "unknown")

echo "Building voicebox-server for platform: $PLATFORM"

# Find the virtual environment
VENV_PATH="$(cd "$(dirname "$0")/.." && pwd)/.venv"
PYTHON_BIN="$VENV_PATH/bin/python3"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python not found in $VENV_PATH"
    exit 1
fi

# Build Python binary
cd backend

# Check if PyInstaller is installed
if ! "$PYTHON_BIN" -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    "$PYTHON_BIN" -m pip install pyinstaller
fi

# Build binary
"$PYTHON_BIN" build_binary.py

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
