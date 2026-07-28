#!/bin/bash
# Setup script for voicebox backend development
# Mirrors the Dockerfile's Python package installation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Installing backend dependencies..."

# Install main requirements
pip install -r "$PROJECT_DIR/backend/requirements.txt"

# Install packages that need --no-deps due to incompatible pinned versions
# (See comments in requirements.txt for details)
pip install --no-deps chatterbox-tts
pip install --no-deps hume-tada

echo "Backend setup complete!"
