#!/bin/bash
# Build and push the voicebox serverless Docker image for RunPod.
#
# Usage:
#   ./scripts/serverless_build.sh                        # build only
#   ./scripts/serverless_build.sh --push                 # build + push
#   ./scripts/serverless_build.sh --push --tag ghcr.io/you/voicebox-serverless:latest
#
# Environment:
#   RUNPOD_API_KEY  — set in .env at the project root (not used by the image itself)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICEBOX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
IMAGE_TAG="voicebox-serverless:latest"
PUSH=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)
            PUSH=1
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--push] [--tag IMAGE:TAG]"
            exit 1
            ;;
    esac
done

echo "Building voicebox serverless image: $IMAGE_TAG"
echo "Context: $VOICEBOX_DIR"

DOCKER_BUILDKIT=1 docker build \
    --build-arg CUDA=1 \
    --build-arg SERVERLESS=1 \
    -t "$IMAGE_TAG" \
    "$VOICEBOX_DIR"

echo "Build complete: $IMAGE_TAG"

if [ "$PUSH" = "1" ]; then
    echo "Pushing $IMAGE_TAG..."
    docker push "$IMAGE_TAG"
    echo "Pushed: $IMAGE_TAG"
    echo ""
    echo "Use this image URL when creating your RunPod endpoint."
fi
