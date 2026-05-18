# ============================================================
# Voicebox — Local TTS Server with Web UI
# 3-stage build: Frontend → Python deps → Runtime
#
# Build variants:
#   CPU (default):  docker compose up --build
#   ROCm (AMD GPU): docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
# ============================================================

# Top-level ARG so it is visible to all stages.
ARG PYTORCH_VARIANT=cpu

# === Stage 1: Build frontend ===
FROM oven/bun:1 AS frontend

WORKDIR /build

# Copy workspace config and frontend source
COPY package.json bun.lock CHANGELOG.md ./
COPY app/ ./app/
COPY web/ ./web/

# Strip workspaces not needed for web build, and fix trailing comma
RUN sed -i '/"tauri"/d; /"landing"/d' package.json && \
    sed -i -z 's/,\n  ]/\n  ]/' package.json
RUN bun install --no-save
# Build frontend (skip tsc — upstream has pre-existing type errors)
RUN cd web && bunx --bun vite build


# === Stage 2: Build Python dependencies ===
FROM python:3.11-slim AS backend-builder

# Re-declare ARG inside the stage (Docker scoping requirement).
ARG PYTORCH_VARIANT=cpu

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY backend/requirements.txt .

# ROCm version to pull PyTorch wheels for. Default is 6.3 (supports RDNA1/2/3).
# Set ROCM_VERSION=7.2 for RDNA 4 (RX 9000 series) support.
ARG ROCM_VERSION=6.3

# When building the ROCm variant, install the ROCm-enabled PyTorch wheels
# first so that the subsequent requirements.txt install sees them as already
# satisfying the torch/torchaudio constraints and leaves them in place.
# The CPU path skips this step and installs torch from PyPI as before.
RUN if [ "$PYTORCH_VARIANT" = "rocm" ]; then \
      pip install --no-cache-dir --prefix=/install \
        torch torchaudio \
        --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}"; \
    fi

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install --no-deps chatterbox-tts
RUN pip install --no-cache-dir --prefix=/install --no-deps hume-tada
RUN pip install --no-cache-dir --prefix=/install \
    git+https://github.com/QwenLM/Qwen3-TTS.git


# === Stage 3: Runtime ===
FROM python:3.11-slim

# Re-declare ARG inside the stage (Docker scoping requirement).
ARG PYTORCH_VARIANT=cpu

# ROCm device access requires the container user to belong to the render
# and video groups. GIDs are parameterised to match the host; Ubuntu 22.04+
# defaults are used here. Override via env vars (docker-compose.rocm.yml
# passes them through automatically):
#   export RENDER_GID=$(getent group render | cut -d: -f3)
#   export VIDEO_GID=$(getent group video  | cut -d: -f3)
ARG RENDER_GID=992
ARG VIDEO_GID=44
RUN if [ "$PYTORCH_VARIANT" = "rocm" ]; then \
      groupadd -f -g ${RENDER_GID} render && \
      groupadd -f -g ${VIDEO_GID}  video; \
    fi

# Create non-root user for security
RUN groupadd -r voicebox && \
    useradd -r -g voicebox -m -s /bin/bash voicebox

# ROCm: add voicebox user to render+video so it can open /dev/kfd and /dev/dri.
RUN if [ "$PYTORCH_VARIANT" = "rocm" ]; then \
      usermod -aG render,video voicebox; \
    fi

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=backend-builder /install /usr/local

# Copy backend application code
COPY --chown=voicebox:voicebox backend/ /app/backend/

# Copy built frontend from frontend stage
COPY --from=frontend --chown=voicebox:voicebox /build/web/dist /app/frontend/

# Create data directories owned by non-root user
RUN mkdir -p /app/data/generations /app/data/profiles /app/data/cache \
    && chown -R voicebox:voicebox /app/data

# Switch to non-root user
USER voicebox

# Expose the API port
EXPOSE 17493

# Health check — auto-restart if the server hangs
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:17493/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "17493"]
