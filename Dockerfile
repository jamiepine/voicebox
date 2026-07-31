# syntax=docker/dockerfile:1.2
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

# Normalize line endings first (a Windows CRLF checkout would otherwise
# defeat the `-z 's/,\n  ]/…/'` match below, since it's LF-anchored), then
# strip workspaces not needed for web build, and fix trailing comma
RUN sed -i 's/\r$//' package.json && \
    sed -i '/"tauri"/d; /"landing"/d' package.json && \
    sed -i -z 's/,\n  ]/\n  ]/' package.json && \
    bun install --no-save && \
    (cd web && bunx --bun vite build) # Build frontend (skip tsc — upstream has pre-existing type errors)


# === Stage 2: Build Python dependencies ===
FROM python:3.11-slim AS backend-builder

# Re-declare ARG inside the stage (Docker scoping requirement).
ARG PYTORCH_VARIANT=cpu

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

COPY backend/requirements.txt .

# ROCm wheel index. Default 6.3 (RDNA1/2/3); set ROCM_VERSION=7.2 for RDNA4.
ARG ROCM_VERSION=6.3

# CPU builds pin torch: 2.8+ breaks CPU inference in every engine tested
# (Kokoro/LuxTTS: "Cannot copy out of meta tensor"; Qwen: "unsupported
# scalarType" in torch.autocast). 2.7.1 is the newest that works and matches
# hume-tada's torch>=2.7,<2.8.
# For ROCm, the PyTorch ROCm index is primary so every install below resolves
# torch to ROCm wheels instead of the default CUDA build.
# The PyTorch wheel index also has compatible torchaudio versions, so all
# subsequent installs (including requirements.txt) resolve torchaudio correctly.
#
# --no-deps: Qwen3-TTS's dependency list would re-resolve torch/transformers
# and undo the versions installed above; everything it needs is already here.
RUN if [ "$PYTORCH_VARIANT" = "rocm" ]; then \
      pip install --no-cache-dir --prefix=/install \
        --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" \
        torch && \
      printf '[global]\nindex-url = https://download.pytorch.org/whl/rocm%s\nextra-index-url = https://pypi.org/simple\n' "$ROCM_VERSION" > /etc/pip.conf; \
    else \
      pip install --no-cache-dir --prefix=/install \
        --index-url "https://download.pytorch.org/whl/cpu" \
        torch==2.7.1 && \
      printf '[global]\nindex-url = https://download.pytorch.org/whl/cpu\nextra-index-url = https://pypi.org/simple\n' > /etc/pip.conf; \
    fi && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install --no-deps chatterbox-tts && \
    pip install --no-cache-dir --prefix=/install --no-deps hume-tada && \
    pip install --no-cache-dir --prefix=/install --no-deps \
    git+https://github.com/QwenLM/Qwen3-TTS.git@022e286b98fbec7e1e916cb940cdf532cd9f488e


# === Stage 3: Runtime ===
FROM python:3.11-slim

# Create non-root user; the entrypoint joins GPU device groups at runtime.
RUN groupadd -r voicebox && \
    useradd -r -g voicebox -m -s /bin/bash voicebox

WORKDIR /app

# Install only runtime system dependencies (gosu drops root in the entrypoint)
RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg gosu sox && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=backend-builder /install /usr/local

# Configure library paths for torch/torchaudio .so files
RUN printf "/usr/local/lib/python3.11/site-packages/torch/lib\n/usr/local/lib/python3.11/site-packages/torchaudio/lib\n" > /etc/ld.so.conf.d/torch.conf && \
    ldconfig

# Copy backend application code
COPY --chown=voicebox:voicebox backend/ /app/backend/

# Copy built frontend from frontend stage
COPY --from=frontend --chown=voicebox:voicebox /build/web/dist /app/frontend/

# Create data directories owned by non-root user
RUN mkdir -p /app/data/generations /app/data/profiles /app/data/cache && \
    chown -R voicebox:voicebox /app/data

# Expose the API port
EXPOSE 17493

# Health check — auto-restart if the server hangs
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:17493/health || exit 1

# Entrypoint joins GPU groups then drops to the voicebox user.
# Normalize CRLF (a Windows checkout otherwise leaves the shebang as
# `#!/bin/sh\r`, which Linux can't resolve — reported as a misleading
# "no such file or directory" even though the file exists).
COPY --chmod=755 scripts/rocm-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "17493"]
