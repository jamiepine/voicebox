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

# ROCm wheel index. Default 6.3 (RDNA1/2/3); set ROCM_VERSION=7.2 for RDNA4.
ARG ROCM_VERSION=6.3

# For ROCm, make the PyTorch ROCm index primary so every install below resolves
# torch to ROCm wheels instead of the default CUDA build. Fix it to torch 2.7.1 as
# it is actually the only working version for rocm. 
RUN if [ "$PYTORCH_VARIANT" = "rocm" ]; then \
      pip install --no-cache-dir --prefix=/install \
        --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && \
      printf '[global]\nindex-url = https://download.pytorch.org/whl/rocm%s\nextra-index-url = https://pypi.org/simple\n' "$ROCM_VERSION" > /etc/pip.conf; \
    fi

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install --no-deps chatterbox-tts
RUN pip install --no-cache-dir --prefix=/install --no-deps hume-tada
RUN pip install --no-cache-dir --prefix=/install \
    git+https://github.com/QwenLM/Qwen3-TTS.git

# Qwen3-TTS's setup.py pulls in plain PyPI "triton" (CUDA-oriented) as an
# unconstrained dependency, alongside its nvidia-cuda-*/cuda-bindings/
# cuda-toolkit sub-deps. This shadows the correct pytorch-triton-rocm
# that shipped with our pinned ROCm torch, and breaks at runtime with
# "libcudart.so.13: cannot open shared object file" since no real NVIDIA
# driver is present. Strip the stray CUDA packages on ROCm builds.
RUN if [ "$PYTORCH_VARIANT" = "rocm" ]; then \
      SITE=/install/lib/python3.11/site-packages && \
      rm -rf "$SITE"/triton "$SITE"/triton-*.dist-info \
             "$SITE"/nvidia "$SITE"/nvidia_*.dist-info \
             "$SITE"/cuda_bindings "$SITE"/cuda_bindings-*.dist-info \
             "$SITE"/cuda_pathfinder "$SITE"/cuda_pathfinder-*.dist-info \
             "$SITE"/cuda_toolkit "$SITE"/cuda_toolkit-*.dist-info \
             "$SITE"/torch "$SITE"/torch-*.dist-info \
             "$SITE"/torchaudio "$SITE"/torchaudio-*.dist-info \
             "$SITE"/torchvision "$SITE"/torchvision-*.dist-info \
             "$SITE"/functorch \
             "$SITE"/*.dist-info/../nvidia* 2>/dev/null; \
      true; \
    fi

# Re-pin torch/torchaudio to the ROCm build. requirements.txt (unpinned
# torch/torchvision) can pull the default CUDA wheels from pypi.org via
# the extra-index-url fallback and silently clobber the ROCm install above.
RUN if [ "$PYTORCH_VARIANT" = "rocm" ]; then \
      PIP_EXTRA_INDEX_URL= pip install --no-cache-dir --prefix=/install --force-reinstall --no-deps \
        --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1; \
    fi

# === Stage 3: Runtime ===
FROM python:3.11-slim

# Create non-root user; the entrypoint joins GPU device groups at runtime.
RUN groupadd -r voicebox && \
    useradd -r -g voicebox -m -s /bin/bash voicebox

WORKDIR /app

# Install only runtime system dependencies (gosu drops root in the entrypoint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    gosu \
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
