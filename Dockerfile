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

# ROCm wheel index. Default 6.3 (RDNA1/2/3); set ROCM_VERSION=7.2 for RDNA4.
ARG ROCM_VERSION=6.3

# For ROCm, make the PyTorch ROCm index primary so every install below resolves
# torch to ROCm wheels instead of the default CUDA build.
RUN if [ "$PYTORCH_VARIANT" = "rocm" ]; then \
      pip install --no-cache-dir --prefix=/install \
        --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" \
        torch torchaudio && \
      printf '[global]\nindex-url = https://download.pytorch.org/whl/rocm%s\nextra-index-url = https://pypi.org/simple\n' "$ROCM_VERSION" > /etc/pip.conf; \
    fi

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install --no-deps chatterbox-tts
RUN pip install --no-cache-dir --prefix=/install --no-deps hume-tada
RUN pip install --no-cache-dir --prefix=/install \
    git+https://github.com/QwenLM/Qwen3-TTS.git


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

# Entrypoint joins GPU groups then drops to the voicebox user
COPY --chmod=755 scripts/rocm-entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "17493"]
