# ============================================================
# Voicebox — Local TTS Server with Web UI (CPU)
# 3-stage build: Frontend → Python deps → Runtime
# ============================================================

# Single source for the Python version. Keep equal to /.python-version —
# this is enforced in CI by scripts/check-python-version.sh.
ARG PYTHON_VERSION=3.12

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
FROM python:${PYTHON_VERSION}-slim AS backend-builder
ARG PYTHON_VERSION
# Pin external sources so image rebuilds are deterministic.
ARG UV_VERSION=0.11.23
ARG QWEN3_TTS_REF=022e286b98fbec7e1e916cb940cdf532cd9f488e

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (self-contained binary, shipped as a PyPI wheel) for fast,
# parallel dependency resolution. Installs into an isolated --prefix that the
# runtime stage copies onto the system path.
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

COPY backend/requirements.txt .
RUN uv pip install --no-cache --prefix=/install --python python${PYTHON_VERSION} -r requirements.txt
# chatterbox-tts pins numpy<1.26 / torch==2.6 — install without its deps
RUN uv pip install --no-cache --prefix=/install --python python${PYTHON_VERSION} --no-deps chatterbox-tts
# hume-tada pins torch>=2.7,<2.8 — install without its deps
RUN uv pip install --no-cache --prefix=/install --python python${PYTHON_VERSION} --no-deps hume-tada
RUN uv pip install --no-cache --prefix=/install --python python${PYTHON_VERSION} \
    git+https://github.com/QwenLM/Qwen3-TTS.git@${QWEN3_TTS_REF}


# === Stage 3: Runtime ===
FROM python:${PYTHON_VERSION}-slim

# Create non-root user for security
RUN groupadd -r voicebox && \
    useradd -r -g voicebox -m -s /bin/bash voicebox

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
