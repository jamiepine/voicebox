# ============================================================
# Voicebox — Local TTS Server with Web UI (CPU)
# 3-stage build: Frontend → Python deps → Runtime
# ============================================================

# === Stage 1: Build frontend ===
FROM oven/bun:1 AS frontend

WORKDIR /build

# Copy all needed files for frontend build
COPY app/ ./app/
COPY web/ ./web/
COPY bun.lock CHANGELOG.md ./

# Create a temporary package.json for container (with all root deps needed)
RUN cat > package-temp.json << 'EOFPKG' && mv package-temp.json package.json
{
  "name": "voicebox-container",
  "version": "0.5.0",
  "private": true,
  "workspaces": [
    "app",
    "web"
  ],
  "scripts": {
    "build:web": "cd web && bun run build"
  },
  "dependencies": {
    "loaders.css": "^0.1.2",
    "react-loaders": "^3.0.1"
  },
  "devDependencies": {
    "@biomejs/biome": "2.3.12",
    "@types/node": "^20.0.0",
    "tailwindcss": "^4.1.18",
    "typescript": "^5.6.0"
  }
}
EOFPKG

# Install workspace dependencies
RUN bun install --no-save

# Build frontend (skip tsc, upstream has type errors)
RUN cd web && bunx vite build


# === Stage 2: Build Python dependencies ===
FROM python:3.11-slim AS backend-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install --no-deps chatterbox-tts
RUN pip install --no-cache-dir --prefix=/install --no-deps hume-tada
RUN pip install --no-cache-dir --prefix=/install \
    git+https://github.com/QwenLM/Qwen3-TTS.git


# === Stage 3: Runtime ===
FROM python:3.11-slim

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
