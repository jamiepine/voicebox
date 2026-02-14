# Voicebox TTS Server
# CUDA 12.1 + Python 3.11 on Ubuntu 22.04
#
# Build:
#   docker build -t voicebox .
#   docker build --build-arg CUDA=0 -t voicebox-cpu .
#
# Run:
#   docker compose up -d

ARG CUDA=0

# --- Base stage ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base-cuda
FROM ubuntu:22.04 AS base-cpu

# --- Pick base based on CUDA arg --
FROM base-cuda AS base-1
FROM base-cpu AS base-0
FROM base-${CUDA} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# --- Dependencies stage (cached layer) ---
FROM base AS deps

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

ARG CUDA=1
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    if [ "$CUDA" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# --- App stage ---
FROM deps AS app

WORKDIR /app

# Copy backend source as a package
COPY backend/ /app/backend/

# HuggingFace cache — models download here
ENV HF_HOME=/app/data/huggingface

EXPOSE 17493

HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:17493/health || exit 1

ENTRYPOINT ["python3", "-m", "backend.main"]
CMD ["--host", "0.0.0.0", "--port", "17493", "--data-dir", "/app/data"]
