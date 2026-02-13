# Voicebox TTS Server
# CUDA 12.1 + Python 3.11 on Ubuntu 22.04
#
# Build:
#   docker build -t voicebox .
#   docker build --build-arg CUDA=0 -t voicebox-cpu .
#
# Run:
#   docker compose up -d

ARG CUDA=1

# --- Base stage ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base-cuda
FROM ubuntu:22.04 AS base-cpu

# --- Pick base based on CUDA arg ---
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

COPY requirements.txt .

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
COPY __init__.py /app/backend/__init__.py
COPY main.py /app/backend/main.py
COPY server.py /app/backend/server.py
COPY config.py /app/backend/config.py
COPY database.py /app/backend/database.py
COPY models.py /app/backend/models.py
COPY tts.py /app/backend/tts.py
COPY transcribe.py /app/backend/transcribe.py
COPY profiles.py /app/backend/profiles.py
COPY history.py /app/backend/history.py
COPY export_import.py /app/backend/export_import.py
COPY channels.py /app/backend/channels.py
COPY stories.py /app/backend/stories.py
COPY cli.py /app/backend/cli.py
COPY backends/ /app/backend/backends/
COPY utils/ /app/backend/utils/

# Data volume
RUN mkdir -p /data
VOLUME /data

# HuggingFace cache — models download here
ENV HF_HOME=/data/huggingface

EXPOSE 17493

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:17493/health || exit 1

ENTRYPOINT ["python3", "-m", "backend.main"]
CMD ["--host", "0.0.0.0", "--port", "17493", "--data-dir", "/data"]
