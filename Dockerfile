FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# System dependencies:
#   git        - required for pip install from GitHub (qwen-tts)
#   libsndfile1 - required by soundfile (audio I/O)
#   ffmpeg     - required by librosa (audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Install Python dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && pip install --no-cache-dir git+https://github.com/QwenLM/Qwen3-TTS.git

# Copy backend source
COPY backend/ backend/

# Data volume mount point
RUN mkdir -p /app/data

EXPOSE 8000

ENTRYPOINT ["python", "-m", "backend.main", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/app/data"]
