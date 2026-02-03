# Base Dockerfile for Voicebox (CPU-only)
# For GPU support, use Dockerfile.cuda

FROM python:3.12-slim

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy backend
COPY backend/ /app/backend/
COPY providers/ /app/providers/

# Copy pre-built web UI
COPY web/dist/ /app/web/dist/

# Install Python dependencies (without PyTorch - will be downloaded via provider system)
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic sqlalchemy alembic \
    librosa soundfile numpy python-multipart Pillow \
    huggingface_hub transformers accelerate

# Create data directory for profiles/generations
RUN mkdir -p /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run server with web UI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
