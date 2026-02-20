FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (audio libs, ffmpeg for processing)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy server entrypoint and package
COPY backend/server.py /app/server.py
COPY backend /app/backend

VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/data"]
