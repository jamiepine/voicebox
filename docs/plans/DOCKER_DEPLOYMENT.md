# Docker Deployment Guide

**Status:** Implemented
**Images:** `ghcr.io/jamiepine/voicebox`

## Overview

Voicebox is available as Docker images with the full web UI included. Images are automatically built and published to GitHub Container Registry on each release.

**What's included:**
- FastAPI backend with all TTS/Whisper capabilities
- Complete web UI (same React app as the Tauri desktop version)
- Provider download system (downloads TTS providers on first use, just like desktop)
- Multi-architecture support (amd64, arm64 for CPU variant)

Docker support is ideal for:
- **Server Deployments**: Run on headless Linux servers
- **GPU Passthrough**: Easy NVIDIA GPU access
- **Consistent Environments**: Same setup across dev/staging/prod
- **Cloud Platforms**: Deploy to AWS, GCP, Azure, DigitalOcean
- **Multi-User Setups**: Isolate instances per user/team

## Quick Start

### Using Pre-Built Images (Recommended)

```bash
# CPU-only version (supports amd64 and arm64)
docker run -p 8000:8000 -v voicebox-data:/app/data \
  ghcr.io/jamiepine/voicebox:latest

# NVIDIA GPU version
docker run --gpus all -p 8000:8000 -v voicebox-data:/app/data \
  ghcr.io/jamiepine/voicebox:latest-cuda

# Specific version (pinned for stability)
docker run -p 8000:8000 -v voicebox-data:/app/data \
  ghcr.io/jamiepine/voicebox:0.1.13
```

Then open: `http://localhost:8000`

The web UI will load automatically. On first use, you'll be prompted to download a TTS provider (PyTorch CPU ~300MB or PyTorch CUDA ~2.4GB).

### Using Docker Compose (Easiest)

Use the provided `docker-compose.yml` (CUDA) or `docker-compose.cpu.yml` in the repository root:

```bash
# CUDA (default)
docker compose up -d

# Or CPU-only
docker compose -f docker-compose.cpu.yml up -d
```

To pin to a specific version, edit the compose file:
```yaml
services:
  voicebox:
    image: ghcr.io/jamiepine/voicebox:0.1.13-cuda  # Pinned version
```

## Building From Source

See `Dockerfile` and `Dockerfile.cuda` in the repository root.

Build and run:
```bash
# Build web UI first
bun install
cd web && bun run build && cd ..

# Build CPU image
docker build -t voicebox .
docker run -p 8000:8000 -v voicebox-data:/app/data voicebox

# Or build CUDA image
docker build -f Dockerfile.cuda -t voicebox:cuda .
docker run --gpus all -p 8000:8000 -v voicebox-data:/app/data voicebox:cuda
```

### Architecture

The Docker images include:
- **Backend**: FastAPI server with TTS/Whisper endpoints
- **Web UI**: Pre-built React app served as static files from the backend
- **Provider System**: Downloads PyTorch CPU/CUDA providers on first use (same UX as desktop app)

Images are automatically built on release and tagged with both version number and `latest`.

## GPU Support

### NVIDIA GPUs (CUDA)

The CUDA image includes PyTorch with CUDA 12.1 support:

**Run with GPU:**
```bash
docker run --gpus all -p 8000:8000 \
  -v voicebox-data:/app/data \
  ghcr.io/jamiepine/voicebox:latest-cuda
```

**Docker Compose with GPU:**
```yaml
services:
  voicebox:
    image: ghcr.io/jamiepine/voicebox:latest-cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### AMD GPUs (ROCm)

ROCm support is not currently available in pre-built images. If you need ROCm, build a custom image using the ROCm base and PyTorch ROCm builds.

## Volume Mounts

### Essential Volumes

```bash
docker run -v voicebox-data:/app/data \           # Profiles, generations, history
           -v huggingface-cache:/root/.cache/huggingface \  # Downloaded models
           -p 8000:8000 voicebox
```

### Development Volume Mounts

For development with hot-reload:

```bash
docker run -v $(pwd)/backend:/app/backend \       # Live code changes
           -v voicebox-data:/app/data \
           -e RELOAD=true \
           -p 8000:8000 voicebox
```

### Custom Model Storage

Use external model directory:

```bash
docker run -v /path/to/models:/models \
           -e MODELS_DIR=/models \
           -v voicebox-data:/app/data \
           -p 8000:8000 voicebox
```

## Environment Variables

Configure Voicebox via environment variables:

```bash
docker run -e TTS_MODE=local \
           -e WHISPER_MODE=openai-api \
           -e OPENAI_API_KEY=sk-... \
           -e GPU_MEMORY_FRACTION=0.8 \
           -e LOG_LEVEL=info \
           -p 8000:8000 voicebox
```

### Available Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_MODE` | `local` | TTS provider: `local`, `remote` |
| `TTS_REMOTE_URL` | - | URL for remote TTS server |
| `WHISPER_MODE` | `local` | Whisper provider: `local`, `openai-api`, `remote` |
| `WHISPER_REMOTE_URL` | - | URL for remote Whisper server |
| `OPENAI_API_KEY` | - | OpenAI API key (if using OpenAI Whisper) |
| `GPU_MEMORY_FRACTION` | `0.9` | Fraction of GPU memory to use (0.0-1.0) |
| `DATA_DIR` | `/app/data` | Directory for profiles/generations |
| `MODELS_DIR` | `/app/models` | Directory for local models |
| `LOG_LEVEL` | `info` | Logging level: `debug`, `info`, `warning`, `error` |
| `RELOAD` | `false` | Enable hot-reload for development |

## Complete Docker Compose Examples

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  voicebox:
    image: ghcr.io/jamiepine/voicebox:latest-cuda
    container_name: voicebox
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - voicebox-data:/app/data
      - huggingface-cache:/root/.cache/huggingface
    environment:
      - TTS_MODE=local
      - WHISPER_MODE=local
      - GPU_MEMORY_FRACTION=0.8
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  voicebox-data:
    driver: local
  huggingface-cache:
    driver: local
```

Run:
```bash
docker compose -f docker-compose.prod.yml up -d
```

### Development Setup

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  voicebox:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend:ro
      - voicebox-data:/app/data
      - huggingface-cache:/root/.cache/huggingface
    environment:
      - RELOAD=true
      - LOG_LEVEL=debug
      - TTS_MODE=local
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  voicebox-data:
  huggingface-cache:
```

### Multi-Service Stack

Full stack with reverse proxy and monitoring:

```yaml
# docker-compose.stack.yml
version: '3.8'

services:
  # Main Voicebox app
  voicebox:
    image: ghcr.io/jamiepine/voicebox:latest-cuda
    restart: unless-stopped
    volumes:
      - voicebox-data:/app/data
      - huggingface-cache:/root/.cache/huggingface
    environment:
      - TTS_MODE=local
      - WHISPER_MODE=local
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - voicebox

  # Prometheus monitoring (optional)
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

volumes:
  voicebox-data:
  huggingface-cache:
  prometheus-data:
```

## Cloud Deployment

### AWS EC2

1. **Launch GPU Instance** (g4dn.xlarge or p3.2xlarge)
2. **Install Docker + nvidia-docker:**
   ```bash
   # Amazon Linux 2
   sudo yum install -y docker
   sudo systemctl start docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```
3. **Deploy:**
   ```bash
   docker run --gpus all -d -p 80:8000 \
     -v voicebox-data:/app/data \
     --restart unless-stopped \
     ghcr.io/jamiepine/voicebox:latest-cuda
   ```

### DigitalOcean

Use GPU Droplet + Docker:

```bash
# Create droplet via CLI
doctl compute droplet create voicebox \
  --size gpu-h100x1-80gb \
  --image ubuntu-22-04-x64 \
  --region nyc3

# SSH and deploy
ssh root@<droplet-ip>
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
docker run --gpus all -d -p 80:8000 voicebox:cuda
```

### Google Cloud Run (CPU-only)

```bash
# Build and push
docker build -t gcr.io/your-project/voicebox .
docker push gcr.io/your-project/voicebox

# Deploy to Cloud Run
gcloud run deploy voicebox \
  --image gcr.io/your-project/voicebox \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --port 8000
```

### Fly.io

Create `fly.toml`:
```toml
app = "voicebox"

[build]
  image = "ghcr.io/jamiepine/voicebox:latest"

[[services]]
  http_checks = []
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

[mounts]
  source = "voicebox_data"
  destination = "/app/data"
```

Deploy:
```bash
fly launch
fly deploy
```

## Troubleshooting

### GPU Not Detected

**Check NVIDIA Docker:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this fails, reinstall nvidia-docker2.

**Check AMD ROCm:**
```bash
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/dev-ubuntu-22.04:6.0 rocminfo
```

### Permission Errors

Container can't write to volumes:
```bash
# Fix permissions
docker run --user $(id -u):$(id -g) -v $(pwd)/data:/app/data voicebox
```

### Out of Memory

Reduce GPU memory usage:
```bash
docker run -e GPU_MEMORY_FRACTION=0.5 voicebox
```

Or use CPU-only:
```bash
docker run -e DEVICE=cpu voicebox
```

### Model Download Fails

Ensure HuggingFace cache is writable:
```bash
docker run -v huggingface-cache:/root/.cache/huggingface voicebox
```

Or use host cache:
```bash
docker run -v ~/.cache/huggingface:/root/.cache/huggingface voicebox
```

### Port Already in Use

Change host port:
```bash
docker run -p 8080:8000 voicebox  # Use port 8080 instead
```

## Security Best Practices

### 1. Don't Run as Root

Create non-root user in Dockerfile:
```dockerfile
RUN useradd -m -u 1000 voicebox
USER voicebox
```

### 2. Use Secrets for API Keys

Don't put API keys in docker-compose.yml:

```bash
# Use Docker secrets
echo "sk-your-key" | docker secret create openai_key -

docker service create \
  --secret openai_key \
  -e OPENAI_API_KEY_FILE=/run/secrets/openai_key \
  voicebox
```

### 3. Network Isolation

Use internal networks for multi-container setups:

```yaml
services:
  voicebox:
    networks:
      - internal
  nginx:
    networks:
      - internal
      - external
    ports:
      - "80:80"

networks:
  internal:
    internal: true
  external:
```

### 4. Resource Limits

Prevent resource exhaustion:

```yaml
services:
  voicebox:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## Performance Tuning

### GPU Memory Management

```bash
# Use 80% of GPU (default 90%)
docker run -e GPU_MEMORY_FRACTION=0.8 voicebox

# Allow GPU memory growth (prevents OOM)
docker run -e TF_FORCE_GPU_ALLOW_GROWTH=true voicebox
```

### Model Caching

Pre-download models to volume:

```bash
# Download models first
docker run --rm -v huggingface-cache:/root/.cache/huggingface \
  voicebox python -c "
from transformers import WhisperProcessor, WhisperForConditionalGeneration
WhisperProcessor.from_pretrained('openai/whisper-base')
WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')
"

# Then run normally
docker run -v huggingface-cache:/root/.cache/huggingface voicebox
```

### Multi-Worker Setup

Use uvicorn workers for better throughput:

```dockerfile
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## Monitoring

### Health Checks

Built-in health endpoint:
```bash
curl http://localhost:8000/health
```

Docker health check:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Prometheus Metrics

Add metrics exporter:
```python
# backend/main.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Then scrape `/metrics` with Prometheus.

### Logs

View container logs:
```bash
docker logs -f voicebox

# Or with compose
docker compose logs -f voicebox
```

## Updates

Docker images are automatically built and published on each GitHub release. To update:

```bash
# Pull latest
docker pull ghcr.io/jamiepine/voicebox:latest
docker compose up -d

# Or pin to a specific version
docker pull ghcr.io/jamiepine/voicebox:0.1.13
```

For automatic updates, use [Watchtower](https://containrrr.dev/watchtower/).

## Future Enhancements

- Kubernetes Helm charts
- Docker Desktop extension
- Automated vulnerability scanning
- ROCm image variant

## Contributing

Help improve Docker support:
1. Test on different platforms (AMD GPU, ARM64, etc.)
2. Submit Dockerfile optimizations
3. Share deployment configurations
4. Report issues: [GitHub Issues](https://github.com/jamiepine/voicebox/issues)

## Resources

- [Docker Documentation](https://docs.docker.com)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [AMD ROCm Docker](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
