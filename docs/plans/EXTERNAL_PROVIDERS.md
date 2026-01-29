# External Provider Support

**Status:** Planned for v0.2.0
**Discussion:** [Reddit Thread](https://reddit.com/r/LocalLLaMA/...)

## Overview

External provider support allows you to connect Voicebox to remotely-hosted TTS and Whisper services instead of running models locally. This is useful for:

- **Existing GPU Infrastructure**: You already have Qwen3-TTS running on a GPU server
- **AMD GPU Users**: Run models on your AMD hardware, use Voicebox as the UI
- **Cloud Deployments**: Host models on Modal, Replicate, RunPod, etc.
- **Team Sharing**: Multiple users share one GPU server running models
- **Mixed Deployments**: Local Whisper + remote TTS, or vice versa

## Architecture

```
┌─────────────────┐         HTTP/API         ┌──────────────────┐
│   Voicebox UI   │ ───────────────────────> │  Your TTS Server │
│   + Backend     │                           │  (Qwen3-TTS on   │
│                 │ <─────────────────────── │   AMD/NVIDIA GPU)│
│  - Profiles     │      Audio + Metadata     └──────────────────┘
│  - History      │
│  - Audio Edit   │         HTTP/API         ┌──────────────────┐
│  - UI           │ ───────────────────────> │ Whisper Service  │
└─────────────────┘                           │ (OpenAI API or   │
                                              │  self-hosted)    │
                                              └──────────────────┘
```

**What Voicebox Still Handles:**
- Voice profile management
- Generation history
- Audio trimming/editing
- Multi-track story editor
- UI/UX layer

**What External Providers Handle:**
- Model inference (TTS generation, transcription)
- GPU allocation
- Model loading/caching

## Configuration

### Environment Variables

```bash
# TTS Provider
TTS_MODE=remote                              # local | remote
TTS_REMOTE_URL=http://192.168.1.100:8000    # Your TTS server URL
TTS_API_KEY=your-api-key                     # Optional authentication

# Whisper Provider
WHISPER_MODE=openai-api                      # local | openai-api | remote
WHISPER_REMOTE_URL=http://localhost:9000     # For self-hosted Whisper
OPENAI_API_KEY=sk-...                        # For OpenAI Whisper API
```

### Voicebox Config UI (Planned)

Settings page will include:
- Provider selection dropdowns
- URL/API key inputs
- Connection test button
- Latency/status indicators

## Hosting External Services

### Option 1: Simple FastAPI Server (Recommended)

Create a lightweight server to expose your local Qwen3-TTS model:

```python
# tts_server.py
from fastapi import FastAPI, UploadFile, File
from qwen_tts import Qwen3TTSModel
import numpy as np
import base64

app = FastAPI()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda"  # or "cpu" for AMD ROCm: use torch+rocm
)

@app.post("/v1/generate")
async def generate(
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: int = None
):
    """Generate speech from text using voice prompt."""
    audio, sample_rate = model.generate_voice_clone(
        text=text,
        voice_clone_prompt=voice_prompt,
    )

    # Return as base64 for transport
    audio_bytes = audio.tobytes()
    return {
        "audio": base64.b64encode(audio_bytes).decode(),
        "sample_rate": sample_rate,
        "dtype": str(audio.dtype)
    }

@app.post("/v1/create_voice_prompt")
async def create_voice_prompt(
    audio: UploadFile = File(...),
    reference_text: str = ""
):
    """Create voice prompt from reference audio."""
    # Save uploaded audio temporarily
    audio_path = f"/tmp/{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # Create voice prompt
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=audio_path,
        ref_text=reference_text,
    )

    return {"voice_prompt": voice_prompt}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "Qwen3-TTS-12Hz-1.7B-Base",
        "device": str(model.device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run it:**
```bash
# Install dependencies
pip install fastapi uvicorn qwen-tts torch

# For AMD GPUs, use ROCm PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4

# Start server
python tts_server.py
```

### Option 2: vLLM (If Supported)

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

### Option 3: Cloud Platforms

**Modal.com Example:**
```python
import modal

app = modal.App("qwen-tts")
image = modal.Image.debian_slim().pip_install("qwen-tts", "torch")

@app.function(gpu="A10G", image=image)
@modal.web_endpoint(method="POST")
def generate(text: str, voice_prompt: dict):
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    audio, sr = model.generate_voice_clone(text, voice_prompt)
    return {"audio": audio.tolist(), "sample_rate": sr}
```

Deploy: `modal deploy tts_server.py`
Get URL: `https://yourapp--generate.modal.run`

## API Specification

External TTS providers must implement these endpoints:

### `POST /v1/generate`

Generate speech from text.

**Request:**
```json
{
  "text": "Hello, this is a test.",
  "voice_prompt": { /* voice prompt object */ },
  "language": "en",
  "seed": 12345
}
```

**Response:**
```json
{
  "audio": "base64-encoded-audio-bytes",
  "sample_rate": 24000,
  "dtype": "float32"
}
```

### `POST /v1/create_voice_prompt`

Create a voice prompt from reference audio.

**Request:** (multipart/form-data)
- `audio`: Audio file upload
- `reference_text`: Transcript of the audio

**Response:**
```json
{
  "voice_prompt": { /* voice prompt object */ }
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "Qwen3-TTS-12Hz-1.7B-Base",
  "device": "cuda:0"
}
```

## Whisper External Providers

### OpenAI Whisper API

Simply set:
```bash
WHISPER_MODE=openai-api
OPENAI_API_KEY=sk-...
```

Voicebox will use OpenAI's Whisper API automatically.

### Self-Hosted Whisper

Run your own Whisper server:

```python
# whisper_server.py
from fastapi import FastAPI, UploadFile, File
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

app = FastAPI()
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

@app.post("/v1/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = None):
    # Load audio
    audio_path = f"/tmp/{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    audio_data, sr = librosa.load(audio_path, sr=16000)

    # Process
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {"text": transcription}
```

Configure Voicebox:
```bash
WHISPER_MODE=remote
WHISPER_REMOTE_URL=http://localhost:9000
```

## Use Cases

### 1. AMD GPU User with Existing Setup

**Scenario:** You have a Radeon 7900 XTX running Qwen3-TTS on Linux.

**Setup:**
1. Run `tts_server.py` on your AMD box (ROCm PyTorch)
2. Configure Voicebox: `TTS_MODE=remote`, `TTS_REMOTE_URL=http://amd-box:8000`
3. Use Voicebox UI for profiles, generation, editing
4. TTS happens on your AMD GPU

### 2. Team Deployment

**Scenario:** 5 team members, 1 GPU server.

**Setup:**
1. Deploy TTS server on shared GPU box
2. Each person runs Voicebox desktop app locally
3. All point to same `TTS_REMOTE_URL`
4. Profiles and history stay local per user
5. GPU usage is shared

### 3. Hybrid Local/Remote

**Scenario:** Fast local Whisper, heavy TTS on cloud.

**Setup:**
```bash
TTS_MODE=remote
TTS_REMOTE_URL=https://your-modal-app.modal.run

WHISPER_MODE=local  # Fast transcription on your CPU
```

### 4. OpenAI Whisper + Self-Hosted TTS

**Scenario:** Use OpenAI's API for transcription, run TTS locally.

**Setup:**
```bash
TTS_MODE=local

WHISPER_MODE=openai-api
OPENAI_API_KEY=sk-...
```

## Security Considerations

### Authentication

Add API key authentication to your external server:

```python
from fastapi import Header, HTTPException

API_KEY = "your-secret-key"

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/v1/generate", dependencies=[Depends(verify_api_key)])
async def generate(...):
    ...
```

Configure Voicebox:
```bash
TTS_API_KEY=your-secret-key
```

### Network Security

- **VPN/Tailscale**: Use private network for remote servers
- **HTTPS**: Use reverse proxy (nginx/Caddy) with SSL certificates
- **Firewall**: Restrict access to known IPs

### Rate Limiting

Protect your external server:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/v1/generate")
@limiter.limit("10/minute")
async def generate(...):
    ...
```

## Performance Considerations

### Latency

External providers add network latency:
- **Local network**: ~10-50ms overhead (negligible)
- **Same datacenter**: ~1-5ms overhead
- **Cross-region cloud**: 50-200ms+ overhead

For real-time applications, keep TTS server on local network or same cloud region.

### Caching

Implement response caching on external server:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_generation(text, voice_prompt_hash, language, seed):
    return model.generate_voice_clone(text, voice_prompt)
```

### Load Balancing

For high-traffic deployments, run multiple TTS servers behind a load balancer:

```
Voicebox ──> Load Balancer ──> TTS Server 1 (GPU 1)
                           ├──> TTS Server 2 (GPU 2)
                           └──> TTS Server 3 (GPU 3)
```

## Future Enhancements

- [ ] **Provider Marketplace**: Built-in directory of compatible providers
- [ ] **Automatic Fallback**: If remote fails, fallback to local
- [ ] **Cost Tracking**: Monitor API usage and costs
- [ ] **Performance Metrics**: Latency, throughput dashboards
- [ ] **Multi-Provider**: Use different providers for different voices/languages

## Contributing

If you build an external provider, please share:
1. Server implementation
2. Performance benchmarks
3. Deployment guide

Submit to: [GitHub Discussions](https://github.com/jamiepine/voicebox/discussions)

## Questions?

- **Discord**: [Join the community](https://discord.gg/...)
- **GitHub**: [Open an issue](https://github.com/jamiepine/voicebox/issues)
- **Docs**: [Full documentation](https://voicebox.sh/docs)
