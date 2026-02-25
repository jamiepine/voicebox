# voicebox Backend

Production-quality FastAPI backend for voice cloning with Qwen3-TTS and Chatterbox TTS.

## Features

- ✅ **Voice Profile Management** - Create, update, delete voice profiles with multi-sample support
- ✅ **Voice Cloning** - Generate speech using voice profiles with caching
- ✅ **Generation History** - Full history tracking with search and filtering
- ✅ **Hebrew Language Support** - Chatterbox TTS for Hebrew voice cloning with ivrit-ai Whisper for transcription
- ✅ **Transcription** - Whisper-based audio transcription (including Hebrew via ivrit-ai models)
- ✅ **Multi-Sample Profiles** - Combine multiple reference samples for better quality
- ✅ **Voice Prompt Caching** - Dual memory + disk caching for fast generation
- ✅ **Audio Validation** - Automatic validation of reference audio quality
- ✅ **Model Management** - Lazy loading and VRAM management

## Architecture

```
backend/
├── main.py              # FastAPI app with all routes
├── models.py            # Pydantic request/response models
├── platform_detect.py   # Platform detection for backend selection
├── tts.py              # TTS backend abstraction (delegates to MLX or PyTorch)
├── transcribe.py       # STT backend abstraction (delegates to MLX or PyTorch)
├── backends/           # Backend implementations
│   ├── __init__.py     # Backend factory and protocols
│   ├── mlx_backend.py  # MLX backend (Apple Silicon)
│   ├── pytorch_backend.py  # PyTorch backend (Windows/Linux/Intel)
│   └── chatterbox_backend.py  # Chatterbox TTS backend (Hebrew)
├── profiles.py         # Voice profile CRUD
├── history.py          # Generation history
├── studio.py           # Audio editing (TODO)
├── database.py         # SQLite ORM
└── utils/
    ├── audio.py        # Audio processing utilities
    ├── cache.py        # Voice prompt caching
    └── validation.py   # Input validation
```

### Backend Selection

Voicebox automatically selects the best backend based on platform and language:

- **Apple Silicon (M1/M2/M3)**: Uses MLX backend with native Metal acceleration (4-5x faster)
- **Windows/Linux/Intel Mac**: Uses PyTorch backend (CUDA GPU if available, CPU fallback)
- **Hebrew**: Uses Chatterbox Multilingual TTS (ResembleAI/chatterbox) on all platforms

The TTS/STT backend is detected at runtime via `platform_detect.py`. For Hebrew, the Chatterbox backend is used regardless of platform. Both Qwen3-TTS backends (MLX and PyTorch) implement the same interface, so the API remains consistent across platforms.

## API Endpoints

### Health & Info

#### `GET /`
Root endpoint with version info.

#### `GET /health`
Health check with model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_type": "Metal (Apple Silicon via MLX)",
  "backend_type": "mlx",
  "vram_used_mb": null
}
```

**Backend Types:**
- `"mlx"` - MLX backend (Apple Silicon with Metal acceleration)
- `"pytorch"` - PyTorch backend (Windows/Linux/Intel Mac)

### Voice Profiles

**Note:** The database is automatically initialized when the server starts. No manual setup required.

#### `POST /profiles`
Create a new voice profile.

**Request:**
```json
{
  "name": "My Voice",
  "description": "Optional description",
  "language": "en"
}
```

**Supported languages:** `zh`, `en`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`, `he`

**Request (Hebrew example):**
```json
{
  "name": "My Voice",
  "description": "Optional description",
  "language": "he"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "My Voice",
  "description": "Optional description",
  "language": "en",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

#### `GET /profiles`
List all voice profiles.

#### `GET /profiles/{profile_id}`
Get a specific profile.

#### `PUT /profiles/{profile_id}`
Update a profile.

#### `DELETE /profiles/{profile_id}`
Delete a profile and all associated samples.

#### `POST /profiles/{profile_id}/samples`
Add a sample to a profile.

**Form Data:**
- `file`: Audio file (WAV, MP3, etc.)
- `reference_text`: Transcript of the audio

**Response:**
```json
{
  "id": "sample-uuid",
  "profile_id": "profile-uuid",
  "audio_path": "/path/to/sample.wav",
  "reference_text": "This is my voice"
}
```

#### `GET /profiles/{profile_id}/samples`
List all samples for a profile.

#### `DELETE /profiles/samples/{sample_id}`
Delete a specific sample.

### Generation

#### `POST /generate`
Generate speech from text using a voice profile.

**Request:**
```json
{
  "profile_id": "uuid",
  "text": "Hello, this is a test.",
  "language": "en",
  "seed": 42
}
```

**Supported languages:** `zh`, `en`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`, `he`

When `language` is `"he"`, the request is routed to the Chatterbox TTS backend. The Chatterbox model is auto-downloaded on first use. All other languages use Qwen3-TTS.

**Response:**
```json
{
  "id": "generation-uuid",
  "profile_id": "profile-uuid",
  "text": "Hello, this is a test.",
  "language": "en",
  "audio_path": "/path/to/audio.wav",
  "duration": 2.5,
  "seed": 42,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### History

#### `GET /history`
List generation history with optional filters.

**Query Parameters:**
- `profile_id` (optional): Filter by profile
- `search` (optional): Search in text content
- `limit` (default: 50): Results per page
- `offset` (default: 0): Pagination offset

#### `GET /history/{generation_id}`
Get a specific generation.

#### `DELETE /history/{generation_id}`
Delete a generation.

#### `GET /history/stats`
Get generation statistics.

**Response:**
```json
{
  "total_generations": 100,
  "total_duration_seconds": 250.5,
  "generations_by_profile": {
    "profile-uuid-1": 50,
    "profile-uuid-2": 50
  }
}
```

### Audio Files

#### `GET /audio/{generation_id}`
Download generated audio file.

Returns WAV file with appropriate headers.

### Transcription

#### `POST /transcribe`
Transcribe audio file to text.

**Form Data:**
- `file`: Audio file
- `language` (optional): Language hint (`zh`, `en`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`, `he`)

For Hebrew (`he`), the ivrit-ai Whisper model (`ivrit-ai/whisper-large-v3-turbo`) is automatically selected for optimal transcription quality. The model is downloaded on first use.

**Response:**
```json
{
  "text": "Transcribed text here",
  "duration": 5.5
}
```

### Model Management

#### `POST /models/load`
Manually load TTS model.

**Query Parameters:**
- `model_size`: Model size (1.7B or 0.6B)

#### `POST /models/unload`
Unload TTS model to free memory.

## Database Schema

### profiles
- `id`: UUID primary key
- `name`: Profile name (unique)
- `description`: Optional description
- `language`: Language code (zh/en/ja/ko/de/fr/ru/pt/es/it/he)
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### profile_samples
- `id`: UUID primary key
- `profile_id`: Foreign key to profiles
- `audio_path`: Path to audio file
- `reference_text`: Transcript

### generations
- `id`: UUID primary key
- `profile_id`: Foreign key to profiles
- `text`: Generated text
- `language`: Language code (zh/en/ja/ko/de/fr/ru/pt/es/it/he)
- `audio_path`: Path to audio file
- `duration`: Duration in seconds
- `seed`: Random seed (optional)
- `created_at`: Creation timestamp

### projects
- `id`: UUID primary key
- `name`: Project name
- `data`: JSON data
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

## File Structure

```
data/
├── profiles/
│   └── {profile_id}/
│       ├── {sample_id}.wav
│       └── ...
├── generations/
│   └── {generation_id}.wav
├── cache/
│   └── {hash}.prompt
├── projects/
│   └── {project_id}.json
└── voicebox.db
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** On Apple Silicon, also install MLX dependencies for faster inference:
```bash
pip install -r requirements-mlx.txt
```

### 2. Download Models (Automatic)

The Qwen3-TTS models are automatically downloaded from HuggingFace Hub on first use, similar to how Whisper models work.

**No manual download required!** The models will be cached locally after the first download.

Available TTS models:
- **Qwen3-TTS 1.7B** (recommended): `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (~4GB)
- **Qwen3-TTS 0.6B** (faster): `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (~2GB)
- **Chatterbox Multilingual** (Hebrew): `ResembleAI/chatterbox` (~1GB) — auto-downloaded when Hebrew is first used

Available STT models:
- **Whisper** (general): `openai/whisper-base` (default, auto-downloaded)
- **ivrit-ai Whisper v3-turbo** (Hebrew): `ivrit-ai/whisper-large-v3-turbo` — auto-selected for Hebrew transcription
- **ivrit-ai Whisper v3** (Hebrew, higher quality): `ivrit-ai/whisper-large-v3`

**Note:** The first generation/transcription will take longer as the model downloads. Subsequent runs use the cached model.

#### Manual Download (Optional)

If you prefer to download models manually or have limited internet during runtime:

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download 1.7B model
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base

# Or use Python
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"
```

Models are cached in `~/.cache/huggingface/hub/` by default.

### 4. Run Server

```bash
# Development (local only)
python -m backend.main

# Production (allow remote access)
python -m backend.main --host 0.0.0.0 --port 8000
```

## Usage Examples

### Creating a Voice Profile

```bash
# 1. Create profile
curl -X POST http://localhost:8000/profiles \
  -H "Content-Type: application/json" \
  -d '{"name": "My Voice", "language": "en"}'

# Response: {"id": "abc-123", ...}

# 2. Add sample
curl -X POST http://localhost:8000/profiles/abc-123/samples \
  -F "file=@sample.wav" \
  -F "reference_text=This is my voice sample"
```

### Generating Speech

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "abc-123",
    "text": "Hello, this is a test.",
    "language": "en",
    "seed": 42
  }'

# Response: {"id": "gen-456", "audio_path": "/path/to/audio.wav", ...}

# Download audio
curl http://localhost:8000/audio/gen-456 -o output.wav
```

### Transcribing Audio

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav" \
  -F "language=en"

# Response: {"text": "Transcribed text", "duration": 5.5}
```

## Advanced Features

### Multi-Sample Profiles

Add multiple samples to a profile for better quality:

```bash
# Add first sample
curl -X POST http://localhost:8000/profiles/abc-123/samples \
  -F "file=@sample1.wav" \
  -F "reference_text=First sample"

# Add second sample
curl -X POST http://localhost:8000/profiles/abc-123/samples \
  -F "file=@sample2.wav" \
  -F "reference_text=Second sample"

# Generation will automatically combine all samples
```

### Voice Prompt Caching

Voice prompts are automatically cached for faster generation:
- First generation: ~5-10 seconds (creates prompt)
- Subsequent generations: ~1-2 seconds (uses cached prompt)

Cache is stored in `data/cache/` and persists across server restarts.

### VRAM Management

Models are lazy-loaded and can be manually unloaded:

```bash
# Unload TTS model
curl -X POST http://localhost:8000/models/unload

# Load specific model size
curl -X POST "http://localhost:8000/models/load?model_size=0.6B"
```

## Error Handling

All endpoints return proper HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid input
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include details:

```json
{
  "detail": "Profile not found"
}
```

## Performance Tips

1. **Use multi-sample profiles** - Better quality than single sample
2. **Let caching work** - Voice prompts are cached automatically
3. **Use 0.6B model on CPU** - Faster than 1.7B with acceptable quality
4. **Use 1.7B model on GPU** - Best quality, still fast
5. **Unload Whisper after transcription** - Frees VRAM for TTS

## TODO

- [ ] WebSocket support for generation progress
- [ ] Batch generation endpoint
- [ ] Audio effects (M3GAN, etc.)
- [ ] Voice design (text-to-voice)
- [ ] Audio studio timeline features
- [ ] Project management
- [ ] Authentication & rate limiting
- [ ] Export/import profiles

## License

See main project LICENSE.
