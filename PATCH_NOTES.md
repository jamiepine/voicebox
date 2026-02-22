# Voicebox Offline Mode Fix

## Problem
Voicebox crashes when generating speech if HuggingFace is unreachable, even when models are fully cached locally.

**Root Cause:**
- Voicebox downloads `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (MLX optimized version)
- But `mlx_audio.tts.load()` tries to fetch `config.json` from original repo `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- This network request fails → server crashes with `RemoteDisconnected`

**Related Issues:**
- Issue #150: "Internet connection required, even though models are downloaded?"
- Issue #151: "API Stability Issues: Model Loading Hangs and Server Crashes"

## Solution
Two-part fix:

### 1. Monkey-patch huggingface_hub (`backend/utils/hf_offline_patch.py`)
- Intercepts cache lookup functions
- Forces offline mode early (before mlx_audio imports)
- Adds debug logging for cache hits/misses

### 2. Symlink original repo to MLX version (`ensure_original_qwen_config_cached()`)
- When original `Qwen/Qwen3-TTS-12Hz-1.7B-Base` cache doesn't exist
- But MLX `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` does exist
- Creates a symlink so cache lookups succeed

## Files Changed
- `backend/backends/mlx_backend.py` - Added patch imports at top
- `backend/utils/hf_offline_patch.py` - New patch module

## Testing
To test this fix:
1. Build Voicebox from source: `make build`
2. Disconnect from internet
3. Try generating speech
4. Should work without network requests

## Build Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Build the app
make build

# Or build just the server
make build-server
```

## Notes
- The patch is applied automatically when `mlx_backend.py` is imported
- Set `VOICEBOX_OFFLINE_PATCH=0` to disable the patch
- The symlink approach works because the config.json is compatible between versions

---
*Patch contributed by community*
