# Voicebox Project Notes

## CLI — voicebox-cli vs cli.py

**`voicebox/voicebox-cli`** is the real CLI. It is stdlib-only (no pip deps), self-contained, and is what users actually run. It has all commands: `server`, `voices`, `import`, `generate`/`say`, `health`, `config`, `transcribe`, `create-voice`. Config persists to `~/.config/voicebox/config.json`.

**`voicebox/backend/cli.py`** is dead code. It predates `voicebox-cli` and was superseded. Its only live reference is the launcher line in `setup-linux.sh` which is intentionally left as-is. **Do not modify cli.py.**

When the user asks for CLI changes, always work on `voicebox-cli`.

## Key Architecture

- **Backend**: FastAPI (`backend/main.py`) served by uvicorn on port 17493
- **Entry points**: `server.py` (PyInstaller binary), `backend/main.py __main__` (dev)
- **Dev script**: `scripts/dev-backend-watch.sh` — loads `.env` from `voicebox/` and `../` then runs uvicorn with `--reload`
- **MLX backend**: `backend/backends/mlx_backend.py` — Apple Silicon only, uses mlx-audio. Models: `mlx-community/Qwen3-TTS-12Hz-{1.7B,0.6B}-Base-4bit`. Uses `Base` variants (not `CustomVoice` — those require a named speaker, not ref_audio).
- **PyTorch backend**: `backend/backends/pytorch_backend.py` — CUDA/CPU, uses qwen-tts
- **Logging**: stdlib `logging`. Set `LOG_LEVEL=DEBUG` env var for verbose output.

## MLX Gotchas

- `transformers` verbosity is suppressed at module-level import in `mlx_backend.py` — do not restore or move this
- Concurrent MLX loads crash Metal (`commit an already committed command buffer`) — serialized via `_MLX_LOAD_LOCK` threading lock in `load_model_async`
- `CustomVoice` model variants require a named speaker arg; `Base` variants support arbitrary voice cloning via `ref_audio`/`ref_text`
- On 16GB unified memory, bf16 models cause swap pressure — use 4-bit quantized variants
