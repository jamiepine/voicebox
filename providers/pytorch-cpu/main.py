"""
Standalone TTS provider server for PyTorch CPU.
"""

import argparse
import asyncio
import base64
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.backends.pytorch_backend import PyTorchTTSBackend


app = FastAPI(title="Voicebox TTS Provider - PyTorch CPU")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global backend instance
_backend: Optional[PyTorchTTSBackend] = None


def get_backend() -> PyTorchTTSBackend:
    """Get or create backend instance."""
    global _backend
    if _backend is None:
        _backend = PyTorchTTSBackend()
    return _backend


@app.get("/tts/health")
async def health():
    """Health check endpoint."""
    backend = get_backend()
    backend_type = "pytorch-cpu"
    
    model_size = None
    if backend.is_loaded():
        if hasattr(backend, '_current_model_size') and backend._current_model_size:
            model_size = backend._current_model_size
    
    device = backend.device if hasattr(backend, 'device') else "cpu"
    
    return {
        "status": "healthy",
        "provider": backend_type,
        "version": "1.0.0",  # TODO: Get from version file
        "model": model_size,
        "device": device,
    }


@app.get("/tts/status")
async def status():
    """Model status endpoint."""
    backend = get_backend()
    
    model_size = None
    if backend.is_loaded():
        if hasattr(backend, '_current_model_size') and backend._current_model_size:
            model_size = backend._current_model_size
    
    available_sizes = ["1.7B", "0.6B"]
    
    gpu_available = False
    vram_used_mb = None
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            vram_used_mb = int(torch.cuda.memory_allocated() / 1024 / 1024)
    except ImportError:
        pass
    
    return {
        "model_loaded": backend.is_loaded(),
        "model_size": model_size,
        "available_sizes": available_sizes,
        "gpu_available": gpu_available,
        "vram_used_mb": vram_used_mb,
    }


@app.post("/tts/generate")
async def generate(
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: Optional[int] = None,
    model_size: str = "1.7B",
):
    """
    Generate speech from text.
    
    Request body (JSON):
    {
        "text": "Hello world!",
        "voice_prompt": {...},
        "language": "en",
        "seed": 12345,
        "model_size": "1.7B"
    }
    """
    backend = get_backend()
    
    # Load model if not loaded or different size
    if not backend.is_loaded() or (
        hasattr(backend, '_current_model_size') and 
        backend._current_model_size != model_size
    ):
        await backend.load_model_async(model_size)
    
    # Generate audio
    audio, sample_rate = await backend.generate(
        text=text,
        voice_prompt=voice_prompt,
        language=language,
        seed=seed,
        instruct=None,  # TODO: Add instruct support
    )
    
    # Convert to base64
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    audio_bytes = buffer.read()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Calculate duration
    duration = len(audio) / sample_rate
    
    return {
        "audio": audio_b64,
        "sample_rate": sample_rate,
        "duration": duration,
    }


@app.post("/tts/create_voice_prompt")
async def create_voice_prompt(
    audio: UploadFile = File(...),
    reference_text: str = Form(...),
    use_cache: bool = Form(True),
):
    """
    Create voice prompt from reference audio.
    
    Request (multipart/form-data):
    - audio: Audio file
    - reference_text: Transcript
    - use_cache: Whether to use cached prompts (default: true)
    """
    backend = get_backend()
    
    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        content = await audio.read()
        tmp_file.write(content)
    
    try:
        # Create voice prompt
        voice_prompt, was_cached = await backend.create_voice_prompt(
            audio_path=tmp_path,
            reference_text=reference_text,
            use_cache=use_cache,
        )
        
        return {
            "voice_prompt": voice_prompt,
            "was_cached": was_cached,
        }
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Voicebox TTS Provider - PyTorch CPU")
    parser.add_argument(
        "--port",
        type=int,
        default=0,  # 0 means random port
        help="Port to bind to",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for models and cache",
    )
    args = parser.parse_args()
    
    # Set data directory if provided
    if args.data_dir:
        from backend import config
        config.set_data_dir(args.data_dir)
    
    # Determine port
    port = args.port
    if port == 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
    
    print(f"Starting TTS Provider (PyTorch CPU) on port {port}")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
