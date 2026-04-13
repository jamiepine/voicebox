"""Lightweight TTS endpoint for external clients (e.g. consciousness-fabricator).

Accepts multipart form data with reference audio + text, routes through
the Qwen 0.6B backend, and returns base64-encoded audio.
"""

import base64
import logging
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])


@router.post("/generate")
async def tts_generate(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...),
    language: str = Form(default="en"),
    model_size: str = Form(default="0.6B"),
):
    """Generate speech from text using voice cloning.

    Accepts reference audio + transcript, creates a voice prompt,
    and generates speech using the Qwen 0.6B model.

    Returns base64-encoded WAV audio.
    """
    from ..backends import get_tts_backend_for_engine, load_engine_model

    if model_size != "0.6B":
        raise HTTPException(status_code=400, detail=f"Unsupported model_size: {model_size}")

    tmp_path = None
    backend = None
    try:
        backend = get_tts_backend_for_engine("qwen")

        await load_engine_model("qwen", model_size)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            ref_bytes = await ref_audio.read()
            tmp.write(ref_bytes)
            tmp_path = tmp.name

        voice_prompt, _ = await backend.create_voice_prompt(
            tmp_path,
            ref_text,
            use_cache=False,
        )

        audio_arr, sample_rate = await backend.generate(
            text=text,
            voice_prompt=voice_prompt,
            language=language,
        )

        wav_bytes = _numpy_to_wav(audio_arr, sample_rate)
        duration_ms = int(len(audio_arr) / sample_rate * 1000) if sample_rate > 0 else 0

        return {
            "audio_b64": base64.b64encode(wav_bytes).decode(),
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "model_size": model_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _numpy_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    import io
    import wave
    import struct

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = (audio * 32767).astype(np.int16)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples.tolist()))
    return buf.getvalue()
