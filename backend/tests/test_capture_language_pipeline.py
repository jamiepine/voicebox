from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile

from backend.backends import TranscriptionResult
from backend.mcp_server import tools
from backend.routes import transcription as transcription_route
from backend.services import captures, transcribe
from backend.services.refinement import RefinementFlags
from backend.utils import audio as audio_utils


@pytest.mark.asyncio
async def test_retranscribe_persists_auto_detected_language(monkeypatch, tmp_path):
    audio_path = tmp_path / "capture.wav"
    audio_path.write_bytes(b"audio")
    row = SimpleNamespace(
        id="capture-1",
        audio_path="captures/capture.wav",
        transcript_raw="old",
        transcript_refined="old refined",
        stt_model="base",
        language=None,
        llm_model="0.6B",
        refinement_flags="{}",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = row
    whisper = SimpleNamespace(
        model_size="turbo",
        transcribe_with_metadata=AsyncMock(return_value=TranscriptionResult(text="bonjour le monde", language="fr")),
    )
    monkeypatch.setattr(captures.config, "resolve_storage_path", lambda _path: audio_path)
    monkeypatch.setattr(captures, "get_whisper_model", lambda: whisper)
    monkeypatch.setattr(captures, "_to_response", lambda value: value)

    result = await captures.retranscribe_capture(
        capture_id="capture-1",
        stt_model=None,
        language=None,
        db=db,
    )

    assert result.transcript_raw == "bonjour le monde"
    assert result.language == "fr"
    assert result.transcript_refined is None


@pytest.mark.asyncio
async def test_mcp_transcribe_returns_detected_language(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    whisper = SimpleNamespace(
        model_size="turbo",
        is_loaded=lambda: True,
        transcribe_with_metadata=AsyncMock(return_value=TranscriptionResult(text="hola mundo", language="es")),
    )
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: whisper)
    monkeypatch.setattr(audio_utils, "load_audio", lambda _path: ([0.0] * 16000, 16000))

    result = await tools._transcribe_file(audio_path, language=" ES ", model=None)

    assert result["text"] == "hola mundo"
    assert result["language"] == "es"
    assert whisper.transcribe_with_metadata.await_args.args[1] == "es"


@pytest.mark.asyncio
async def test_http_transcribe_returns_detected_language(monkeypatch):
    whisper = SimpleNamespace(
        model_size="turbo",
        is_loaded=lambda: True,
        transcribe_with_metadata=AsyncMock(return_value=TranscriptionResult(text="hallo welt", language="de")),
    )
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: whisper)
    monkeypatch.setattr(audio_utils, "load_audio", lambda _path: ([0.0] * 16000, 16000))
    upload = UploadFile(filename="sample.wav", file=BytesIO(b"audio"))

    response = await transcription_route.transcribe_audio(
        upload,
        language=" AUTO ",
        model=None,
    )

    assert response.text == "hallo welt"
    assert response.language == "de"
    assert whisper.transcribe_with_metadata.await_args.args[1] is None


@pytest.mark.asyncio
async def test_capture_refinement_receives_persisted_language(monkeypatch):
    row = SimpleNamespace(
        id="capture-1",
        transcript_raw="打开 package.json",
        transcript_refined=None,
        language="zh",
        llm_model=None,
        refinement_flags=None,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = row
    refine = AsyncMock(return_value=("打开 package.json。", "0.6B"))
    monkeypatch.setattr(captures, "refine_transcript", refine)
    monkeypatch.setattr(captures, "_to_response", lambda value: value)

    result = await captures.refine_capture(
        capture_id="capture-1",
        flags=RefinementFlags(),
        model_size="0.6B",
        db=db,
    )

    assert result.transcript_refined == "打开 package.json。"
    assert refine.await_args.kwargs["language"] == "zh"
