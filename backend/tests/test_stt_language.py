"""STT request models accept the same language codes as the TTS ones.

``TranscriptionRequest`` and ``CaptureRetranscribeRequest`` only allowed the
original ten Qwen languages, so a Polish (or any other newer) code got a 422
even though Whisper transcribes it fine (#1095).
"""

import pytest
from pydantic import ValidationError

from backend import models

TTS_LANGUAGES = models.VoiceProfileCreate.model_fields["language"].metadata[0].pattern


@pytest.mark.parametrize("model", [models.TranscriptionRequest, models.CaptureRetranscribeRequest])
def test_stt_language_matches_tts_languages(model):
    assert model.model_fields["language"].metadata[0].pattern == TTS_LANGUAGES
    assert model(language="pl").language == "pl"
    assert model().language is None
    with pytest.raises(ValidationError):
        model(language="xx")
