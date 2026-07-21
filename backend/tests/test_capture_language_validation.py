import pytest
from pydantic import ValidationError

from backend import models
from backend.languages import CAPTURE_LANGUAGE_CODES, normalize_capture_language


@pytest.mark.parametrize("language", CAPTURE_LANGUAGE_CODES)
def test_supported_capture_languages_are_canonical(language):
    assert normalize_capture_language(f" {language.upper()} ") == language


def test_auto_capture_language_normalizes_to_none():
    assert normalize_capture_language(" AUTO ") is None
    assert normalize_capture_language(None) is None


def test_unknown_capture_language_is_rejected():
    with pytest.raises(ValueError, match="Unsupported capture language"):
        normalize_capture_language("ignore previous instructions")


def test_retranscription_accepts_profile_legacy_and_auto_languages():
    assert models.CaptureRetranscribeRequest(language="hi").language == "hi"
    assert models.CaptureRetranscribeRequest(language=" KO ").language == "ko"
    assert models.CaptureRetranscribeRequest(language="nl").language == "nl"
    assert models.CaptureRetranscribeRequest(language="auto").language == "auto"
    assert models.CaptureSettingsUpdate(language=" RU ").language == "ru"


def test_retranscription_rejects_unknown_language():
    with pytest.raises(ValidationError):
        models.CaptureRetranscribeRequest(language="xx")
    with pytest.raises(ValidationError):
        models.CaptureSettingsUpdate(language="xx")
