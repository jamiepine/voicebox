"""Canonical language handling for Voicebox captures."""

from typing import Final

# Canonical OpenAI Whisper language codes. The capture UI intentionally offers
# a smaller curated subset, but API validation must not break existing captures
# or persisted settings that use the rest of Whisper's supported languages.
CAPTURE_LANGUAGE_CODES: Final[tuple[str, ...]] = (
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "yue",
    "zh",
)
_CAPTURE_LANGUAGE_SET = frozenset(CAPTURE_LANGUAGE_CODES)


def normalize_capture_language(language: str | None) -> str | None:
    """Normalize a capture language, treating ``auto`` as auto-detection.

    Only languages exposed by the capture UI are accepted. This keeps raw API
    input out of Whisper decoder hints and refinement instructions.
    """
    if language is None:
        return None

    normalized = language.strip().lower()
    if normalized == "auto":
        return None
    if normalized not in _CAPTURE_LANGUAGE_SET:
        supported = ", ".join(("auto", *CAPTURE_LANGUAGE_CODES))
        raise ValueError(f"Unsupported capture language '{language}'. Expected one of: {supported}")
    return normalized
