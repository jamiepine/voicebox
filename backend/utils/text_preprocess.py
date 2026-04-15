"""LLM-based text preprocessing for humanizing TTS input."""

import logging
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)

DISFLUENCY_PROMPTS = {
    "es": {
        "system": (
            "Eres un preprocesador de texto para síntesis de voz. "
            "Agrega disfluencias naturales del habla para que el texto suene más humano al ser leído en voz alta. "
            "Inserta hesitaciones (eh, mm, am), muletillas (bueno, pues, o sea, digo), "
            "micro-pausas (representadas como ...), y auto-correcciones. "
            "Mantén el significado idéntico. NO agregues tags entre corchetes. "
            "Devuelve SOLO el texto modificado, nada más."
        ),
        "intensity_hint": {
            "light": "Agrega muy pocas disfluencias, solo 1-2 en todo el texto.",
            "medium": "Agrega disfluencias moderadas, como una conversación casual.",
            "heavy": "Agrega muchas disfluencias, como alguien pensando en voz alta.",
        },
    },
    "en": {
        "system": (
            "You are a text preprocessor for text-to-speech. "
            "Add natural speech disfluencies to make the text sound more human when spoken aloud. "
            "Insert hesitations (um, uh, hmm), filler words (like, you know, well, I mean), "
            "micro-pauses (represented as ...), and self-corrections. "
            "Keep the meaning identical. Do NOT add bracketed tags. "
            "Return ONLY the modified text, nothing else."
        ),
        "intensity_hint": {
            "light": "Add very few disfluencies, only 1-2 in the entire text.",
            "medium": "Add moderate disfluencies, like casual conversation.",
            "heavy": "Add many disfluencies, like someone thinking out loud.",
        },
    },
}

# Default prompts for languages not explicitly defined
DEFAULT_LANG = "en"


async def check_ollama_available(ollama_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


async def inject_disfluencies(
    text: str,
    language: str = "es",
    intensity: Literal["light", "medium", "heavy"] = "light",
    ollama_model: str = "llama3.2:1b",
    ollama_url: str = "http://localhost:11434",
) -> str:
    """
    Inject natural disfluencies into text using a local LLM via Ollama.

    Returns the original text unchanged if Ollama is not available.
    """
    if not await check_ollama_available(ollama_url):
        logger.warning("Ollama not available at %s — skipping text preprocessing", ollama_url)
        return text

    # Get language-specific prompts
    lang_key = language if language in DISFLUENCY_PROMPTS else DEFAULT_LANG
    prompts = DISFLUENCY_PROMPTS[lang_key]
    system_prompt = prompts["system"]
    intensity_hint = prompts["intensity_hint"].get(intensity, prompts["intensity_hint"]["light"])

    full_prompt = f"{system_prompt}\n\nIntensity: {intensity_hint}\n\nText to process:\n{text}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": len(text) * 3,  # Allow up to 3x expansion
                    },
                },
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()

        # Validate result is reasonable
        if not result:
            logger.warning("Empty response from Ollama — using original text")
            return text

        # If result is way too long or too short, it's probably garbage
        if len(result) > len(text) * 4 or len(result) < len(text) * 0.3:
            logger.warning(
                "Ollama result length suspicious (original=%d, result=%d) — using original text",
                len(text),
                len(result),
            )
            return text

        logger.info("Text preprocessed: %d -> %d chars (intensity=%s)", len(text), len(result), intensity)
        return result

    except Exception as e:
        logger.warning("Ollama preprocessing failed: %s — using original text", e)
        return text
