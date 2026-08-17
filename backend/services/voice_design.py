"""MiniMax voice design — turn a written description into a reusable voice.

The provider exposes one JSON endpoint that takes a natural-language voice
description and returns a voice id usable for later speech synthesis:

    POST /v1/voice_design
    {"prompt": "...", "voice_id": "...", "preview_text": "..."}
    -> {"voice_id": "...", "trial_audio": "...", "base_resp": {"status_code": 0}}

Two regional hosts serve the same API; select one with ``MINIMAX_API_REGION``.
The ``voice_id`` is supplied by us rather than discovered afterwards, so the
profile row and the remote voice stay in sync without a second lookup — see
``services.profiles.design_profile_voice``, which persists the returned id.

Failures arrive inside the body: the HTTP status stays 200 and ``base_resp``
carries a non-zero ``status_code`` plus a human-readable ``status_msg``.

Requirements:
  - ``MINIMAX_API_KEY`` environment variable
  - httpx (already in requirements.txt)

Reference: https://platform.minimax.io/docs/api-reference/voice-design-design
"""

import logging
import os
import re
import secrets

import httpx

logger = logging.getLogger(__name__)

# Regional hosts for the voice-design endpoint. Both speak the same API.
MINIMAX_VOICE_DESIGN_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/voice_design",
    "cn_zh": "https://api.minimaxi.com/v1/voice_design",
}

MINIMAX_DEFAULT_REGION = "global_en"

API_KEY_ENV_VAR = "MINIMAX_API_KEY"
API_REGION_ENV_VAR = "MINIMAX_API_REGION"

# Mirrors VoiceProfileCreate.design_prompt so the HTTP call and the DB agree.
DESIGN_PROMPT_MAX_CHARS = 2000

# The provider renders a short preview of every new voice and caps its length.
PREVIEW_TEXT_MAX_CHARS = 500
DEFAULT_PREVIEW_TEXT = "This is a preview of the voice you just designed."

# Voice ids we mint carry a prefix so designed voices stay recognisable in the
# provider's account-wide voice list.
VOICE_ID_PREFIX = "voicebox_design_"
VOICE_ID_MIN_CHARS = 8
VOICE_ID_MAX_CHARS = 100
_VOICE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

REQUEST_TIMEOUT_SECONDS = 60.0


def resolve_region(region: str | None = None) -> str:
    """Resolve the API region, honouring ``MINIMAX_API_REGION``.

    Raises:
        ValueError: if the region is not one this endpoint is served from.
    """
    candidate = (region or os.environ.get(API_REGION_ENV_VAR) or MINIMAX_DEFAULT_REGION).strip()
    if candidate not in MINIMAX_VOICE_DESIGN_ENDPOINTS:
        supported = ", ".join(sorted(MINIMAX_VOICE_DESIGN_ENDPOINTS))
        raise ValueError(f"Unknown MiniMax API region '{candidate}'. Supported regions: {supported}")
    return candidate


def get_voice_design_endpoint(region: str | None = None) -> str:
    """Return the voice-design endpoint URL for the resolved region."""
    return MINIMAX_VOICE_DESIGN_ENDPOINTS[resolve_region(region)]


def load_api_key() -> str | None:
    """Load ``MINIMAX_API_KEY`` from the environment."""
    return os.environ.get(API_KEY_ENV_VAR, "").strip() or None


def generate_voice_id() -> str:
    """Mint a fresh voice id for a designed voice."""
    return f"{VOICE_ID_PREFIX}{secrets.token_hex(8)}"


def validate_voice_id(voice_id: str | None) -> str:
    """Return ``voice_id`` unchanged, or raise ``ValueError`` if unusable."""
    candidate = (voice_id or "").strip()
    if not candidate:
        raise ValueError("voice_id is required to design a voice")
    if len(candidate) < VOICE_ID_MIN_CHARS:
        raise ValueError(f"voice_id must be at least {VOICE_ID_MIN_CHARS} characters")
    if len(candidate) > VOICE_ID_MAX_CHARS:
        raise ValueError(f"voice_id must be at most {VOICE_ID_MAX_CHARS} characters")
    if not _VOICE_ID_RE.match(candidate):
        raise ValueError("voice_id must start with a letter and use only letters, digits, hyphens or underscores")
    return candidate


def build_request_payload(
    design_prompt: str | None,
    voice_id: str,
    preview_text: str | None = None,
) -> dict:
    """Validate the design inputs and build the JSON request body."""
    prompt = (design_prompt or "").strip()
    if not prompt:
        raise ValueError("design_prompt is required to design a voice")
    if len(prompt) > DESIGN_PROMPT_MAX_CHARS:
        raise ValueError(f"design_prompt must be at most {DESIGN_PROMPT_MAX_CHARS} characters")

    preview = (preview_text or DEFAULT_PREVIEW_TEXT).strip()
    if not preview:
        raise ValueError("preview_text cannot be blank")
    if len(preview) > PREVIEW_TEXT_MAX_CHARS:
        raise ValueError(f"preview_text must be at most {PREVIEW_TEXT_MAX_CHARS} characters")

    return {
        "prompt": prompt,
        "voice_id": validate_voice_id(voice_id),
        "preview_text": preview,
    }


def extract_voice_id(body: dict, requested_voice_id: str) -> str:
    """Pull the designed voice id out of a response body.

    Raises:
        RuntimeError: if the body reports a provider-side failure or omits the id.
    """
    base_resp = body.get("base_resp") or {}
    status_code = base_resp.get("status_code") or 0
    if status_code:
        status_msg = base_resp.get("status_msg") or "Unknown error"
        raise RuntimeError(f"Voice design failed (status {status_code}): {status_msg}")

    voice_id = (body.get("voice_id") or "").strip()
    if not voice_id:
        raise RuntimeError("Voice design response did not contain a voice_id")
    if voice_id != requested_voice_id:
        logger.warning("Voice design returned voice_id %s instead of the requested %s", voice_id, requested_voice_id)
    return voice_id


async def post_voice_design(url: str, api_key: str, payload: dict) -> dict:
    """POST a design request and return the decoded JSON body.

    A module-level seam so tests can stand in for the network call.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Could not reach the voice design endpoint: {e}") from e

    if response.status_code != 200:
        raise RuntimeError(f"Voice design request failed with HTTP {response.status_code}")

    try:
        body = response.json()
    except ValueError as e:
        raise RuntimeError("Voice design response was not valid JSON") from e
    if not isinstance(body, dict):
        raise RuntimeError("Voice design response was not a JSON object")
    return body


async def design_voice(
    design_prompt: str | None,
    voice_id: str | None = None,
    preview_text: str | None = None,
    region: str | None = None,
) -> str:
    """Design a voice from a written description and return its voice id.

    Args:
        design_prompt: Natural-language description of how the voice should sound.
        voice_id: Voice id to claim; a fresh one is minted when omitted.
        preview_text: Text spoken in the provider's preview sample.
        region: API region override, otherwise ``MINIMAX_API_REGION``.

    Returns:
        The voice id to persist and reference from later synthesis.
    """
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV_VAR} is not configured. Set it in your environment.")

    resolved_region = resolve_region(region)
    requested_voice_id = voice_id or generate_voice_id()
    payload = build_request_payload(design_prompt, requested_voice_id, preview_text)

    body = await post_voice_design(get_voice_design_endpoint(resolved_region), api_key, payload)
    designed_voice_id = extract_voice_id(body, requested_voice_id)
    logger.info("Designed voice %s in region %s", designed_voice_id, resolved_region)
    return designed_voice_id
