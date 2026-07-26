"""Voice cloning consent gate.

Voicebox's own RESPONSIBLE_USE.md is explicit: it cannot verify who owns a
voice sample, and cloning someone without permission is a listed prohibited
use. `/voiceclone` is the one command in this bot that touches another
person's likeness, so it's the one place a typed attestation is mandatory —
recorded with the uploader's ID and timestamp, not just accepted and
forgotten. This doesn't make consent verifiable; it makes the person who
claimed it accountable, and gives a server owner an audit trail if a clone
turns out to be non-consensual.
"""

from __future__ import annotations

from .store import Store
from .voicebox_client import Profile, VoiceboxClient

CONSENT_PHRASE = "I OWN THIS VOICE OR HAVE PERMISSION TO CLONE IT"

MAX_SAMPLE_BYTES = 25 * 1024 * 1024
ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}


class ConsentError(ValueError):
    pass


async def clone_voice(
    voicebox: VoiceboxClient,
    store: Store,
    *,
    guild_id: int,
    uploader_id: int,
    name: str,
    audio: bytes,
    filename: str,
    reference_text: str | None,
    attestation: str,
    language: str = "en",
) -> Profile:
    """Create a profile, attach the sample, and record consent. Raises ConsentError
    if the attestation doesn't match, without touching Voicebox at all."""
    if attestation.strip().upper() != CONSENT_PHRASE:
        raise ConsentError(
            f"You must type the consent phrase exactly: `{CONSENT_PHRASE}`"
        )
    if len(audio) > MAX_SAMPLE_BYTES:
        raise ConsentError(f"File too large (max {MAX_SAMPLE_BYTES // (1024 * 1024)} MB).")

    if not reference_text or not reference_text.strip():
        # No transcript supplied — Voicebox's sample endpoint requires one to
        # align the reference audio, so generate it with the same Whisper
        # pass used for moderation.
        reference_text = await voicebox.transcribe(audio, filename=filename, language=language)
        if not reference_text.strip():
            raise ConsentError(
                "Couldn't auto-transcribe that sample — please pass `reference_text` "
                "with what's said in the clip."
            )

    profile = await voicebox.create_profile(name, language=language)
    await voicebox.add_sample(
        profile.id, audio, filename=filename, reference_text=reference_text
    )
    store.record_consent(guild_id, profile.id, profile.name, uploader_id, attestation, filename)
    return profile
