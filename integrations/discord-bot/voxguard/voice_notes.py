"""Discord voice-message (voice note) moderation.

Discord's native voice messages arrive as a regular `Message` with
`flags.voice_message` set and a single audio attachment (an Ogg/Opus blob).
There's no live stream to buffer here — the whole clip is already recorded —
so this is a straight download-transcribe-match-enforce pipeline, reusing the
same Matcher and Enforcer as live voice.
"""

from __future__ import annotations

import logging

import aiohttp
import discord

from .matching import Matcher
from .moderation import Enforcer
from .voicebox_client import VoiceboxClient, VoiceboxError

log = logging.getLogger(__name__)

MAX_VOICE_NOTE_BYTES = 15 * 1024 * 1024


def is_voice_message(message: discord.Message) -> discord.Attachment | None:
    if not message.flags.voice_message or not message.attachments:
        return None
    return message.attachments[0]


class VoiceNoteModerator:
    def __init__(self, voicebox: VoiceboxClient, enforcer: Enforcer) -> None:
        self.voicebox = voicebox
        self.enforcer = enforcer

    async def handle(self, message: discord.Message, config: dict, matcher: Matcher) -> None:
        cfg = config.get("voice_notes", {})
        if not cfg.get("enabled", False):
            return
        if not isinstance(message.author, discord.Member):
            return

        attachment = is_voice_message(message)
        if attachment is None or attachment.size > MAX_VOICE_NOTE_BYTES:
            return

        try:
            data = await attachment.read()
        except (discord.HTTPException, aiohttp.ClientError) as exc:
            log.warning("Could not download voice note: %s", exc)
            return

        try:
            text = await self.voicebox.transcribe(data, filename=attachment.filename or "note.ogg")
        except VoiceboxError as exc:
            log.warning("Voice note transcription failed: %s", exc)
            return

        if not text.strip():
            return

        matches = matcher.scan(text, min_confidence=config.get("voice", {}).get("min_confidence", 0.55))
        if not matches:
            return

        log.info(
            "[voice-note] %s in #%s: %r -> matched %s",
            message.author, getattr(message.channel, "name", message.channel.id),
            text, [m.term for m in matches],
        )

        source = f"voice note in {getattr(message.channel, 'mention', '#unknown')}"
        await self.enforcer.enforce(
            message.author,
            scope="voice_notes",
            config=config,
            matches=matches,
            transcript=text,
            source=source,
            message=message,
        )
