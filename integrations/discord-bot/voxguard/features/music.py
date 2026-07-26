"""Music playback.

Streams audio through FFmpeg rather than downloading it: yt-dlp resolves a
direct media URL, FFmpeg pulls it over the network and re-encodes to the
48 kHz stereo PCM Discord wants. Nothing hits disk, so a long queue costs no
storage and there's no cleanup to get wrong.

Two details that matter in practice:

* **Resolution happens off the event loop.** yt-dlp is synchronous and can
  block for seconds on a playlist; running it inline would stall every other
  guild's voice, moderation and heartbeat.
* **URLs are resolved late.** Streaming URLs expire, often within hours, so a
  track queued now is resolved when it reaches the front rather than at
  queue time.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import discord

log = logging.getLogger(__name__)

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    YTDLP_AVAILABLE = False

MISSING_YTDLP = (
    "Music playback needs yt-dlp. Install it with `pip install yt-dlp` and restart."
)

# Reconnect flags matter: streaming URLs drop mid-track surprisingly often,
# and without these FFmpeg exits silently and the track just stops.
FFMPEG_BEFORE = (
    "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 "
    "-nostdin -hide_banner -loglevel error"
)
FFMPEG_OPTS = "-vn"

YDL_OPTS = {
    "format": "bestaudio/best",
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
    "extract_flat": False,
    "skip_download": True,
}

MAX_QUEUE = 200
MAX_DURATION = 3 * 3600  # refuse 3h+ streams; usually a livestream or a mistake


@dataclass
class Track:
    query: str
    title: str
    url: str            # page URL, for display
    duration: int       # seconds, 0 if unknown
    thumbnail: str | None
    uploader: str | None
    requested_by: int
    stream_url: str | None = None

    @property
    def pretty_duration(self) -> str:
        if not self.duration:
            return "live"
        h, rem = divmod(self.duration, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class GuildPlayer:
    guild_id: int
    queue: list[Track] = field(default_factory=list)
    current: Track | None = None
    volume: float = 0.5
    loop_track: bool = False
    loop_queue: bool = False
    text_channel_id: int | None = None
    _skipped: bool = False

    def clear(self) -> None:
        self.queue.clear()
        self.current = None
        self.loop_track = False
        self.loop_queue = False


class MusicError(RuntimeError):
    pass


class MusicPlayer:
    """One playback pipeline per guild."""

    def __init__(self) -> None:
        self._players: dict[int, GuildPlayer] = {}
        self._advancing: set[int] = set()

    def player(self, guild_id: int) -> GuildPlayer:
        player = self._players.get(guild_id)
        if player is None:
            player = GuildPlayer(guild_id=guild_id)
            self._players[guild_id] = player
        return player

    def drop(self, guild_id: int) -> None:
        self._players.pop(guild_id, None)

    # -- resolution ---------------------------------------------------------

    @staticmethod
    def _extract(query: str, *, playlist: bool) -> list[dict]:
        opts = dict(YDL_OPTS)
        opts["noplaylist"] = not playlist
        if playlist:
            # Flat extraction keeps a 100-track playlist from taking minutes;
            # each entry is resolved properly when it reaches the front.
            opts["extract_flat"] = "in_playlist"

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(query, download=False)

        if info is None:
            return []
        if "entries" in info:
            return [e for e in info["entries"] if e]
        return [info]

    async def search(
        self, query: str, requested_by: int, *, playlist: bool = False
    ) -> list[Track]:
        """Resolve a URL or search term into tracks. Runs off the event loop."""
        if not YTDLP_AVAILABLE:
            raise MusicError(MISSING_YTDLP)

        try:
            entries = await asyncio.to_thread(self._extract, query, playlist=playlist)
        except Exception as exc:  # yt-dlp raises a wide variety
            raise MusicError(f"Couldn't resolve that: {_clean_ytdlp_error(exc)}") from exc

        if not entries:
            raise MusicError("Nothing found for that query.")

        tracks: list[Track] = []
        for entry in entries:
            duration = int(entry.get("duration") or 0)
            if duration and duration > MAX_DURATION:
                continue
            tracks.append(
                Track(
                    query=query,
                    title=entry.get("title") or "Unknown title",
                    url=entry.get("webpage_url") or entry.get("url") or query,
                    duration=duration,
                    thumbnail=entry.get("thumbnail"),
                    uploader=entry.get("uploader") or entry.get("channel"),
                    requested_by=requested_by,
                    # Present only on a full (non-flat) extraction.
                    stream_url=entry.get("url") if entry.get("acodec") else None,
                )
            )
        if not tracks:
            raise MusicError("Everything found was too long to queue.")
        return tracks

    async def _resolve_stream(self, track: Track) -> str:
        """Get a fresh direct media URL for a track about to play."""
        if track.stream_url:
            return track.stream_url
        try:
            entries = await asyncio.to_thread(self._extract, track.url, playlist=False)
        except Exception as exc:
            raise MusicError(f"Couldn't resolve stream: {_clean_ytdlp_error(exc)}") from exc
        if not entries or not entries[0].get("url"):
            raise MusicError("No playable audio stream for that track.")
        return entries[0]["url"]

    # -- playback -----------------------------------------------------------

    async def play_next(
        self, voice_client: discord.VoiceClient, bot: discord.Client
    ) -> Track | None:
        """Advance the queue. Returns the track that started, if any."""
        guild_id = voice_client.guild.id
        player = self.player(guild_id)

        # Guard against two callbacks racing into the same advance.
        if guild_id in self._advancing:
            return None
        self._advancing.add(guild_id)
        try:
            previous = player.current

            if player.loop_track and previous and not player._skipped:
                nxt = previous
            else:
                if player.loop_queue and previous and not player._skipped:
                    player.queue.append(previous)
                nxt = player.queue.pop(0) if player.queue else None
            player._skipped = False

            player.current = nxt
            if nxt is None:
                return None

            try:
                stream = await self._resolve_stream(nxt)
            except MusicError as exc:
                log.warning("Skipping %r: %s", nxt.title, exc)
                await self._notify(bot, player, f"Skipped **{nxt.title}** — {exc}")
                return await self.play_next(voice_client, bot)

            source = discord.FFmpegPCMAudio(
                stream, before_options=FFMPEG_BEFORE, options=FFMPEG_OPTS
            )
            source = discord.PCMVolumeTransformer(source, volume=player.volume)

            loop = asyncio.get_running_loop()

            def after(error: Exception | None) -> None:
                if error:
                    log.warning("Playback error in guild %s: %s", guild_id, error)
                # play() calls this from a worker thread.
                asyncio.run_coroutine_threadsafe(
                    self._on_finished(voice_client, bot), loop
                )

            if voice_client.is_playing() or voice_client.is_paused():
                voice_client.stop()
            voice_client.play(source, after=after)
            return nxt
        finally:
            self._advancing.discard(guild_id)

    async def _on_finished(
        self, voice_client: discord.VoiceClient, bot: discord.Client
    ) -> None:
        if not voice_client.is_connected():
            return
        track = await self.play_next(voice_client, bot)
        if track is not None:
            player = self.player(voice_client.guild.id)
            await self._notify(bot, player, f"Now playing **{track.title}**")

    async def _notify(self, bot: discord.Client, player: GuildPlayer, text: str) -> None:
        if not player.text_channel_id:
            return
        channel = bot.get_channel(player.text_channel_id)
        if isinstance(channel, discord.abc.Messageable):
            try:
                await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
            except discord.HTTPException:
                pass

    def skip(self, voice_client: discord.VoiceClient) -> bool:
        player = self.player(voice_client.guild.id)
        if not (voice_client.is_playing() or voice_client.is_paused()):
            return False
        # Tell the advance logic this was deliberate, so loop modes don't
        # immediately replay the track the user just skipped.
        player._skipped = True
        voice_client.stop()
        return True

    def stop(self, voice_client: discord.VoiceClient) -> None:
        self.player(voice_client.guild.id).clear()
        if voice_client.is_playing() or voice_client.is_paused():
            voice_client.stop()

    def set_volume(self, voice_client: discord.VoiceClient, volume: float) -> None:
        player = self.player(voice_client.guild.id)
        player.volume = max(0.0, min(2.0, volume))
        if isinstance(voice_client.source, discord.PCMVolumeTransformer):
            voice_client.source.volume = player.volume


# yt-dlp failures are long, ANSI-coloured, and often expose local network
# detail (proxy hosts, file paths) that has no business in a Discord message.
# These map the common causes onto something a user can act on.
_ERROR_HINTS = (
    ("unable to connect to proxy", "Couldn't reach the video site from this host."),
    ("unable to download api page", "Couldn't reach the video site — it may be rate-limiting."),
    ("sign in to confirm your age", "That video is age-restricted."),
    ("private video", "That video is private."),
    ("video unavailable", "That video is unavailable."),
    ("is not available in your country", "That video is blocked in this region."),
    ("members-only", "That video is members-only."),
    ("no video formats", "No playable audio stream for that link."),
    ("unsupported url", "That link isn't supported."),
    ("http error 429", "The video site is rate-limiting this host. Try again shortly."),
    ("is not a valid url", "That doesn't look like a valid link."),
)


def _clean_ytdlp_error(exc: Exception) -> str:
    """Turn a yt-dlp failure into one actionable sentence."""
    raw = str(exc)
    lowered = raw.lower()
    for needle, friendly in _ERROR_HINTS:
        if needle in lowered:
            return friendly

    # Unrecognised: strip ANSI/prefixes and keep it short rather than dumping
    # a paragraph of yt-dlp internals into chat.
    text = raw.replace("\x1b[0;31m", "").replace("\x1b[0m", "").strip()
    text = text.removeprefix("ERROR: ").split("\n")[0]
    text = text.split(";")[0].split(" (caused by")[0]
    return text[:160] or type(exc).__name__
