"""`/play` and the rest of the music controls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..features.music import MAX_QUEUE, MusicError

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

log = logging.getLogger(__name__)


async def ensure_voice(interaction: discord.Interaction) -> discord.VoiceClient | None:
    """Connect to the caller's voice channel, or return the existing client."""
    user_voice = getattr(interaction.user, "voice", None)
    if user_voice is None or user_voice.channel is None:
        await interaction.followup.send("Join a voice channel first.")
        return None

    existing = interaction.guild.voice_client
    if existing is not None and existing.is_connected():
        if existing.channel.id != user_voice.channel.id and not (
            existing.is_playing() or existing.is_paused()
        ):
            await existing.move_to(user_voice.channel)
        return existing

    try:
        # Reuse the receive-capable client so moderation and music can share
        # one connection rather than fighting over the voice websocket.
        from discord.ext import voice_recv

        return await user_voice.channel.connect(cls=voice_recv.VoiceRecvClient, timeout=30.0)
    except Exception as exc:
        await interaction.followup.send(f"Couldn't join your voice channel: {exc}")
        return None


class MusicCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    @property
    def music(self):
        return self.bot.runtime.music

    @app_commands.command(name="play", description="Play a YouTube link, playlist, or search term.")
    @app_commands.describe(
        query="A YouTube/SoundCloud URL, or words to search for",
        playlist="Queue the whole playlist rather than just the first track",
    )
    async def play(
        self, interaction: discord.Interaction, query: str, playlist: bool = False
    ) -> None:
        await interaction.response.defer(thinking=True)

        voice_client = await ensure_voice(interaction)
        if voice_client is None:
            return

        try:
            tracks = await self.music.search(query, interaction.user.id, playlist=playlist)
        except MusicError as exc:
            await interaction.followup.send(str(exc))
            return

        player = self.music.player(interaction.guild.id)
        player.text_channel_id = interaction.channel.id

        room = MAX_QUEUE - len(player.queue)
        if room <= 0:
            await interaction.followup.send(f"The queue is full ({MAX_QUEUE} tracks).")
            return
        queued = tracks[:room]
        player.queue.extend(queued)

        was_idle = not (voice_client.is_playing() or voice_client.is_paused())
        if was_idle:
            try:
                started = await self.music.play_next(voice_client, self.bot)
            except MusicError as exc:
                await interaction.followup.send(str(exc))
                return
            if started is not None:
                await interaction.followup.send(embed=self._now_playing_embed(started, player))
                return

        if len(queued) == 1:
            track = queued[0]
            embed = discord.Embed(
                title="Added to queue",
                description=f"[{track.title}]({track.url})",
                colour=0x2B2D31,
            )
            embed.add_field(name="Duration", value=track.pretty_duration, inline=True)
            embed.add_field(name="Position", value=f"#{len(player.queue)}", inline=True)
            if track.thumbnail:
                embed.set_thumbnail(url=track.thumbnail)
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send(
                f"Queued **{len(queued)}** tracks. {len(player.queue)} in the queue."
            )

    @staticmethod
    def _now_playing_embed(track, player) -> discord.Embed:  # noqa: ANN001
        embed = discord.Embed(
            title="Now playing",
            description=f"[{track.title}]({track.url})",
            colour=0x2B2D31,
        )
        embed.add_field(name="Duration", value=track.pretty_duration, inline=True)
        embed.add_field(name="Requested by", value=f"<@{track.requested_by}>", inline=True)
        if track.uploader:
            embed.add_field(name="Uploader", value=track.uploader, inline=True)
        if player.queue:
            embed.add_field(name="Up next", value=player.queue[0].title[:60], inline=False)
        if track.thumbnail:
            embed.set_thumbnail(url=track.thumbnail)
        return embed

    @app_commands.command(name="skip", description="Skip the current track.")
    async def skip(self, interaction: discord.Interaction) -> None:
        voice_client = interaction.guild.voice_client
        if voice_client is None or not self.music.skip(voice_client):
            await interaction.response.send_message("Nothing is playing.", ephemeral=True)
            return
        await interaction.response.send_message("Skipped.")

    @app_commands.command(name="stop", description="Stop playback and clear the queue.")
    async def stop(self, interaction: discord.Interaction) -> None:
        voice_client = interaction.guild.voice_client
        if voice_client is None:
            await interaction.response.send_message("I'm not in a voice channel.", ephemeral=True)
            return
        self.music.stop(voice_client)
        await interaction.response.send_message("Stopped and cleared the queue.")

    @app_commands.command(name="pause", description="Pause the current track.")
    async def pause(self, interaction: discord.Interaction) -> None:
        voice_client = interaction.guild.voice_client
        if voice_client is None or not voice_client.is_playing():
            await interaction.response.send_message("Nothing is playing.", ephemeral=True)
            return
        voice_client.pause()
        await interaction.response.send_message("Paused.")

    @app_commands.command(name="resume", description="Resume a paused track.")
    async def resume(self, interaction: discord.Interaction) -> None:
        voice_client = interaction.guild.voice_client
        if voice_client is None or not voice_client.is_paused():
            await interaction.response.send_message("Nothing is paused.", ephemeral=True)
            return
        voice_client.resume()
        await interaction.response.send_message("Resumed.")

    @app_commands.command(name="nowplaying", description="Show the current track.")
    async def nowplaying(self, interaction: discord.Interaction) -> None:
        player = self.music.player(interaction.guild.id)
        if player.current is None:
            await interaction.response.send_message("Nothing is playing.", ephemeral=True)
            return
        await interaction.response.send_message(
            embed=self._now_playing_embed(player.current, player)
        )

    @app_commands.command(name="queue", description="Show the upcoming tracks.")
    async def queue(
        self, interaction: discord.Interaction, page: app_commands.Range[int, 1, 20] = 1
    ) -> None:
        player = self.music.player(interaction.guild.id)
        if player.current is None and not player.queue:
            await interaction.response.send_message("The queue is empty.", ephemeral=True)
            return

        per_page = 10
        start = (page - 1) * per_page
        window = player.queue[start : start + per_page]

        lines = []
        if player.current and page == 1:
            lines.append(f"**Now:** [{player.current.title}]({player.current.url})\n")
        for index, track in enumerate(window, start=start + 1):
            lines.append(f"`{index}.` [{track.title}]({track.url}) · {track.pretty_duration}")

        total_seconds = sum(t.duration for t in player.queue)
        hours, rem = divmod(total_seconds, 3600)
        embed = discord.Embed(
            title="Queue",
            description="\n".join(lines) or "Nothing queued.",
            colour=0x2B2D31,
        )
        embed.set_footer(
            text=(
                f"{len(player.queue)} queued · {hours}h {rem // 60}m total · "
                f"loop: {'track' if player.loop_track else 'queue' if player.loop_queue else 'off'}"
            )
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="volume", description="Set playback volume (0-200%).")
    async def volume(
        self, interaction: discord.Interaction, percent: app_commands.Range[int, 0, 200]
    ) -> None:
        voice_client = interaction.guild.voice_client
        if voice_client is None:
            await interaction.response.send_message("I'm not in a voice channel.", ephemeral=True)
            return
        self.music.set_volume(voice_client, percent / 100)
        await interaction.response.send_message(f"Volume set to {percent}%.")

    @app_commands.command(name="loop", description="Loop the current track, the queue, or nothing.")
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="Off", value="off"),
            app_commands.Choice(name="Current track", value="track"),
            app_commands.Choice(name="Whole queue", value="queue"),
        ]
    )
    async def loop(
        self, interaction: discord.Interaction, mode: app_commands.Choice[str]
    ) -> None:
        player = self.music.player(interaction.guild.id)
        player.loop_track = mode.value == "track"
        player.loop_queue = mode.value == "queue"
        await interaction.response.send_message(f"Loop: **{mode.name.lower()}**.")

    @app_commands.command(name="shuffle", description="Shuffle the queue.")
    async def shuffle(self, interaction: discord.Interaction) -> None:
        import random

        player = self.music.player(interaction.guild.id)
        if len(player.queue) < 2:
            await interaction.response.send_message("Not enough tracks to shuffle.", ephemeral=True)
            return
        random.shuffle(player.queue)
        await interaction.response.send_message(f"Shuffled {len(player.queue)} tracks.")

    @app_commands.command(name="remove", description="Remove a track from the queue by position.")
    async def remove(
        self, interaction: discord.Interaction, position: app_commands.Range[int, 1, 200]
    ) -> None:
        player = self.music.player(interaction.guild.id)
        if position > len(player.queue):
            await interaction.response.send_message(
                f"There are only {len(player.queue)} tracks queued.", ephemeral=True
            )
            return
        removed = player.queue.pop(position - 1)
        await interaction.response.send_message(f"Removed **{removed.title}**.")


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(MusicCmds(bot))
