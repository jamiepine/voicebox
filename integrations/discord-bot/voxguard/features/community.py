"""Community features: welcome, starboard, tickets, giveaways.

These are the engagement staples people install MEE6, Carl-bot, Ticket Tool
and GiveawayBot for. Grouped together because each is small and they share
nothing but the store.
"""

from __future__ import annotations

import datetime as dt
import logging
import random

import discord

from ..store import Store

log = logging.getLogger(__name__)


def render(template: str, member: discord.Member, guild: discord.Guild) -> str:
    """Fill the placeholders used by welcome/goodbye templates."""
    return (
        template.replace("{mention}", member.mention)
        .replace("{user}", member.display_name)
        .replace("{tag}", str(member))
        .replace("{server}", guild.name)
        .replace("{count}", str(guild.member_count or 0))
        .replace("{id}", str(member.id))
    )


class WelcomeManager:
    def __init__(self, store: Store) -> None:
        self.store = store

    async def on_join(self, member: discord.Member, config: dict) -> None:
        cfg = config.get("welcome", {})
        guild = member.guild

        # Autorole first: it matters even if the greeting fails to send.
        for role_id in cfg.get("autorole_ids", []):
            role = guild.get_role(int(role_id))
            if role is None or role >= guild.me.top_role:
                continue
            try:
                await member.add_roles(role, reason="Autorole on join")
            except discord.HTTPException as exc:
                log.warning("Autorole failed in %s: %s", guild.id, exc)

        if not cfg.get("enabled", False):
            return

        if channel_id := cfg.get("channel_id"):
            channel = guild.get_channel(int(channel_id))
            if isinstance(channel, discord.abc.Messageable):
                try:
                    await channel.send(render(cfg.get("message", ""), member, guild))
                except discord.HTTPException:
                    pass

        if dm := cfg.get("dm_message"):
            try:
                await member.send(render(dm, member, guild))
            except discord.HTTPException:
                pass

    async def on_leave(self, member: discord.Member, config: dict) -> None:
        cfg = config.get("welcome", {})
        if not cfg.get("goodbye_enabled", False):
            return
        channel_id = cfg.get("goodbye_channel_id") or cfg.get("channel_id")
        if not channel_id:
            return
        channel = member.guild.get_channel(int(channel_id))
        if isinstance(channel, discord.abc.Messageable):
            try:
                await channel.send(render(cfg.get("goodbye_message", ""), member, member.guild))
            except discord.HTTPException:
                pass


class Starboard:
    def __init__(self, store: Store) -> None:
        self.store = store

    async def on_reaction(
        self, payload: discord.RawReactionActionEvent, guild: discord.Guild, config: dict
    ) -> None:
        cfg = config.get("starboard", {})
        if not cfg.get("enabled", False) or not cfg.get("channel_id"):
            return
        if str(payload.emoji) != cfg.get("emoji", "⭐"):
            return
        if str(payload.channel_id) in {str(c) for c in cfg.get("ignore_channels", [])}:
            return
        if str(payload.channel_id) == str(cfg["channel_id"]):
            return

        source = guild.get_channel(payload.channel_id)
        if not isinstance(source, discord.abc.Messageable):
            return
        try:
            message = await source.fetch_message(payload.message_id)
        except discord.HTTPException:
            return

        reaction = discord.utils.find(
            lambda r: str(r.emoji) == cfg.get("emoji", "⭐"), message.reactions
        )
        stars = reaction.count if reaction else 0
        if not cfg.get("self_star", False) and reaction:
            # Discount the author starring themselves.
            try:
                async for user in reaction.users():
                    if user.id == message.author.id:
                        stars -= 1
                        break
            except discord.HTTPException:
                pass

        board = guild.get_channel(int(cfg["channel_id"]))
        if not isinstance(board, discord.abc.Messageable):
            return

        existing = self.store.get_star(guild.id, message.id)
        threshold = int(cfg.get("threshold", 3))

        if stars < threshold:
            # Dropped below the bar — remove the board post if we made one.
            if existing and existing["star_msg_id"]:
                try:
                    old = await board.fetch_message(int(existing["star_msg_id"]))
                    await old.delete()
                except discord.HTTPException:
                    pass
                self.store.upsert_star(guild.id, message.id, stars, None)
            return

        embed = discord.Embed(
            description=message.content or "",
            colour=0xF1C40F,
            timestamp=message.created_at,
        )
        embed.set_author(
            name=message.author.display_name,
            icon_url=message.author.display_avatar.url,
        )
        embed.add_field(name="Source", value=f"[Jump]({message.jump_url})", inline=False)
        if message.attachments:
            first = message.attachments[0]
            if (first.content_type or "").startswith("image/"):
                embed.set_image(url=first.url)

        content = f"{cfg.get('emoji', '⭐')} **{stars}** — {message.channel.mention}"

        if existing and existing["star_msg_id"]:
            try:
                post = await board.fetch_message(int(existing["star_msg_id"]))
                await post.edit(content=content, embed=embed)
                self.store.upsert_star(guild.id, message.id, stars, post.id)
                return
            except discord.HTTPException:
                pass

        try:
            post = await board.send(content=content, embed=embed)
            self.store.upsert_star(guild.id, message.id, stars, post.id)
            self.store.bump_metric(guild.id, "starboard_posts")
        except discord.HTTPException:
            pass


class GiveawayManager:
    def __init__(self, store: Store) -> None:
        self.store = store

    async def draw(self, bot: discord.Client, row) -> None:  # noqa: ANN001
        guild = bot.get_guild(int(row["guild_id"]))
        if guild is None:
            self.store.end_giveaway(int(row["id"]))
            return

        channel = guild.get_channel(int(row["channel_id"]))
        entries = self.store.giveaway_entries(int(row["id"]))
        self.store.end_giveaway(int(row["id"]))

        if not isinstance(channel, discord.abc.Messageable):
            return

        if not entries:
            try:
                await channel.send(f"🎉 Giveaway for **{row['prize']}** ended — no entries.")
            except discord.HTTPException:
                pass
            return

        count = min(int(row["winners"]), len(entries))
        winners = random.sample(entries, count)
        mentions = ", ".join(f"<@{w}>" for w in winners)
        try:
            await channel.send(f"🎉 Congratulations {mentions}! You won **{row['prize']}**.")
            self.store.bump_metric(guild.id, "giveaways_ended")
        except discord.HTTPException:
            pass


class TicketManager:
    def __init__(self, store: Store) -> None:
        self.store = store

    async def open(
        self, guild: discord.Guild, member: discord.Member, subject: str | None, config: dict
    ) -> tuple[discord.TextChannel | None, str]:
        cfg = config.get("tickets", {})
        if not cfg.get("enabled", False):
            return None, "Tickets aren't enabled here."

        if existing := self.store.open_ticket_for(guild.id, member.id):
            return None, f"You already have an open ticket: <#{existing['channel_id']}>"

        category = None
        if cat_id := cfg.get("category_id"):
            found = guild.get_channel(int(cat_id))
            category = found if isinstance(found, discord.CategoryChannel) else None

        overwrites = {
            guild.default_role: discord.PermissionOverwrite(view_channel=False),
            member: discord.PermissionOverwrite(view_channel=True, send_messages=True),
            guild.me: discord.PermissionOverwrite(view_channel=True, send_messages=True),
        }
        if role_id := cfg.get("support_role_id"):
            role = guild.get_role(int(role_id))
            if role:
                overwrites[role] = discord.PermissionOverwrite(view_channel=True, send_messages=True)

        try:
            channel = await guild.create_text_channel(
                f"ticket-{member.name}"[:100],
                category=category,
                overwrites=overwrites,
                topic=(subject or "")[:1024] or None,
                reason=f"Ticket opened by {member}",
            )
        except discord.HTTPException as exc:
            return None, f"Couldn't create the ticket channel: {exc}"

        self.store.open_ticket(guild.id, channel.id, member.id, subject)
        self.store.bump_metric(guild.id, "tickets_opened")

        support = f" <@&{cfg['support_role_id']}>" if cfg.get("support_role_id") else ""
        try:
            await channel.send(
                f"{member.mention}{support}\n{cfg.get('open_message', '')}"
                + (f"\n\n**Subject:** {subject}" if subject else "")
            )
        except discord.HTTPException:
            pass
        return channel, f"Opened {channel.mention}."

    async def close(
        self, channel: discord.TextChannel, closed_by: discord.Member, config: dict
    ) -> str:
        guild = channel.guild
        if not self.store.close_ticket(guild.id, channel.id, closed_by.id):
            return "This channel isn't an open ticket."

        self.store.bump_metric(guild.id, "tickets_closed")
        cfg = config.get("tickets", {})
        if log_id := cfg.get("log_channel_id"):
            log_channel = guild.get_channel(int(log_id))
            if isinstance(log_channel, discord.abc.Messageable):
                try:
                    await log_channel.send(
                        f"🎫 Ticket `{channel.name}` closed by {closed_by.mention}."
                    )
                except discord.HTTPException:
                    pass

        try:
            await channel.delete(reason=f"Ticket closed by {closed_by}")
        except discord.HTTPException as exc:
            return f"Marked closed, but couldn't delete the channel: {exc}"
        return "Ticket closed."
