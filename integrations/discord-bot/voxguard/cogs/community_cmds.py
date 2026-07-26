"""Community features: welcome, starboard, tickets, giveaways, tags, polls."""

from __future__ import annotations

import datetime as dt
import time
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator
from ..cogs.mod_cmds import parse_duration

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

POLL_EMOJI = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]


class GiveawayView(discord.ui.View):
    """Persistent enter/leave button for a giveaway."""

    def __init__(self, bot: "VoxGuardBot") -> None:
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="Enter", emoji="🎉", style=discord.ButtonStyle.primary,
                       custom_id="voxguard:giveaway:enter")
    async def enter(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        store = self.bot.runtime.store
        row = store.giveaway_by_message(interaction.message.id)
        if row is None or row["ended"]:
            await interaction.response.send_message("That giveaway has ended.", ephemeral=True)
            return
        if store.enter_giveaway(int(row["id"]), interaction.user.id):
            count = len(store.giveaway_entries(int(row["id"])))
            await interaction.response.send_message(
                f"You're entered — {count} entrant(s) so far. Click again to withdraw.",
                ephemeral=True,
            )
        else:
            store.leave_giveaway(int(row["id"]), interaction.user.id)
            await interaction.response.send_message("You've withdrawn your entry.", ephemeral=True)


class CommunityCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    # -- welcome ------------------------------------------------------------

    welcome = app_commands.Group(name="welcome", description="Welcome and goodbye messages.")

    @welcome.command(name="configure", description="Set up join/leave messages.")
    @app_commands.describe(
        channel="Where to post welcomes",
        message="Supports {mention} {user} {server} {count}",
        goodbye_message="Message when someone leaves",
        dm_message="Optional DM sent to the new member",
    )
    @require_operator()
    async def welcome_configure(
        self,
        interaction: discord.Interaction,
        enabled: bool | None = None,
        channel: discord.TextChannel | None = None,
        message: str | None = None,
        goodbye_enabled: bool | None = None,
        goodbye_message: str | None = None,
        dm_message: str | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["welcome"]
        if enabled is not None:
            cfg["enabled"] = enabled
        if channel is not None:
            cfg["channel_id"] = channel.id
        if message:
            cfg["message"] = message[:1500]
        if goodbye_enabled is not None:
            cfg["goodbye_enabled"] = goodbye_enabled
        if goodbye_message:
            cfg["goodbye_message"] = goodbye_message[:1500]
        if dm_message:
            cfg["dm_message"] = dm_message[:1500]
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message("Welcome settings updated.")

    @welcome.command(name="test", description="Preview the welcome message on yourself.")
    @require_operator()
    async def welcome_test(self, interaction: discord.Interaction) -> None:
        from ..features.community import render

        cfg = self.bot.runtime.config(interaction.guild.id)["welcome"]
        preview = render(cfg.get("message", ""), interaction.user, interaction.guild)
        await interaction.response.send_message(f"Preview:\n{preview}", ephemeral=True)

    # -- starboard ----------------------------------------------------------

    starboard = app_commands.Group(name="starboard", description="Highlight popular messages.")

    @starboard.command(name="configure", description="Set up the starboard.")
    @require_operator()
    async def starboard_configure(
        self,
        interaction: discord.Interaction,
        enabled: bool | None = None,
        channel: discord.TextChannel | None = None,
        threshold: app_commands.Range[int, 1, 50] | None = None,
        emoji: str | None = None,
        allow_self_star: bool | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["starboard"]
        if enabled is not None:
            cfg["enabled"] = enabled
        if channel is not None:
            cfg["channel_id"] = channel.id
        if threshold is not None:
            cfg["threshold"] = threshold
        if emoji:
            cfg["emoji"] = emoji.strip()
        if allow_self_star is not None:
            cfg["self_star"] = allow_self_star
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"Starboard {'on' if cfg['enabled'] else 'off'} — "
            f"{cfg['threshold']}× {cfg['emoji']} required."
        )

    # -- tickets ------------------------------------------------------------

    ticket = app_commands.Group(name="ticket", description="Support tickets.")

    @ticket.command(name="setup", description="Configure the ticket system.")
    @require_operator()
    async def ticket_setup(
        self,
        interaction: discord.Interaction,
        enabled: bool | None = None,
        category: discord.CategoryChannel | None = None,
        support_role: discord.Role | None = None,
        log_channel: discord.TextChannel | None = None,
        open_message: str | None = None,
    ) -> None:
        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        cfg = config["tickets"]
        if enabled is not None:
            cfg["enabled"] = enabled
        if category is not None:
            cfg["category_id"] = category.id
        if support_role is not None:
            cfg["support_role_id"] = support_role.id
        if log_channel is not None:
            cfg["log_channel_id"] = log_channel.id
        if open_message:
            cfg["open_message"] = open_message[:1000]
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message("Ticket settings updated.")

    @ticket.command(name="open", description="Open a private support ticket.")
    async def ticket_open(
        self, interaction: discord.Interaction, subject: str | None = None
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        _, note = await self.bot.runtime.tickets.open(
            interaction.guild,
            interaction.user,
            subject,
            self.bot.runtime.config(interaction.guild.id),
        )
        await interaction.followup.send(note)

    @ticket.command(name="close", description="Close the ticket you're in.")
    async def ticket_close(self, interaction: discord.Interaction) -> None:
        if not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message("Run this inside a ticket.", ephemeral=True)
            return
        await interaction.response.send_message("Closing this ticket…")
        note = await self.bot.runtime.tickets.close(
            interaction.channel, interaction.user, self.bot.runtime.config(interaction.guild.id)
        )
        if "closed" not in note.lower():
            await interaction.followup.send(note, ephemeral=True)

    # -- giveaways ----------------------------------------------------------

    giveaway = app_commands.Group(name="giveaway", description="Run giveaways.")

    @giveaway.command(name="start", description="Start a giveaway.")
    @app_commands.describe(
        prize="What's being given away",
        duration="How long it runs: 30m, 6h, 2d",
        winners="How many winners to draw",
    )
    @require_operator()
    async def giveaway_start(
        self,
        interaction: discord.Interaction,
        prize: str,
        duration: str,
        winners: app_commands.Range[int, 1, 20] = 1,
    ) -> None:
        seconds = parse_duration(duration)
        if not seconds:
            await interaction.response.send_message(
                "Couldn't read that duration. Try `30m`, `6h`, or `2d`.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)
        ends_at = time.time() + seconds
        store = self.bot.runtime.store
        giveaway_id = store.create_giveaway(
            interaction.guild.id, interaction.channel.id, prize, winners, interaction.user.id, ends_at
        )

        embed = discord.Embed(
            title="🎉 Giveaway", description=f"**{prize}**", colour=0xF1C40F
        )
        embed.add_field(name="Winners", value=str(winners), inline=True)
        embed.add_field(
            name="Ends",
            value=discord.utils.format_dt(
                discord.utils.utcnow() + dt.timedelta(seconds=seconds), "R"
            ),
            inline=True,
        )
        embed.set_footer(text=f"Hosted by {interaction.user.display_name}")

        message = await interaction.channel.send(embed=embed, view=GiveawayView(self.bot))
        store.set_giveaway_message(giveaway_id, message.id)
        await interaction.followup.send("Giveaway started.", ephemeral=True)

    @giveaway.command(name="end", description="End a giveaway early and draw now.")
    @require_operator()
    async def giveaway_end(self, interaction: discord.Interaction, message_id: str) -> None:
        if not message_id.isdigit():
            await interaction.response.send_message("Give a numeric message ID.", ephemeral=True)
            return
        row = self.bot.runtime.store.giveaway_by_message(int(message_id))
        if row is None:
            await interaction.response.send_message("No giveaway with that message ID.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        await self.bot.runtime.giveaways.draw(self.bot, row)
        await interaction.followup.send("Drawn.", ephemeral=True)

    # -- tags ---------------------------------------------------------------

    tag = app_commands.Group(name="tag", description="Reusable canned responses.")

    @tag.command(name="show", description="Post a saved tag.")
    async def tag_show(self, interaction: discord.Interaction, name: str) -> None:
        row = self.bot.runtime.store.get_tag(interaction.guild.id, name)
        if row is None:
            await interaction.response.send_message(f"No tag named `{name}`.", ephemeral=True)
            return
        # allowed_mentions guards against a tag being used to mass-ping.
        await interaction.response.send_message(
            row["content"], allowed_mentions=discord.AllowedMentions.none()
        )

    @tag.command(name="set", description="Create or update a tag.")
    @require_operator()
    async def tag_set(self, interaction: discord.Interaction, name: str, content: str) -> None:
        self.bot.runtime.store.set_tag(interaction.guild.id, name, content, interaction.user.id)
        await interaction.response.send_message(f"Saved tag `{name.lower()}`.")

    @tag.command(name="delete", description="Delete a tag.")
    @require_operator()
    async def tag_delete(self, interaction: discord.Interaction, name: str) -> None:
        ok = self.bot.runtime.store.delete_tag(interaction.guild.id, name)
        await interaction.response.send_message(
            f"Deleted `{name.lower()}`." if ok else f"No tag named `{name}`.", ephemeral=not ok
        )

    @tag.command(name="list", description="List all tags.")
    async def tag_list(self, interaction: discord.Interaction) -> None:
        rows = self.bot.runtime.store.list_tags(interaction.guild.id)
        if not rows:
            await interaction.response.send_message("No tags yet.", ephemeral=True)
            return
        listing = ", ".join(f"`{r['name']}` ({r['uses']})" for r in rows[:60])
        await interaction.response.send_message(f"**Tags:** {listing}", ephemeral=True)

    # -- polls --------------------------------------------------------------

    @app_commands.command(name="poll", description="Run a quick reaction poll (up to 5 options).")
    async def poll(
        self,
        interaction: discord.Interaction,
        question: str,
        option1: str,
        option2: str,
        option3: str | None = None,
        option4: str | None = None,
        option5: str | None = None,
    ) -> None:
        options = [o for o in (option1, option2, option3, option4, option5) if o]
        embed = discord.Embed(title=question[:256], colour=0x5865F2)
        embed.description = "\n".join(
            f"{POLL_EMOJI[i]} {opt}" for i, opt in enumerate(options)
        )
        embed.set_footer(text=f"Poll by {interaction.user.display_name}")

        await interaction.response.send_message(embed=embed)
        message = await interaction.original_response()
        for i in range(len(options)):
            try:
                await message.add_reaction(POLL_EMOJI[i])
            except discord.HTTPException:
                break


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(CommunityCmds(bot))
