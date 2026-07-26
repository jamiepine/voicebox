"""Shared slash-command permission gate.

Discord's per-command permission UI (`default_permissions=...`) is a
default, not an enforcement mechanism — a server admin can reopen any
command to `@everyone` from Integrations settings. Since these commands
configure automated moderation and an agent with ban/kick/channel-management
tools, the operator check is re-verified in code on every call rather than
trusted to Discord's UI alone.
"""

from __future__ import annotations

import discord
from discord import app_commands

from . import guardrails


class NotAnOperator(app_commands.CheckFailure):
    pass


def require_operator():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None or not isinstance(interaction.user, discord.Member):
            raise NotAnOperator("This command only works in a server.")

        bot = interaction.client
        owner_ids = getattr(getattr(bot, "settings", None), "owner_ids", set())
        if guardrails.is_operator(interaction.user, interaction.guild, owner_ids):
            return True
        raise NotAnOperator(
            "You need `Manage Server` (or Administrator) to use this command."
        )

    return app_commands.check(predicate)


async def on_app_command_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
) -> None:
    if isinstance(error, NotAnOperator):
        message = str(error) or "You don't have permission to use this command."
    else:
        message = f"Something went wrong: {error}"

    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)
