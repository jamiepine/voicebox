"""Role management, role icons, and self-assignable button roles."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from ..checks import require_operator

if TYPE_CHECKING:
    from ..bot import VoxGuardBot

log = logging.getLogger(__name__)

MAX_ICON_BYTES = 256 * 1024  # Discord's role-icon limit


class RoleButton(discord.ui.Button):
    """One self-assign button. Toggles the role for whoever clicks it."""

    def __init__(self, role_id: int, label: str, emoji: str | None, style: discord.ButtonStyle):
        super().__init__(
            label=label[:80],
            emoji=emoji or None,
            style=style,
            custom_id=f"voxguard:role:{role_id}",
        )
        self.role_id = role_id

    async def callback(self, interaction: discord.Interaction) -> None:
        member = interaction.user
        guild = interaction.guild
        if not isinstance(member, discord.Member) or guild is None:
            return

        role = guild.get_role(self.role_id)
        if role is None:
            await interaction.response.send_message("That role no longer exists.", ephemeral=True)
            return
        if role >= guild.me.top_role:
            await interaction.response.send_message(
                "I can't assign that role — it's above me in the hierarchy.", ephemeral=True
            )
            return
        # A self-assign button must never be a privilege-escalation route,
        # even if an admin misconfigures it onto a staff role.
        if role.permissions.administrator or role.permissions.manage_guild:
            await interaction.response.send_message(
                "That role grants elevated permissions, so it isn't self-assignable.",
                ephemeral=True,
            )
            return

        try:
            if role in member.roles:
                await member.remove_roles(role, reason="Self-assign button")
                await interaction.response.send_message(f"Removed {role.mention}.", ephemeral=True)
            else:
                await member.add_roles(role, reason="Self-assign button")
                await interaction.response.send_message(f"Added {role.mention}.", ephemeral=True)
        except discord.HTTPException as exc:
            await interaction.response.send_message(f"Failed: {exc}", ephemeral=True)


class RoleButtonView(discord.ui.View):
    """Persistent view rebuilt on startup from the stored reaction_roles rows."""

    def __init__(self, entries: list[tuple[int, str, str | None]]) -> None:
        super().__init__(timeout=None)
        for role_id, label, emoji in entries[:25]:
            self.add_item(RoleButton(role_id, label, emoji, discord.ButtonStyle.secondary))


class RolesCmds(commands.Cog):
    def __init__(self, bot: "VoxGuardBot") -> None:
        self.bot = bot

    role = app_commands.Group(name="role", description="Create and manage roles.")

    @role.command(name="create", description="Create a role.")
    @app_commands.describe(
        name="Role name", colour="Hex colour like #5865F2", hoist="Show separately in the member list"
    )
    @require_operator()
    async def role_create(
        self,
        interaction: discord.Interaction,
        name: str,
        colour: str | None = None,
        hoist: bool = False,
        mentionable: bool = False,
    ) -> None:
        parsed = discord.Colour.default()
        if colour:
            try:
                parsed = discord.Colour(int(colour.lstrip("#"), 16))
            except ValueError:
                await interaction.response.send_message(
                    "Colour must be hex, like `#5865F2`.", ephemeral=True
                )
                return
        await interaction.response.defer(thinking=True)
        try:
            created = await interaction.guild.create_role(
                name=name[:100],
                colour=parsed,
                hoist=hoist,
                mentionable=mentionable,
                permissions=discord.Permissions.none(),
                reason=f"Created by {interaction.user}",
            )
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Couldn't create the role: {exc}")
            return
        await interaction.followup.send(
            f"Created {created.mention} with no permissions — grant what it needs in Server Settings."
        )

    @role.command(name="delete", description="Delete a role.")
    @require_operator()
    async def role_delete(self, interaction: discord.Interaction, role: discord.Role) -> None:
        if role >= interaction.guild.me.top_role or role.is_default():
            await interaction.response.send_message(
                "I can't delete that role — it's at or above my highest role.", ephemeral=True
            )
            return
        await interaction.response.defer(thinking=True)
        name = role.name
        try:
            await role.delete(reason=f"Deleted by {interaction.user}")
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Failed: {exc}")
            return
        await interaction.followup.send(f"Deleted **@{name}**.")

    @role.command(name="icon", description="Upload an icon for a role (needs server boost level 2).")
    @app_commands.describe(role="Role to change", image="PNG/JPG under 256 KB", emoji="Or a unicode emoji")
    @require_operator()
    async def role_icon(
        self,
        interaction: discord.Interaction,
        role: discord.Role,
        image: discord.Attachment | None = None,
        emoji: str | None = None,
    ) -> None:
        if image is None and emoji is None:
            await interaction.response.send_message(
                "Provide an `image` or an `emoji`.", ephemeral=True
            )
            return
        if role >= interaction.guild.me.top_role:
            await interaction.response.send_message(
                "That role is at or above my highest role.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)
        try:
            if image is not None:
                if image.size > MAX_ICON_BYTES:
                    await interaction.followup.send("Icon must be under 256 KB.")
                    return
                if not (image.content_type or "").startswith("image/"):
                    await interaction.followup.send("That attachment isn't an image.")
                    return
                await role.edit(display_icon=await image.read(), reason=f"Icon by {interaction.user}")
            else:
                await role.edit(display_icon=emoji, reason=f"Icon by {interaction.user}")
        except discord.Forbidden:
            await interaction.followup.send(
                "Discord refused that — role icons need server boost level 2."
            )
            return
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Failed: {exc}")
            return
        await interaction.followup.send(f"Updated the icon for {role.mention}.")

    @role.command(name="give", description="Give a role to a member.")
    @require_operator()
    async def role_give(
        self, interaction: discord.Interaction, member: discord.Member, role: discord.Role
    ) -> None:
        if role >= interaction.guild.me.top_role:
            await interaction.response.send_message(
                "That role is at or above my highest role.", ephemeral=True
            )
            return
        try:
            await member.add_roles(role, reason=f"Assigned by {interaction.user}")
        except discord.HTTPException as exc:
            await interaction.response.send_message(f"Failed: {exc}", ephemeral=True)
            return
        await interaction.response.send_message(f"Gave {role.mention} to {member.mention}.")

    @role.command(name="take", description="Remove a role from a member.")
    @require_operator()
    async def role_take(
        self, interaction: discord.Interaction, member: discord.Member, role: discord.Role
    ) -> None:
        if role >= interaction.guild.me.top_role:
            await interaction.response.send_message(
                "That role is at or above my highest role.", ephemeral=True
            )
            return
        try:
            await member.remove_roles(role, reason=f"Removed by {interaction.user}")
        except discord.HTTPException as exc:
            await interaction.response.send_message(f"Failed: {exc}", ephemeral=True)
            return
        await interaction.response.send_message(f"Removed {role.mention} from {member.mention}.")

    @role.command(name="all", description="Give a role to every member (slow on big servers).")
    @require_operator()
    async def role_all(self, interaction: discord.Interaction, role: discord.Role) -> None:
        if role >= interaction.guild.me.top_role:
            await interaction.response.send_message(
                "That role is at or above my highest role.", ephemeral=True
            )
            return
        if role.permissions.administrator or role.permissions.manage_guild:
            await interaction.response.send_message(
                "I won't mass-assign a role with administrator or manage-server permissions.",
                ephemeral=True,
            )
            return
        await interaction.response.defer(thinking=True)
        added = 0
        for member in interaction.guild.members:
            if member.bot or role in member.roles:
                continue
            try:
                await member.add_roles(role, reason=f"Mass-assign by {interaction.user}")
                added += 1
            except discord.HTTPException:
                continue
        await interaction.followup.send(f"Gave {role.mention} to {added} member(s).")

    # -- self-assignable button roles ---------------------------------------

    buttonroles = app_commands.Group(
        name="buttonroles", description="Post a message with self-assign role buttons."
    )

    @buttonroles.command(name="create", description="Post a role picker with up to 5 roles.")
    @app_commands.describe(
        title="Heading for the picker",
        description="Text under the heading",
        role1="A self-assignable role",
    )
    @require_operator()
    async def buttonroles_create(
        self,
        interaction: discord.Interaction,
        title: str,
        role1: discord.Role,
        description: str | None = None,
        role2: discord.Role | None = None,
        role3: discord.Role | None = None,
        role4: discord.Role | None = None,
        role5: discord.Role | None = None,
    ) -> None:
        roles = [r for r in (role1, role2, role3, role4, role5) if r is not None]
        me = interaction.guild.me

        unusable = [r.name for r in roles if r >= me.top_role]
        if unusable:
            await interaction.response.send_message(
                f"These roles are at or above my highest role: {', '.join(unusable)}", ephemeral=True
            )
            return
        privileged = [
            r.name for r in roles if r.permissions.administrator or r.permissions.manage_guild
        ]
        if privileged:
            await interaction.response.send_message(
                f"These roles grant elevated permissions and can't be self-assigned: "
                f"{', '.join(privileged)}",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        embed = discord.Embed(
            title=title[:256], description=description or "Click a button to toggle a role.",
            colour=0x5865F2,
        )
        view = RoleButtonView([(r.id, r.name, None) for r in roles])
        try:
            message = await interaction.channel.send(embed=embed, view=view)
        except discord.HTTPException as exc:
            await interaction.followup.send(f"Couldn't post the picker: {exc}")
            return

        for role in roles:
            self.bot.runtime.store.add_reaction_role(
                interaction.guild.id, message.id, f"voxguard:role:{role.id}", role.id, role.name
            )
        await interaction.followup.send(f"Posted a role picker with {len(roles)} role(s).", ephemeral=True)

    @buttonroles.command(name="remove", description="Stop tracking a role-picker message.")
    @require_operator()
    async def buttonroles_remove(self, interaction: discord.Interaction, message_id: str) -> None:
        if not message_id.isdigit():
            await interaction.response.send_message("Give a numeric message ID.", ephemeral=True)
            return
        count = self.bot.runtime.store.delete_reaction_roles(interaction.guild.id, int(message_id))
        await interaction.response.send_message(
            f"Removed {count} button-role binding(s).", ephemeral=True
        )

    # -- autorole -----------------------------------------------------------

    @app_commands.command(name="autorole", description="Roles automatically given to new members.")
    @require_operator()
    async def autorole(
        self, interaction: discord.Interaction, role: discord.Role, remove: bool = False
    ) -> None:
        if role >= interaction.guild.me.top_role:
            await interaction.response.send_message(
                "That role is at or above my highest role.", ephemeral=True
            )
            return
        if not remove and (role.permissions.administrator or role.permissions.manage_guild):
            await interaction.response.send_message(
                "I won't auto-assign a role with administrator or manage-server permissions.",
                ephemeral=True,
            )
            return

        runtime = self.bot.runtime
        config = runtime.config(interaction.guild.id)
        current = {str(r) for r in config["welcome"]["autorole_ids"]}
        if remove:
            current.discard(str(role.id))
        else:
            current.add(str(role.id))
        config["welcome"]["autorole_ids"] = list(current)
        runtime.save_config(interaction.guild.id, config)
        await interaction.response.send_message(
            f"{role.mention} {'removed from' if remove else 'added to'} autoroles."
        )


async def setup(bot: "VoxGuardBot") -> None:
    await bot.add_cog(RolesCmds(bot))
