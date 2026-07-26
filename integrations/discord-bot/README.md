# VoxGuard — a Voicebox-powered Discord bot

VoxGuard is a Discord bot for live voice-channel moderation, voice-message
moderation, raid detection, and an Ollama-backed server agent with a cloned
voice. It's built as an integration on top of Voicebox: **Voicebox does the
STT and TTS, this bot does the Discord-specific plumbing** (voice capture,
slash commands, enforcement, the agent's tool loop).

It talks to a running Voicebox instance over its REST API — see the main
[README's API section](../../README.md#api) for the endpoints this bot uses
(`/transcribe`, `/profiles`, `/generate/stream`).

## What it does

- **Live voice moderation** — joins a VC with `/join`, transcribes speech in
  real time, and matches it against a per-guild blacklist (typos, spacing
  tricks, and repeated-letter obfuscation included). `/catch` configures the
  response ladder (warn → timeout/kick/ban).
- **Voice-message moderation** — `/voicenotes toggle` moderates Discord's
  native voice messages the same way.
- **Raid detection** — `/raid` scores join bursts on rate, account age,
  missing avatars, and name similarity, then can alert, lock the server down,
  or remove the accounts involved.
- **A local AI agent** — backed by Ollama (auto-pulls the model on first
  use). `/personality-ai` sets its tone; `/roam` lets it participate in text
  channels unprompted and, if you enable it, manage channels/roles or
  moderate members — gated by capability tiers and (for destructive actions)
  human approval.
- **Voice cloning + live conversation** — `/voiceclone` clones a voice from
  an uploaded sample (consent-gated), `/vctalk join` starts a live spoken
  conversation with the agent in that voice.

## Requirements

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/) on `PATH` (voice playback)
- A running [Voicebox](../../README.md) instance (defaults to
  `http://127.0.0.1:17493`)
- [Ollama](https://ollama.com) — the bot starts it and pulls the configured
  model automatically if the `ollama` binary is present; see
  `VOXGUARD_AUTO_INSTALL_OLLAMA` in `.env.example` if you want the bot to
  install the binary itself
- A Discord bot application with the **Server Members**, **Message
  Content**, and **Voice States** privileged intents enabled, invited with
  the `applications.commands` and `bot` scopes and (per the design goal of
  this bot) the **Administrator** permission

## Setup

```bash
cd integrations/discord-bot
pip install -r requirements.txt
cp .env.example .env
# edit .env: DISCORD_TOKEN at minimum
python main.py
```

Slash commands sync globally on startup (can take up to an hour to propagate
the first time) or instantly to `VOXGUARD_DEV_GUILD_ID` if set.

## Command reference

| Command | Purpose |
|---|---|
| `/join <channel>` | Join a voice channel and start live moderation |
| `/leave` | Leave the current voice channel |
| `/catch` | Configure the enforcement ladder for live voice (warn/timeout/kick/ban) |
| `/blacklist add/remove/list/clear` | Manage blocked (and allowed-exception) word lists, by text or uploaded `.txt` |
| `/voicenotes toggle` / `/voicenotes actions` | Turn voice-message moderation on/off and configure its response |
| `/raid toggle` / `/raid configure` / `/raid lockdown` / `/raid lift` | Raid detection and manual server lockdown |
| `/guard status` / `/guard dry-run` / `/guard resume` / `/guard immune-role` | Cross-cutting safety controls (see below) |
| `/personality-ai` | Set the agent's persona, bound voice, model, and emotion |
| `/voiceclone` | Clone a voice from an uploaded sample (requires a typed consent attestation) |
| `/vctalk join/stop` | Start/stop a live spoken conversation with the agent |
| `/roam toggle/channels/configure/status` | Let the agent speak unprompted in text channels, with optional management/moderation tool access |

All configuration commands require Manage Server (or a user ID listed in
`VOXGUARD_OWNER_IDS`) — this is re-checked in code, not left to Discord's
default-permission UI, since it gates automated moderation and an agent with
destructive tools.

## Guardrails (read this before turning on `/roam moderate`)

This bot can be handed Administrator and told to "make its own rules," per
the design brief. A few things stand between that and a bot that bans the
wrong person on a bad transcription:

- **Immunity** — the guild owner, bots, and anyone with an immune role/
  permission (Administrator, Manage Server, Moderate Members, by default)
  are never auto-actioned. Add more with `/guard immune-role`.
- **Role hierarchy** — the bot refuses to act on anyone at or above its own
  top role, same as Discord would refuse the action anyway.
- **Capability tiers** — the agent's tools are split into `chat` (always on),
  `manage` (channels/roles/server icon), and `moderate` (timeout/kick/ban/
  purge). Only `chat` is on by default; `/roam toggle` turns the others on
  explicitly.
- **Human approval for irreversible actions** — bans, kicks, channel/role
  deletion, and purges post an approve/deny card by default
  (`require_confirm_destructive`, on by default — see `/roam configure`).
- **An hourly circuit breaker** — if automated enforcement (filter-based or
  agent-initiated) exceeds a per-hour budget, it drops to log-only and
  alerts. This is usually the sign of a bad word list or a false-positive
  storm, not a sign to act faster. Resume with `/guard resume`.
- **`/guard dry-run`** — log every detection and every agent tool call
  without applying anything, useful when first configuring a word list or a
  personality.

None of this makes the bot's moderation decisions correct — it's speech
recognition and an LLM, both of which make mistakes. Review `/guard status`
and `/guard warnings <member>` periodically, especially right after changing
a word list.

## Voice cloning and consent

`/voiceclone` requires typing an exact consent attestation
(`I OWN THIS VOICE OR HAVE PERMISSION TO CLONE IT`) before it will touch
Voicebox. This doesn't verify consent — it can't — but it records who
claimed it, when, and for which sample, so a server owner has an audit trail
if a clone turns out to be non-consensual. See Voicebox's own
[Responsible Use](../../RESPONSIBLE_USE.md) policy, which this bot inherits:
don't clone a voice you don't have the right to use.

## Architecture

```
voxguard/
  config.py        settings + per-guild config schema/merge
  store.py          sqlite persistence (word lists, infractions, memory, consent, audit)
  matching.py        blacklist matching (exact/obfuscated/fuzzy/regex, with allowlist)
  voicebox_client.py  async client for Voicebox's /transcribe, /profiles, /generate
  ollama_client.py     Ollama bootstrap, model pull, chat
  guardrails.py         immunity, hierarchy, circuit breaker — shared by every action
  moderation.py           detection -> warn/timeout/kick/ban ladder
  audio.py                 PCM helpers (numpy, no audioop)
  listener.py                per-speaker utterance segmentation + transcription
  tts.py                      Voicebox WAV -> Discord voice playback
  voice_notes.py                Discord voice-message moderation
  raid.py                        join-burst scoring + lockdown/kick/ban response
  agent.py                        Ollama tool-calling agent + tiered tool registry
  vctalk.py                        live voice conversation (listener + agent + tts)
  roam.py                           unprompted text-channel presence
  voiceclone.py                      consent-gated cloning
  runtime.py                          wires all of the above together
  bot.py                                discord.py client + event handlers
  cogs/                                  slash commands
```
