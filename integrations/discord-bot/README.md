<p align="center">
  <img src="assets/logo-mark.svg" alt="VoxGuard" width="96">
</p>

<h1 align="center">VoxGuard</h1>

<p align="center">
  <strong>A Voicebox-powered Discord bot.</strong><br>
  Real-time voice moderation, a full moderation suite, raid and nuke protection,
  a local AI agent that speaks in a cloned voice — and a web dashboard for all of it.
</p>

---

VoxGuard is built on [Voicebox](../../README.md): **Voicebox does the speech-to-text
and the cloned-voice synthesis**, VoxGuard does the Discord side — voice capture,
slash commands, enforcement, and the agent's tool loop. The AI runs locally on
[Ollama](https://ollama.com); nothing is sent to a third-party API.

## Features

### Voice
| | |
|---|---|
| **Live VC moderation** | Joins a voice channel, transcribes every speaker in real time, matches against your blocklist, and enforces a warn → timeout → kick → ban ladder |
| **Obfuscation-resistant matching** | Catches `f u c k`, `fuuuck`, `sh1t` and transcription slips, with an allowlist so "Scunthorpe" doesn't trip it |
| **Voice notes** | Transcribes and moderates Discord's native voice messages |
| **Voice cloning** | `/voiceclone` clones a voice from an uploaded sample, consent-gated |
| **Live AI conversation** | `/vctalk` — speak to the bot in VC, it answers out loud in the cloned voice with LLM-driven emotional delivery |
| **Spoken commands** | Say *"VoxGuard, lock the general channel"* in voice chat and it does it. Each person's spoken commands run with **their own** permissions |
| **Voice utilities** | `/voice say`, `/voice status`, `/voice summary` (AI recap of a call), `/voice language`, `/voice transcript` |

### Moderation & security
| | |
|---|---|
| **Case system** | Every ban/kick/timeout/warn gets a numbered case with reason, history and a mod log |
| **Text automod** | Invites, links, mass mentions, flood/spam, all-caps, and word filters |
| **AI text moderation** | The local model reads what word lists structurally miss — threats with no slur in them, scams, coordinated harassment — and classifies by category, severity and confidence |
| **Raid detection** | Scores join bursts on rate, account age, avatars and name similarity → alert, lockdown, kick or ban |
| **Anti-nuke** | Detects one actor mass-deleting channels/roles or mass-banning, and strips their privileged roles |
| **Event logging** | Message edits/deletions, joins/leaves, voice activity, role changes, mod actions |
| **Channel control** | `/purge`, `/lock`, `/unlock`, `/slowmode` |

### Engagement
| | |
|---|---|
| **Levelling with voice XP** | Text *and* voice XP — time spent actually talking counts. `/rank`, `/leaderboard`, role rewards |
| **Button roles** | Self-assign role pickers that can't be used to escalate privileges |
| **Welcome & autorole** | Join/leave messages with placeholders, DM greetings, automatic roles |
| **Tickets** | Private support channels with a support role and transcript logging |
| **Starboard** | Highlights messages that clear a star threshold |
| **Giveaways** | Button-entry giveaways with automatic drawing |
| **Threads** | `/thread create/archive/lock`, plus auto-threading a channel |
| **Roles** | Full CRUD including **role icons**, colours, and mass-assign |
| **Tags, polls, info** | Canned responses, reaction polls, `/userinfo`, `/serverinfo`, `/avatar` |

### AI agent
`/chat` talks to it in text. `/personality-ai` sets its persona and bound voice.
`/roam` lets it participate in text channels on its own. It has **25 tools**
across three tiers — send messages, remember facts, look up members, create
channels/roles/threads, toggle bot features, edit word lists, grant XP, and (at
the top tier) timeout, kick, ban and purge.

Every route into the agent — `/chat`, `/roam`, and spoken commands — is capped by
the **invoker's own Discord permissions**. See [Guardrails](#guardrails).

## Dashboard

The bot serves a stats dashboard on `http://localhost:8420`:

- **Overview** — servers, members reached, bans/kicks/timeouts/warnings, automod and AI-moderation hits, error counts, moderation-actions-over-time chart, server-growth line, filter activity
- **Servers** — every server as a card with member count, total actions and which features are enabled; click through to
- **Server detail** — per-server stats, action charts, member-join graph, the full feature matrix with each feature's live configuration, recent cases, and the XP leaderboard
- **Error log** — every runtime failure with source, message, full traceback and counts by hour/day/week

It reads live gateway state, so latency and uptime are real-time. Monochrome
glass UI, inline stroke icons, and hand-built SVG charts — no CDN, no external
requests, works fully offline.

Set `VOXGUARD_DASHBOARD=1` and `VOXGUARD_DASHBOARD_TOKEN` in `.env`. It binds to
`127.0.0.1` by default; exposing it publicly requires a token of at least 32
characters, and it refuses to start otherwise.

## Requirements

- Python 3.11+ (3.13 supported)
- [FFmpeg](https://ffmpeg.org/) on `PATH` — voice playback
- A running [Voicebox](../../README.md) instance (default `http://127.0.0.1:17493`)
- [Ollama](https://ollama.com) — the model is pulled automatically on first use
- A Discord application with the **Server Members**, **Message Content** and
  **Voice States** privileged intents enabled

### Permissions

VoxGuard does **not** need Administrator. Grant only what you use:

| Feature | Permission |
|---|---|
| Voice moderation, `/vctalk` | View Channel, Connect, Speak |
| Warn / timeout | Moderate Members |
| Kick / ban | Kick Members / Ban Members |
| Channels, threads, slowmode, lockdown | Manage Channels, Manage Threads |
| Roles, autorole, button roles, level rewards | Manage Roles |
| Message logging, purge, starboard | Manage Messages, Read Message History |
| Anti-nuke attribution | View Audit Log |
| Server icon, welcome, tickets | Manage Server |

Administrator is only worth granting if you intend to enable the agent's `manage`
and `moderate` roam tiers and want it to operate without per-permission tuning —
and it is the configuration those guardrails exist for. Whatever you grant, the
bot's role must sit **above** any role it needs to act on.

## Setup

```bash
cd integrations/discord-bot
pip install -r requirements.txt
cp .env.example .env      # set DISCORD_TOKEN at minimum
python main.py
```

Commands sync globally on start (up to an hour to propagate the first time), or
instantly to `VOXGUARD_DEV_GUILD_ID` if set.

## Command reference

<details>
<summary><strong>Voice</strong></summary>

`/join` `/leave` `/catch` `/blacklist add|remove|list|clear`
`/voicenotes toggle|actions` `/voiceclone` `/vctalk join|stop` `/personality-ai`
`/voice commands|wakeword|say|status|language|transcript|summary` `/chat`
</details>

<details>
<summary><strong>Moderation</strong></summary>

`/ban` `/unban` `/kick` `/timeout` `/untimeout` `/warn` `/purge` `/lock` `/unlock`
`/slowmode` `/case view|reason|history` `/modlog` `/thread create|archive|lock|auto`
</details>

<details>
<summary><strong>Protection</strong></summary>

`/automod toggle|rule|allow-domain|log-channel|status`
`/aimod toggle|configure|category|status|test` `/antinuke toggle|configure|whitelist`
`/raid toggle|configure|lockdown|lift|status` `/guard status|dry-run|resume|immune-role|warnings|clear-warnings`
</details>

<details>
<summary><strong>Community</strong></summary>

`/rank` `/leaderboard` `/levels toggle|configure|reward|rewards|reset|give`
`/role create|delete|icon|give|take|all` `/buttonroles create|remove` `/autorole`
`/welcome configure|test` `/starboard configure` `/ticket setup|open|close`
`/giveaway start|end` `/tag show|set|delete|list` `/poll`
</details>

<details>
<summary><strong>Utility</strong></summary>

`/userinfo` `/serverinfo` `/avatar` `/stats` `/dashboard`
`/logs configure|status` `/data retention|forget|purge-now`
</details>

Configuration commands require Manage Server (or a `VOXGUARD_OWNER_IDS` entry).
This is re-checked in code, not left to Discord's default-permission UI, since
those commands gate automated moderation and an agent with destructive tools.

## Guardrails

VoxGuard can be handed broad permissions and an AI told to act on its own. These
stand between that and a bot that bans the wrong person off a bad transcription:

- **Immunity** — the guild owner, bots, and anyone with an immune role or
  permission are never auto-actioned. Extend with `/guard immune-role`.
- **Role hierarchy** — never acts on anyone at or above its own top role.
- **Capability tiers** — agent tools are split into `chat` (always), `manage`
  (channels/roles/icon) and `moderate` (timeout/kick/ban/purge). Only `chat` is on
  by default. Tools outside the enabled tiers are never offered to the model, so
  it can't be argued into them.
- **Human approval** — bans, kicks, channel/role deletion and purges post an
  approve/deny card by default (`/roam configure`).
- **Per-actor circuit breaker** — if automated enforcement exceeds an hourly
  budget it drops to log-only and alerts. Budgets are tracked per subsystem, so a
  runaway word list pauses the voice filter without disarming the agent. Resume
  with `/guard resume`.
- **`/guard dry-run`** — detect and report everything, apply nothing.
- **Speaking grants no authority** — spoken commands run with the intersection of
  what the guild enabled and what the *speaker* could already do by typing. A
  member who cannot `/ban` cannot ban by saying it out loud. Discord attributes
  every audio packet to a user, so this is enforced per utterance.
- **Idle chatter never gets tools** — an utterance only widens past the chat tier
  when it both addresses the bot by name and reads like an instruction.
- **AI moderation needs confidence** — a verdict below the confidence floor is
  logged, never enforced, and `/aimod test` lets you tune it against real
  examples before it acts.
- **No privilege escalation** — the agent creates roles with zero permissions and
  refuses to hand out administrator or manage-server roles; button roles refuse
  the same.
- **Anti-nuke covers the bot too** — if the agent ever went on a spree, the same
  detector that catches a compromised admin catches it.

None of this makes the moderation decisions *correct* — it's speech recognition
and an LLM, both of which make mistakes. Review `/guard status` and the dashboard
after changing a word list.

## Data & privacy

This bot records voice channels. Defaults are finite, not forever:

| Data | Default retention |
|---|---|
| Voice transcripts on infractions | 30 days, then blanked |
| Infraction rows | 180 days |
| AI conversation history | 30 days |
| Audit trail | 365 days |
| Error log | 30 days |

Tune with `/data retention`, apply immediately with `/data purge-now`, and erase
one member completely with `/data forget`. Tell your members the bot transcribes
voice — in many jurisdictions you must.

**Voice cloning** requires typing an exact consent attestation, recorded with the
uploader's ID and timestamp. That doesn't verify consent — nothing can — but it
creates accountability and an audit trail. See Voicebox's
[Responsible Use](../../RESPONSIBLE_USE.md) policy, which this bot inherits.

## Architecture

```text
voxguard/
  config.py           settings + per-guild config schema
  store.py            sqlite: config, word lists, cases, XP, memory, consent, metrics
  matching.py         blocklist matching (exact/obfuscated/fuzzy/regex + ReDoS guard)
  guardrails.py       immunity, hierarchy, per-actor circuit breaker
  moderation.py       detection -> enforcement ladder
  voicebox_client.py  Voicebox API: /transcribe, /profiles, /generate
  ollama_client.py    Ollama bootstrap, model pull, tool-calling chat
  audio.py            PCM helpers (numpy; no audioop, so 3.13-safe)
  listener.py         per-speaker utterance segmentation + transcription
  tts.py              Voicebox WAV -> Discord playback
  agent.py            tiered tool registry + approval flow
  raid.py             join-burst scoring, lockdown
  roam.py             unprompted text presence
  vctalk.py           live voice conversation
  voiceclone.py       consent-gated cloning
  voice_notes.py      voice-message moderation
  voicecommands.py    wake-word routing + per-speaker authority
  features/
    levels.py         text + voice XP, rank roles
    automod.py        text rules + anti-nuke
    ai_moderation.py  LLM content classification
    community.py      welcome, starboard, tickets, giveaways
    logs.py           event logging
  dashboard/
    server.py         aiohttp API
    static/           dashboard UI (vanilla JS, hand-built SVG charts)
  cogs/               slash commands
  bot.py              client + event wiring
```
