"""Shared runtime wiring — one instance, handed to every cog.

Keeping this separate from `bot.py` means cogs import a plain dataclass
instead of reaching back into the bot object for services.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import guardrails
from .agent import ServerAgent
from .config import Settings, merged_config
from .features.ai_moderation import AIModerator
from .features.automod import AntiNuke, TextAutomod
from .features.community import GiveawayManager, Starboard, TicketManager, WelcomeManager
from .features.levels import LevelEngine
from .features.logs import EventLogger
from .features.music import MusicPlayer
from .listener import SessionManager
from .matching import Matcher, Term
from .moderation import Enforcer
from .ollama_client import OllamaClient
from .raid import RaidDetector
from .store import Store
from .tts import Speaker
from .voice_notes import VoiceNoteModerator
from .voicebox_client import VoiceboxClient


class MatcherCache:
    """Rebuilds a guild's Matcher only when its word list actually changes."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._cache: dict[tuple[int, str], tuple[Matcher, float]] = {}
        self._dirty: dict[int, float] = {}

    def invalidate(self, guild_id: int) -> None:
        self._dirty[guild_id] = time.monotonic()

    def get(self, guild_id: int, scope: str) -> Matcher:
        key = (guild_id, scope)
        dirty_at = self._dirty.get(guild_id, 0)
        cached = self._cache.get(key)
        if cached and cached[1] >= dirty_at:
            return cached[0]

        blocked = [
            Term(row["term"], row["kind"], row["severity"])
            for row in self.store.list_terms(guild_id, scope, listing="block")
        ]
        allowed = [
            Term(row["term"], row["kind"], row["severity"])
            for row in self.store.list_terms(guild_id, scope, listing="allow")
        ]
        matcher = Matcher(blocked, allowed)
        self._cache[key] = (matcher, time.monotonic())
        return matcher


@dataclass
class Runtime:
    settings: Settings
    store: Store
    voicebox: VoiceboxClient
    ollama: OllamaClient
    limiter: guardrails.RateLimiter
    enforcer: Enforcer
    raid: RaidDetector
    agent: ServerAgent
    sessions: SessionManager
    speaker: Speaker
    voice_notes: VoiceNoteModerator
    matchers: MatcherCache
    levels: LevelEngine
    automod: TextAutomod
    ai_moderation: AIModerator
    antinuke: AntiNuke
    events: EventLogger
    welcome: WelcomeManager
    starboard: Starboard
    giveaways: GiveawayManager
    tickets: TicketManager
    music: MusicPlayer
    started_at: float = field(default_factory=time.time)
    # guild_id -> channel_id the bot is actively vc-talking in
    vctalk_active: dict[int, int] = field(default_factory=dict)
    # guild_id -> monotonic time of the last unprompted roam reply per channel
    roam_last_spoke: dict[tuple[int, int], float] = field(default_factory=dict)

    def config(self, guild_id: int) -> dict:
        return merged_config(self.store.get_config(guild_id))

    def save_config(self, guild_id: int, config: dict) -> None:
        self.store.save_config(guild_id, config)

    @classmethod
    def build(cls, settings: Settings) -> "Runtime":
        store = Store(settings.data_dir / "voxguard.sqlite3")
        voicebox = VoiceboxClient(
            settings.voicebox_url,
            whisper_model=settings.whisper_model,
            tts_engine=settings.tts_engine,
        )
        ollama = OllamaClient(
            settings.ollama_host, settings.ollama_model, auto_install=settings.auto_install_ollama
        )
        limiter = guardrails.RateLimiter(store)
        enforcer = Enforcer(store, limiter)
        raid = RaidDetector(store, limiter)
        agent = ServerAgent(ollama, store, limiter, settings.owner_ids)
        sessions = SessionManager(voicebox)
        speaker = Speaker(voicebox)
        voice_notes = VoiceNoteModerator(voicebox, enforcer)
        matchers = MatcherCache(store)

        return cls(
            settings=settings,
            store=store,
            voicebox=voicebox,
            ollama=ollama,
            limiter=limiter,
            enforcer=enforcer,
            raid=raid,
            agent=agent,
            sessions=sessions,
            speaker=speaker,
            voice_notes=voice_notes,
            matchers=matchers,
            levels=LevelEngine(store),
            automod=TextAutomod(store),
            ai_moderation=AIModerator(ollama, store),
            antinuke=AntiNuke(store),
            events=EventLogger(),
            welcome=WelcomeManager(store),
            starboard=Starboard(store),
            giveaways=GiveawayManager(store),
            tickets=TicketManager(store),
            music=MusicPlayer(),
        )

    async def aclose(self) -> None:
        await self.sessions.shutdown()
        await self.voicebox.close()
        await self.ollama.close()
        self.store.close()
