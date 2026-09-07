"""Database package — ORM models, session management, and migrations.

Re-exports all public symbols so that ``from .database import get_db``
and ``from .database import Generation as DBGeneration`` continue to work
without changing any importers.
"""

from .models import (
    Base,
    AudioChannel,
    Call,
    CallTurn,
    Capture,
    CaptureSettings,
    ChannelDeviceMapping,
    CloudSettings,
    Contact,
    DoNotCallEntry,
    EffectPreset,
    Generation,
    GenerationSettings,
    GenerationVersion,
    KnowledgeArticle,
    MCPClientBinding,
    ProfileChannelMapping,
    ProfileSample,
    Project,
    Story,
    StoryItem,
    Ticket,
    VoiceAgent,
    VoiceProfile,
)
from .session import engine, SessionLocal, _db_path, init_db, get_db

__all__ = [
    # Models
    "Base",
    "AudioChannel",
    "Call",
    "CallTurn",
    "Capture",
    "CaptureSettings",
    "ChannelDeviceMapping",
    "CloudSettings",
    "Contact",
    "DoNotCallEntry",
    "EffectPreset",
    "Generation",
    "GenerationSettings",
    "GenerationVersion",
    "KnowledgeArticle",
    "MCPClientBinding",
    "ProfileChannelMapping",
    "ProfileSample",
    "Project",
    "Story",
    "StoryItem",
    "Ticket",
    "VoiceAgent",
    "VoiceProfile",
    # Session
    "engine",
    "SessionLocal",
    "_db_path",
    "init_db",
    "get_db",
]
