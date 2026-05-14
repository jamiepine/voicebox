"""Database package — ORM models, session management, and migrations.

Re-exports all public symbols so that ``from .database import get_db``
and ``from .database import Generation as DBGeneration`` continue to work
without changing any importers.
"""

from .models import (
    AudioChannel,
    Base,
    Capture,
    CaptureSettings,
    ChannelDeviceMapping,
    EffectPreset,
    Generation,
    GenerationSettings,
    GenerationVersion,
    MCPClientBinding,
    ProfileChannelMapping,
    ProfileSample,
    Project,
    Story,
    StoryItem,
    VoiceProfile,
)
from .session import SessionLocal, _db_path, engine, get_db, init_db

__all__ = [
    "AudioChannel",
    # Models
    "Base",
    "Capture",
    "CaptureSettings",
    "ChannelDeviceMapping",
    "EffectPreset",
    "Generation",
    "GenerationSettings",
    "GenerationVersion",
    "MCPClientBinding",
    "ProfileChannelMapping",
    "ProfileSample",
    "Project",
    "SessionLocal",
    "Story",
    "StoryItem",
    "VoiceProfile",
    "_db_path",
    # Session
    "engine",
    "get_db",
    "init_db",
]
