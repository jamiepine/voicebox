"""Database package — ORM models, session management, and migrations.

Re-exports all public symbols so that ``from .database import get_db``
and ``from .database import Generation as DBGeneration`` continue to work
without changing any importers.
"""

from .models import (
    Base,
    AudioChannel,
    ChannelDeviceMapping,
    EffectPreset,
    Generation,
    GenerationVersion,
    ProfileChannelMapping,
    ProfileSample,
    Project,
    Story,
    StoryItem,
    VoiceProfile,
)
from .session import engine, SessionLocal, _db_path, init_db, get_db

__all__ = [
    # Models
    "Base",
    "AudioChannel",
    "ChannelDeviceMapping",
    "EffectPreset",
    "Generation",
    "GenerationVersion",
    "ProfileChannelMapping",
    "ProfileSample",
    "Project",
    "Story",
    "StoryItem",
    "VoiceProfile",
    # Session
    "engine",
    "SessionLocal",
    "_db_path",
    "init_db",
    "get_db",
]
