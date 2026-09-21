"""Database package — ORM models, session management, and migrations.

Re-exports all public symbols so that ``from .database import get_db``
and ``from .database import Generation as DBGeneration`` continue to work
without changing any importers.
"""

from .models import (
    AgentTool,
    Appointment,
    AudioChannel,
    Base,
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
    Message,
    ProfileChannelMapping,
    ProfileSample,
    Project,
    Story,
    StoryItem,
    Ticket,
    VoiceAgent,
    VoiceAgentVersion,
    VoiceProfile,
    WebhookDelivery,
)
from .session import SessionLocal, _db_path, engine, get_db, init_db

# RUF022: grouped by origin (models vs. session plumbing) rather than one
# flat alphabetical list; each group is sorted.
__all__ = [  # noqa: RUF022
    # Models
    "AgentTool",
    "Appointment",
    "AudioChannel",
    "Base",
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
    "Message",
    "ProfileChannelMapping",
    "ProfileSample",
    "Project",
    "Story",
    "StoryItem",
    "Ticket",
    "VoiceAgent",
    "VoiceAgentVersion",
    "VoiceProfile",
    "WebhookDelivery",
    # Session
    "SessionLocal",
    "_db_path",
    "engine",
    "get_db",
    "init_db",
]
