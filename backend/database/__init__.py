"""Database package — ORM models, session management, and migrations.

Re-exports all public symbols so that ``from .database import get_db``
and ``from .database import Generation as DBGeneration`` continue to work
without changing any importers.
"""

from .models import (
    Base,
    AgentTool,
    Appointment,
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
from .session import engine, SessionLocal, _db_path, init_db, get_db

__all__ = [
    # Models
    "Base",
    "AgentTool",
    "Appointment",
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
    "engine",
    "SessionLocal",
    "_db_path",
    "init_db",
    "get_db",
]
