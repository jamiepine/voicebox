"""ORM model definitions for the voicebox SQLite database."""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

from ..utils.capture_chords import (
    default_push_to_talk_chord,
    default_toggle_to_talk_chord,
)

Base = declarative_base()


class VoiceProfile(Base):
    """Voice profile.

    voice_type discriminates three flavours:
      - "cloned"   — traditional reference-audio profiles (all cloning engines)
      - "preset"   — engine-specific pre-built voice (e.g. Kokoro voices)
      - "designed"  — text-described voice (e.g. Qwen CustomVoice, future)
    """

    __tablename__ = "profiles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    language = Column(String, default="en")
    avatar_path = Column(String, nullable=True)
    effects_chain = Column(Text, nullable=True)

    # Voice type system — added v0.3.x
    voice_type = Column(String, default="cloned")  # "cloned" | "preset" | "designed"
    preset_engine = Column(String, nullable=True)   # e.g. "kokoro" — only for preset
    preset_voice_id = Column(String, nullable=True)  # e.g. "am_adam" — only for preset
    design_prompt = Column(Text, nullable=True)      # text description — only for designed
    default_engine = Column(String, nullable=True)   # auto-selected engine, locked for preset
    # Free-form character prompt used by the compose button and the
    # personality-rewrite path on /generate. Describes *what* this voice
    # says and how, orthogonal to how it sounds (handled by the preset /
    # cloning metadata above).
    personality = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProfileSample(Base):
    """Audio sample attached to a voice profile."""

    __tablename__ = "profile_samples"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=False)
    audio_path = Column(String, nullable=False)
    reference_text = Column(Text, nullable=False)


class Generation(Base):
    """A single TTS generation."""

    __tablename__ = "generations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=False)
    text = Column(Text, nullable=False)
    language = Column(String, default="en")
    audio_path = Column(String, nullable=True)
    duration = Column(Float, nullable=True)
    seed = Column(Integer)
    instruct = Column(Text)
    engine = Column(String, default="qwen")
    model_size = Column(String, nullable=True)
    status = Column(String, default="completed")
    error = Column(Text, nullable=True)
    is_favorited = Column(Boolean, default=False)
    # Origin of this generation — "manual" for plain /generate calls,
    # "personality_speak" for rows whose text was rewritten through the
    # profile's personality LLM before TTS. Future sources (bulk import,
    # agent replies, etc.) can extend this.
    source = Column(String, nullable=False, default="manual")
    created_at = Column(DateTime, default=datetime.utcnow)


class Story(Base):
    """A story that sequences multiple generations."""

    __tablename__ = "stories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StoryItem(Base):
    """Links a generation to a story at a specific timecode."""

    __tablename__ = "story_items"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    story_id = Column(String, ForeignKey("stories.id"), nullable=False)
    generation_id = Column(String, ForeignKey("generations.id"), nullable=False)
    version_id = Column(String, ForeignKey("generation_versions.id"), nullable=True)
    start_time_ms = Column(Integer, nullable=False, default=0)
    track = Column(Integer, nullable=False, default=0)
    trim_start_ms = Column(Integer, nullable=False, default=0)
    trim_end_ms = Column(Integer, nullable=False, default=0)
    volume = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Project(Base):
    """Audio studio project (JSON blob)."""

    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    data = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GenerationVersion(Base):
    """A version of a generation's audio (original, processed, alternate takes)."""

    __tablename__ = "generation_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    generation_id = Column(String, ForeignKey("generations.id"), nullable=False)
    label = Column(String, nullable=False)
    audio_path = Column(String, nullable=False)
    effects_chain = Column(Text, nullable=True)
    source_version_id = Column(String, ForeignKey("generation_versions.id"), nullable=True)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EffectPreset(Base):
    """Saved effect chain preset."""

    __tablename__ = "effect_presets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)
    effects_chain = Column(Text, nullable=False)
    is_builtin = Column(Boolean, default=False)
    sort_order = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)


class AudioChannel(Base):
    """Audio output channel (bus)."""

    __tablename__ = "audio_channels"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChannelDeviceMapping(Base):
    """Mapping between a channel and an OS audio device."""

    __tablename__ = "channel_device_mappings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    channel_id = Column(String, ForeignKey("audio_channels.id"), nullable=False)
    device_id = Column(String, nullable=False)


class ProfileChannelMapping(Base):
    """Many-to-many mapping between voice profiles and audio channels."""

    __tablename__ = "profile_channel_mappings"

    profile_id = Column(String, ForeignKey("profiles.id"), primary_key=True)
    channel_id = Column(String, ForeignKey("audio_channels.id"), primary_key=True)


class CaptureSettings(Base):
    """Singleton row holding user defaults for the capture/refine flow.

    Kept server-side so every window, CLI client, and API consumer reads the
    same preferences. The ``id`` column is always 1.
    """

    __tablename__ = "capture_settings"

    id = Column(Integer, primary_key=True, default=1)
    stt_model = Column(String, nullable=False, default="turbo")
    language = Column(String, nullable=False, default="auto")
    auto_refine = Column(Boolean, nullable=False, default=True)
    llm_model = Column(String, nullable=False, default="0.6B")
    smart_cleanup = Column(Boolean, nullable=False, default=True)
    self_correction = Column(Boolean, nullable=False, default=True)
    preserve_technical = Column(Boolean, nullable=False, default=True)
    allow_auto_paste = Column(Boolean, nullable=False, default=True)
    default_playback_voice_id = Column(String, nullable=True)
    # Default OFF — opting in is what triggers the macOS Input Monitoring TCC
    # prompt. We deliberately don't spawn the global keyboard tap until the
    # user flips this on so a fresh-install user doesn't see a scary
    # "Voicebox would like to receive keystrokes from any application" dialog
    # before they've even opened the Captures tab.
    hotkey_enabled = Column(Boolean, nullable=False, default=False)
    # Lists of keytap key names (e.g. "MetaRight", "ControlRight"). Right-hand
    # modifiers by default so they don't collide with left-hand shortcuts.
    chord_push_to_talk_keys = Column(
        JSON, nullable=False, default=default_push_to_talk_chord
    )
    chord_toggle_to_talk_keys = Column(
        JSON, nullable=False, default=default_toggle_to_talk_chord
    )
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GenerationSettings(Base):
    """Singleton row for long-form TTS generation preferences."""

    __tablename__ = "generation_settings"

    id = Column(Integer, primary_key=True, default=1)
    max_chunk_chars = Column(Integer, nullable=False, default=800)
    crossfade_ms = Column(Integer, nullable=False, default=50)
    normalize_audio = Column(Boolean, nullable=False, default=True)
    autoplay_on_generate = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CloudSettings(Base):
    """Singleton row holding the link to a Voicebox Cloud account.

    Populated by the "Log in with browser" pairing flow (see services/cloud.py):
    the browser hands back a one-time code, which the backend exchanges for an
    ``api_key`` it stores here. The key is a bearer credential for
    api.voicebox.sh — auth only, never an encryption key (E2E key material lives
    elsewhere). Stored in the local app database alongside the user's other data;
    moving it to the OS keychain is a future hardening step. The ``id`` is
    always 1; a null ``api_key`` means "not connected".
    """

    __tablename__ = "cloud_settings"

    id = Column(Integer, primary_key=True, default=1)
    api_key = Column(String, nullable=True)
    device_name = Column(String, nullable=True)
    account_user_id = Column(String, nullable=True)
    connected_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MCPClientBinding(Base):
    """Per-MCP-client settings (voice profile, engine, personality default).

    Lets users bind distinct voices to distinct agents — e.g. Claude Code
    speaks in "Morgan," Cursor in "Scarlett." The MCP client identifies
    itself via the ``X-Voicebox-Client-Id`` HTTP header; direct-HTTP
    clients set it in their MCP config's ``headers`` block, the stdio
    shim forwards it from the ``VOICEBOX_CLIENT_ID`` env var.
    """

    __tablename__ = "mcp_client_bindings"

    client_id = Column(String, primary_key=True)
    label = Column(String, nullable=True)  # display name
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=True)
    default_engine = Column(String, nullable=True)
    # When true, voicebox.speak routes through the profile's personality LLM
    # (rewrite) before TTS by default. Callers can still override per call.
    default_personality = Column(Boolean, nullable=False, default=False)
    last_seen_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Capture(Base):
    """A single voice input capture (dictation, recording, or uploaded file).

    Stores the original audio alongside the raw transcript and, optionally, a
    refined version produced by the LLM. Refinement flags are serialized as
    JSON so we can reproduce the prompt that generated the refined text.
    """

    __tablename__ = "captures"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    audio_path = Column(String, nullable=False)
    source = Column(String, nullable=False, default="file")  # dictation | recording | file
    language = Column(String, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    transcript_raw = Column(Text, nullable=False, default="")
    transcript_refined = Column(Text, nullable=True)
    stt_model = Column(String, nullable=True)
    llm_model = Column(String, nullable=True)
    refinement_flags = Column(Text, nullable=True)  # JSON blob
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Voice AI agent ────────────────────────────────────────────────────
# Conversational phone/voice agents driven by the local LLM (dialogue),
# TTS (the agent's voice) and Whisper (the customer's side). One
# ``VoiceAgent`` row is a fully configured persona for one of three modes:
# outbound sales (telemarketing), inbound customer service, or inbound
# support / issue resolution. See services/voice_agent.py for the call
# lifecycle and docs/overview/voice-agent.mdx for the user guide.


def _default_filler_phrases() -> list[str]:
    return ["One moment.", "Sure, let me check that for you.", "Okay, bear with me a second."]


def _default_sms_outcomes() -> list[str]:
    return ["interested"]


class VoiceAgent(Base):
    """A configured voice agent: identity, mode, what it may say, which
    voice it speaks in, and the compliance guard-rails.

    The LLM only ever sees ``brief`` / ``objection_notes`` / ``persona``
    plus retrieved knowledge articles and tool results, so the operator
    controls every claim the agent can make.
    """

    __tablename__ = "va_agents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    mode = Column(String, nullable=False, default="outbound_sales")  # outbound_sales | customer_service | support
    status = Column(String, nullable=False, default="draft")         # draft | active | paused | completed
    version = Column(Integer, nullable=False, default=1)
    # Voice + language the agent speaks in.
    profile_id = Column(String, ForeignKey("profiles.id"), nullable=False)
    engine = Column(String, nullable=True)
    language = Column(String, nullable=False, default="en")
    llm_model_size = Column(String, nullable=True)
    # Delivery instructions for engines that take them (Qwen3-TTS,
    # Qwen CustomVoice): the default style, and the one used once the
    # caller sounds upset.
    voice_style = Column(String, nullable=True)
    empathetic_voice_style = Column(String, nullable=True, default="calm, warm and apologetic")
    # Identity the agent presents on the call.
    agent_name = Column(String, nullable=False)
    company_name = Column(String, nullable=False)
    # What the agent is allowed to say / do. Supports {{contact.name}},
    # {{contact.custom.<field>}}, {{agent.company_name}}, {{today}} ...
    brief = Column(Text, nullable=False)          # offer facts (sales) or service scope (service/support)
    goal = Column(Text, nullable=False)
    objection_notes = Column(Text, nullable=True)
    persona = Column(Text, nullable=True)
    opening_line = Column(Text, nullable=True)
    # Spoken verbatim at the top of every call. Defaults to an AI-disclosure
    # sentence; operators can localise it but not blank it.
    disclosure = Column(Text, nullable=False)
    # Where unresolved issues / handoffs go — free text shown to the caller
    # ("a specialist will call you back within one business day").
    escalation_promise = Column(Text, nullable=True)
    # A/B script variants: [{"name", "weight", "opening_line"?, "brief"?, "goal"?}].
    variants = Column(JSON, nullable=True)
    # Latency masking: short phrases pre-generated in the agent's voice and
    # played the instant the caller stops talking, while the LLM thinks.
    filler_phrases = Column(JSON, nullable=False, default=_default_filler_phrases)
    filler_audio = Column(JSON, nullable=True)  # {phrase: generation_id}
    # Voice the first sentence of a reply as its own generation so audio
    # starts sooner on long replies (local calls only).
    fast_first_audio = Column(Boolean, nullable=False, default=True)
    # Tools the model may call mid-conversation (built-ins + va_tools rows).
    tools_enabled = Column(Boolean, nullable=False, default=True)
    booking_instructions = Column(Text, nullable=True)  # e.g. "weekday mornings, 30 minutes"
    appointment_duration_min = Column(Integer, nullable=False, default=30)
    # Post-call analysis: [{"key", "question", "type": "string|boolean|number|enum", "options"?}].
    analysis_schema = Column(JSON, nullable=True)
    # Post-call webhook (HMAC-SHA256 signed with webhook_secret).
    webhook_url = Column(String, nullable=True)
    webhook_secret = Column(String, nullable=True)
    # Compliance guard-rails (outbound only; inbound calls ignore the window).
    timezone = Column(String, nullable=False, default="UTC")
    calling_window_start = Column(Integer, nullable=False, default=9)   # local hour, inclusive
    calling_window_end = Column(Integer, nullable=False, default=20)    # local hour, exclusive
    calling_days = Column(JSON, nullable=False, default=lambda: [0, 1, 2, 3, 4])  # Mon=0 … Sun=6
    max_attempts = Column(Integer, nullable=False, default=3)
    daily_call_cap = Column(Integer, nullable=False, default=200)
    retry_delay_hours = Column(Integer, nullable=False, default=24)
    callback_delay_hours = Column(Integer, nullable=False, default=24)
    require_consent = Column(Boolean, nullable=False, default=False)
    max_turns = Column(Integer, nullable=False, default=30)
    # Auto-handoff after this many consecutive negative customer turns.
    handoff_after_negative_turns = Column(Integer, nullable=False, default=3)
    # Mask card numbers / national IDs / one-time codes in stored transcripts.
    redact_pii = Column(Boolean, nullable=False, default=True)
    # Campaign pacing (outbound): parallel lines (Twilio only) and an
    # optional run window.
    max_concurrent_calls = Column(Integer, nullable=False, default=1)
    schedule_start_at = Column(DateTime, nullable=True)
    schedule_end_at = Column(DateTime, nullable=True)
    # Telephony: "local" plays through the speakers and takes customer turns
    # over the API / MCP; "twilio" dials / answers real numbers via webhooks.
    provider = Column(String, nullable=False, default="local")
    from_number = Column(String, nullable=True)
    transfer_number = Column(String, nullable=True)     # warm transfer target for hand-offs (Twilio)
    voicemail_message = Column(Text, nullable=True)     # voicemail drop; templated
    sms_followup_template = Column(Text, nullable=True)  # templated; sent after the outcomes below
    sms_followup_outcomes = Column(JSON, nullable=False, default=_default_sms_outcomes)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VoiceAgentVersion(Base):
    """Snapshot of an agent's configuration, taken before every change so
    an operator can diff and roll back a script edit that hurt results."""

    __tablename__ = "va_agent_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    version = Column(Integer, nullable=False)
    snapshot = Column(JSON, nullable=False)
    note = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Contact(Base):
    """A person the agent talks to — an outbound lead or an inbound caller.

    ``phone`` is stored normalised (digits with a leading ``+`` where
    present) so DNC matching is exact. ``memory`` accumulates LLM call
    summaries so the next conversation starts with context.
    """

    __tablename__ = "va_contacts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    company = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    memory = Column(Text, nullable=True)
    timezone = Column(String, nullable=True)
    language = Column(String, nullable=True)          # overrides the agent's language for this person
    custom_fields = Column(JSON, nullable=True)       # {"plan": "Pro", ...} — usable as {{contact.custom.plan}}
    is_test = Column(Boolean, nullable=False, default=False)  # simulation stand-in; never dialled or counted
    # Operator's record that this person agreed to be contacted. The
    # scheduler refuses contacts without it when the agent requires consent.
    consent = Column(Boolean, nullable=False, default=False)
    # new | callback | calling | contacted | interested | not_interested |
    # resolved | unresolved | do_not_call | exhausted
    status = Column(String, nullable=False, default="new")
    attempts = Column(Integer, nullable=False, default=0)
    last_attempt_at = Column(DateTime, nullable=True)
    next_attempt_at = Column(DateTime, nullable=True)
    last_outcome = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeArticle(Base):
    """Grounding material the agent may quote from: FAQs, pricing, policies,
    troubleshooting steps. Retrieved per turn by keyword overlap and
    injected into the system prompt, so the agent answers from the
    operator's facts rather than the model's imagination."""

    __tablename__ = "va_knowledge"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(String, nullable=True)  # comma-separated
    source = Column(String, nullable=True)  # url / filename the entry was imported from
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentTool(Base):
    """An HTTP tool the model may call mid-conversation: look up an order,
    check stock, create a CRM record. Arguments the model supplies are
    validated against ``params`` before the request goes out."""

    __tablename__ = "va_tools"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    name = Column(String, nullable=False)          # snake_case, unique per agent
    description = Column(Text, nullable=False)     # when / why to use it — the model reads this
    method = Column(String, nullable=False, default="GET")
    url = Column(String, nullable=False)           # may contain {param} placeholders
    headers = Column(JSON, nullable=True)          # {"Authorization": "Bearer …"}
    params = Column(JSON, nullable=True)           # [{"name", "type", "description", "required"}]
    timeout_s = Column(Integer, nullable=False, default=10)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Call(Base):
    """One conversation (outbound dial attempt, inbound call, or a
    simulated test call) with its running state."""

    __tablename__ = "va_calls"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    contact_id = Column(String, ForeignKey("va_contacts.id"), nullable=False)
    direction = Column(String, nullable=False, default="outbound")  # outbound | inbound | simulation
    status = Column(String, nullable=False, default="in_progress")  # in_progress | completed | failed
    stage = Column(String, nullable=False, default="opening")       # opening | conversation | closing | ended
    # interested | not_interested | callback | opt_out | resolved |
    # unresolved | ticket_created | handoff | no_answer | voicemail |
    # voicemail_left | max_turns | error
    outcome = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    variant = Column(String, nullable=True)         # A/B variant name used on this call
    # Supervisor take-over: while paused the model stays silent and an
    # operator speaks as the agent through POST /calls/{id}/agent_say.
    ai_paused = Column(Boolean, nullable=False, default=False)
    # Post-call analysis answers keyed by analysis_schema keys, plus a
    # 0-100 goal-achievement score with the model's one-line reason.
    analysis = Column(JSON, nullable=True)
    score = Column(Integer, nullable=True)
    score_reason = Column(Text, nullable=True)
    flags = Column(JSON, nullable=True)  # ["injection_attempt", "pii_redacted", "appointment_booked", ...]
    webhook_status = Column(String, nullable=True)  # delivered | failed | skipped
    provider = Column(String, nullable=False, default="local")
    provider_call_id = Column(String, nullable=True)
    turn_count = Column(Integer, nullable=False, default=0)
    negative_streak = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    last_activity_at = Column(DateTime, default=datetime.utcnow)


class CallTurn(Base):
    """A single utterance on a call. Agent turns link to the TTS generation
    that voiced them (plus per-sentence chunk ids when fast-first-audio
    split the reply); customer turns carry a sentiment score and link to
    the capture when the audio came in through the API. ``tool`` turns
    record a tool call and its result and are never spoken."""

    __tablename__ = "va_call_turns"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    call_id = Column(String, ForeignKey("va_calls.id"), nullable=False)
    role = Column(String, nullable=False)  # agent | customer | tool
    text = Column(Text, nullable=False)
    source = Column(String, nullable=False, default="llm")  # llm | system | operator | tool
    sentiment = Column(Float, nullable=True)  # -1 … 1, customer turns only
    interrupted = Column(Boolean, nullable=False, default=False)
    stt_ms = Column(Integer, nullable=True)   # on agent turns: how long the preceding transcription took
    llm_ms = Column(Integer, nullable=True)   # on agent turns: model latency (all passes)
    tool_name = Column(String, nullable=True)
    meta = Column(JSON, nullable=True)        # tool args/result, chunk timings, …
    generation_id = Column(String, ForeignKey("generations.id"), nullable=True)
    generation_ids = Column(JSON, nullable=True)  # all chunk ids in playback order
    capture_id = Column(String, ForeignKey("captures.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Appointment(Base):
    """A booking the agent made with the built-in ``book_appointment`` tool."""

    __tablename__ = "va_appointments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    contact_id = Column(String, ForeignKey("va_contacts.id"), nullable=False)
    call_id = Column(String, ForeignKey("va_calls.id"), nullable=True)
    starts_at = Column(DateTime, nullable=False)  # UTC
    ends_at = Column(DateTime, nullable=False)    # UTC
    timezone = Column(String, nullable=True)      # the contact's zone, for display
    notes = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="booked")  # booked | confirmed | cancelled | completed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    """An outbound text message (SMS follow-up or the ``send_sms`` tool)."""

    __tablename__ = "va_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    contact_id = Column(String, ForeignKey("va_contacts.id"), nullable=False)
    call_id = Column(String, ForeignKey("va_calls.id"), nullable=True)
    channel = Column(String, nullable=False, default="sms")
    to_number = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="queued")  # sent | failed | unsent_no_provider
    provider_message_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class WebhookDelivery(Base):
    """One attempt-set at delivering a post-call webhook."""

    __tablename__ = "va_webhook_deliveries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    call_id = Column(String, ForeignKey("va_calls.id"), nullable=True)
    event = Column(String, nullable=False, default="call.ended")
    url = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending | delivered | failed
    attempts = Column(Integer, nullable=False, default=0)
    response_code = Column(Integer, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Ticket(Base):
    """An issue the agent could not close on the call: escalations, human
    handoffs, and support cases that need follow-up."""

    __tablename__ = "va_tickets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("va_agents.id"), nullable=False)
    contact_id = Column(String, ForeignKey("va_contacts.id"), nullable=False)
    call_id = Column(String, ForeignKey("va_calls.id"), nullable=True)
    kind = Column(String, nullable=False, default="support")  # support | handoff | callback | sales_lead
    priority = Column(String, nullable=False, default="normal")  # low | normal | high | urgent
    status = Column(String, nullable=False, default="open")  # open | in_progress | resolved | closed
    subject = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DoNotCallEntry(Base):
    """Global do-not-call list. Checked before every outbound dial, across
    agents. Rows are appended automatically when a customer opts out
    mid-call."""

    __tablename__ = "va_do_not_call"

    phone = Column(String, primary_key=True)
    reason = Column(String, nullable=True)
    source = Column(String, nullable=False, default="manual")  # manual | opt_out | import
    created_at = Column(DateTime, default=datetime.utcnow)
