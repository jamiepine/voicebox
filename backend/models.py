"""
Pydantic models for request/response validation.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from .utils.capture_chords import (
    default_push_to_talk_chord,
    default_toggle_to_talk_chord,
)


class VoiceProfileCreate(BaseModel):
    """Request model for creating a voice profile."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    language: str = Field(
        default="en", pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$"
    )
    voice_type: str | None = Field(default="cloned", pattern="^(cloned|preset|designed)$")
    preset_engine: str | None = Field(None, max_length=50)
    preset_voice_id: str | None = Field(None, max_length=100)
    design_prompt: str | None = Field(None, max_length=2000)
    default_engine: str | None = Field(None, max_length=50)
    personality: str | None = Field(None, max_length=2000)


class VoiceProfileResponse(BaseModel):
    """Response model for voice profile."""

    id: str
    name: str
    description: str | None
    language: str
    avatar_path: str | None = None
    effects_chain: list["EffectConfig"] | None = None
    voice_type: str = "cloned"
    preset_engine: str | None = None
    preset_voice_id: str | None = None
    design_prompt: str | None = None
    default_engine: str | None = None
    personality: str | None = None
    generation_count: int = 0
    sample_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProfileSampleCreate(BaseModel):
    """Request model for adding a sample to a profile."""

    reference_text: str = Field(..., min_length=1, max_length=1000)


class ProfileSampleUpdate(BaseModel):
    """Request model for updating a profile sample."""

    reference_text: str = Field(..., min_length=1, max_length=1000)


class ProfileSampleResponse(BaseModel):
    """Response model for profile sample."""

    id: str
    profile_id: str
    audio_path: str
    reference_text: str

    class Config:
        from_attributes = True


class GenerationRequest(BaseModel):
    """Request model for voice generation."""

    profile_id: str
    text: str = Field(..., min_length=1, max_length=50000)
    language: str = Field(
        default="en", pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$"
    )
    seed: int | None = Field(None, ge=0)
    model_size: str | None = Field(default="1.7B", pattern="^(1\\.7B|0\\.6B|1B|3B)$")
    instruct: str | None = Field(None, max_length=500)
    engine: str | None = Field(
        default="qwen", pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$"
    )
    personality: bool = Field(
        default=False,
        description="When true and the profile has a personality prompt, the input text is rewritten in-character before TTS.",
    )
    max_chunk_chars: int = Field(
        default=800, ge=100, le=5000, description="Max characters per chunk for long text splitting"
    )
    crossfade_ms: int = Field(
        default=50, ge=0, le=500, description="Crossfade duration in ms between chunks (0 for hard cut)"
    )
    normalize: bool = Field(default=True, description="Normalize output audio volume")
    effects_chain: list["EffectConfig"] | None = Field(
        None, description="Effects chain to apply after generation (overrides profile default)"
    )


class GenerationResponse(BaseModel):
    """Response model for voice generation."""

    id: str
    profile_id: str
    text: str
    language: str
    audio_path: str | None = None
    duration: float | None = None
    seed: int | None = None
    instruct: str | None = None
    engine: str | None = "qwen"
    model_size: str | None = None
    status: str = "completed"
    error: str | None = None
    is_favorited: bool = False
    source: str = "manual"
    created_at: datetime
    versions: list["GenerationVersionResponse"] | None = None
    active_version_id: str | None = None

    class Config:
        from_attributes = True


class HistoryQuery(BaseModel):
    """Query model for generation history."""

    profile_id: str | None = None
    search: str | None = None
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class HistoryResponse(BaseModel):
    """Response model for history entry (includes profile name)."""

    id: str
    profile_id: str
    profile_name: str
    text: str
    language: str
    audio_path: str | None = None
    duration: float | None = None
    seed: int | None = None
    instruct: str | None = None
    engine: str | None = "qwen"
    model_size: str | None = None
    status: str = "completed"
    error: str | None = None
    is_favorited: bool = False
    created_at: datetime
    versions: list["GenerationVersionResponse"] | None = None
    active_version_id: str | None = None

    class Config:
        from_attributes = True


class HistoryListResponse(BaseModel):
    """Response model for history list."""

    items: list[HistoryResponse]
    total: int


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""

    language: str | None = Field(None, pattern="^(en|zh|ja|ko|de|fr|ru|pt|es|it)$")
    model: str | None = Field(None, pattern="^(base|small|medium|large|turbo)$")


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""

    text: str
    duration: float


class RefinementFlagsModel(BaseModel):
    """Boolean toggles that drive the refinement prompt builder."""

    smart_cleanup: bool = True
    self_correction: bool = True
    preserve_technical: bool = True


class CaptureResponse(BaseModel):
    """Response model for a capture."""

    id: str
    audio_path: str
    source: str
    language: str | None = None
    duration_ms: int | None = None
    transcript_raw: str
    transcript_refined: str | None = None
    stt_model: str | None = None
    llm_model: str | None = None
    refinement_flags: RefinementFlagsModel | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class CaptureListResponse(BaseModel):
    """Response model for paginated capture list."""

    items: list[CaptureResponse]
    total: int


class CaptureCreateResponse(CaptureResponse):
    """
    Response model for ``POST /captures``.

    Adds ``auto_refine`` and ``allow_auto_paste`` — the server-side settings
    captured at the moment the capture was created. The client reads these to
    decide whether to chain a refinement request and whether to fire the
    synthetic-paste pipeline, so it doesn't need a synced local copy of the
    capture_settings table across sibling Tauri webviews.
    """

    auto_refine: bool
    allow_auto_paste: bool


class CaptureRefineRequest(BaseModel):
    """Request to refine a capture's transcript via the LLM."""

    flags: RefinementFlagsModel | None = None
    model_size: str | None = Field(default=None, pattern="^(0\\.6B|1\\.7B|4B)$")


class CaptureRetranscribeRequest(BaseModel):
    """Request to re-run STT on a capture's audio with a different model."""

    model: str | None = Field(None, pattern="^(base|small|medium|large|turbo)$")
    language: str | None = Field(None, pattern="^(en|zh|ja|ko|de|fr|ru|pt|es|it)$")


class CaptureSettingsResponse(BaseModel):
    """Server-persisted defaults for the capture / refine flow."""

    stt_model: str = Field(default="turbo", pattern="^(base|small|medium|large|turbo)$")
    language: str = Field(default="auto")
    auto_refine: bool = True
    llm_model: str = Field(default="0.6B", pattern="^(0\\.6B|1\\.7B|4B)$")
    smart_cleanup: bool = True
    self_correction: bool = True
    preserve_technical: bool = True
    allow_auto_paste: bool = True
    default_playback_voice_id: str | None = None
    hotkey_enabled: bool = False
    chord_push_to_talk_keys: list[str] = Field(default_factory=default_push_to_talk_chord)
    chord_toggle_to_talk_keys: list[str] = Field(default_factory=default_toggle_to_talk_chord)

    class Config:
        from_attributes = True


class CaptureSettingsUpdate(BaseModel):
    """Partial update for capture settings — every field is optional."""

    stt_model: str | None = Field(default=None, pattern="^(base|small|medium|large|turbo)$")
    language: str | None = None
    auto_refine: bool | None = None
    llm_model: str | None = Field(default=None, pattern="^(0\\.6B|1\\.7B|4B)$")
    smart_cleanup: bool | None = None
    self_correction: bool | None = None
    preserve_technical: bool | None = None
    allow_auto_paste: bool | None = None
    default_playback_voice_id: str | None = None
    hotkey_enabled: bool | None = None
    chord_push_to_talk_keys: list[str] | None = Field(default=None, min_length=1, max_length=6)
    chord_toggle_to_talk_keys: list[str] | None = Field(default=None, min_length=1, max_length=6)


class GenerationSettingsResponse(BaseModel):
    """Server-persisted defaults for the generation flow."""

    max_chunk_chars: int = Field(default=800, ge=100, le=5000)
    crossfade_ms: int = Field(default=50, ge=0, le=500)
    normalize_audio: bool = True
    autoplay_on_generate: bool = True

    class Config:
        from_attributes = True


class GenerationSettingsUpdate(BaseModel):
    """Partial update for generation settings — every field is optional."""

    max_chunk_chars: int | None = Field(default=None, ge=100, le=5000)
    crossfade_ms: int | None = Field(default=None, ge=0, le=500)
    normalize_audio: bool | None = None
    autoplay_on_generate: bool | None = None


class MCPClientBindingResponse(BaseModel):
    """Per-MCP-client voice binding — what voice / engine the server should
    use when a given client_id calls voicebox.speak without args, plus an
    opt-in personality-rewrite default."""

    client_id: str
    label: str | None = None
    profile_id: str | None = None
    default_engine: str | None = Field(
        None,
        pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$",
    )
    default_personality: bool = False
    last_seen_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MCPClientBindingUpsert(BaseModel):
    """Create or update a binding. Matched by ``client_id``."""

    client_id: str = Field(..., min_length=1, max_length=64)
    label: str | None = Field(None, max_length=128)
    profile_id: str | None = None
    default_engine: str | None = Field(
        None,
        pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$",
    )
    default_personality: bool = False


class MCPClientBindingListResponse(BaseModel):
    items: list[MCPClientBindingResponse]


class SpeakRequest(BaseModel):
    """Body for POST /speak — non-MCP REST surface that mirrors voicebox.speak."""

    text: str = Field(..., min_length=1, max_length=10000)
    profile: str | None = Field(
        None,
        description="Voice profile name or id. Falls back to per-client binding, then default.",
    )
    engine: str | None = Field(
        None,
        pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$",
    )
    personality: bool | None = Field(
        None,
        description="When true and the profile has a personality prompt, the input text is rewritten in-character before TTS. When null, the per-client binding's default_personality flag decides.",
    )
    language: str | None = Field(
        None,
        pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$",
    )


class LLMGenerateRequest(BaseModel):
    """Request model for LLM text generation."""

    prompt: str = Field(..., min_length=1, max_length=50000)
    system: str | None = Field(None, max_length=4000)
    model_size: str | None = Field(default="0.6B", pattern="^(0\\.6B|1\\.7B|4B)$")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Few-shot (user, assistant) pairs prepended as real chat turns.
    # Used by the refinement service to pin tricky rules (imperatives
    # staying imperatives, technical-term punctuation) that small models
    # lose when the examples live inline in the system prompt.
    examples: list[list[str]] | None = Field(default=None, max_length=8)


class LLMGenerateResponse(BaseModel):
    """Response model for LLM text generation."""

    text: str
    model_size: str


# ── Profile personality endpoint ──────────────────────────────────────
# The sole standalone personality endpoint is ``/profiles/{id}/compose``,
# which produces a fresh in-character utterance the UI drops into the
# generate textarea. Rewrite is now reached via ``/generate`` with
# ``personality=true``.


class PersonalityTextResponse(BaseModel):
    """Response returned by the ``/profiles/{id}/compose`` endpoint."""

    text: str
    model_size: str


class ModelReadiness(BaseModel):
    """Per-model entry in the dictation readiness checklist.

    ``model_name`` is the canonical id used by ``POST /models/download`` so the
    frontend can wire a one-click "Download" button without a second lookup.
    ``size`` is the user's chosen variant (e.g. "turbo", "0.6B"); ``display_name``
    is what the checklist row should show ("Whisper Turbo").
    """

    ready: bool
    model_name: str
    display_name: str
    size: str
    size_mb: int | None = None


class CaptureReadinessResponse(BaseModel):
    """Backend gates that must be green before the global hotkey will fire.

    The frontend combines this with its own TCC permission checks (input
    monitoring, accessibility) into the full dictation readiness checklist.
    Hotkey-enabled is the user's intent toggle and lives outside this struct.
    """

    stt: ModelReadiness
    llm: ModelReadiness


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_downloaded: bool | None = None  # Whether model is cached/downloaded
    model_size: str | None = None  # Current model size if loaded
    gpu_available: bool
    gpu_type: str | None = None  # GPU type (CUDA, MPS, or None)
    vram_used_mb: float | None = None
    backend_type: str | None = None  # Backend type (mlx or pytorch)
    backend_variant: str | None = None  # Binary variant (cpu, cuda, or rocm)
    supports_rocm: bool = False  # AMD GPU on Windows — the ROCm backend is applicable
    gpu_compatibility_warning: str | None = None  # Warning if GPU arch unsupported


class DirectoryCheck(BaseModel):
    """Health status for a single directory."""

    path: str
    exists: bool
    writable: bool
    error: str | None = None


class FilesystemHealthResponse(BaseModel):
    """Response model for filesystem health check."""

    healthy: bool
    disk_free_mb: float | None = None
    disk_total_mb: float | None = None
    directories: list[DirectoryCheck]


class ModelStatus(BaseModel):
    """Response model for model status."""

    model_name: str
    display_name: str
    hf_repo_id: str | None = None  # HuggingFace repository ID
    downloaded: bool
    downloading: bool = False  # True if download is in progress
    size_mb: float | None = None
    loaded: bool = False


class ModelStatusListResponse(BaseModel):
    """Response model for model status list."""

    models: list[ModelStatus]


class ModelDownloadRequest(BaseModel):
    """Request model for triggering model download."""

    model_name: str


class ModelMigrateRequest(BaseModel):
    """Request model for migrating models to a new directory."""

    destination: str


class ActiveDownloadTask(BaseModel):
    """Response model for active download task."""

    model_name: str
    status: str
    started_at: datetime
    error: str | None = None
    progress: float | None = None  # 0-100 percentage
    current: int | None = None  # bytes downloaded
    total: int | None = None  # total bytes
    filename: str | None = None  # current file being downloaded


class ActiveGenerationTask(BaseModel):
    """Response model for active generation task."""

    task_id: str
    profile_id: str
    text_preview: str
    started_at: datetime


class ActiveTasksResponse(BaseModel):
    """Response model for active tasks."""

    downloads: list[ActiveDownloadTask]
    generations: list[ActiveGenerationTask]


class AudioChannelCreate(BaseModel):
    """Request model for creating an audio channel."""

    name: str = Field(..., min_length=1, max_length=100)
    device_ids: list[str] = Field(default_factory=list)


class AudioChannelUpdate(BaseModel):
    """Request model for updating an audio channel."""

    name: str | None = Field(None, min_length=1, max_length=100)
    device_ids: list[str] | None = None


class AudioChannelResponse(BaseModel):
    """Response model for audio channel."""

    id: str
    name: str
    is_default: bool
    device_ids: list[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ChannelVoiceAssignment(BaseModel):
    """Request model for assigning voices to a channel."""

    profile_ids: list[str]


class ProfileChannelAssignment(BaseModel):
    """Request model for assigning channels to a profile."""

    channel_ids: list[str]


class StoryCreate(BaseModel):
    """Request model for creating a story."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)


class StoryResponse(BaseModel):
    """Response model for story (list view)."""

    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    item_count: int = 0

    class Config:
        from_attributes = True


class StoryItemDetail(BaseModel):
    """Detail model for story item with generation info."""

    id: str
    story_id: str
    generation_id: str
    version_id: str | None = None
    start_time_ms: int
    track: int = 0
    trim_start_ms: int = 0
    trim_end_ms: int = 0
    created_at: datetime
    # Generation details
    profile_id: str
    profile_name: str
    text: str
    language: str
    audio_path: str
    duration: float
    seed: int | None
    instruct: str | None
    engine: str | None = None
    volume: float = 1.0
    generation_created_at: datetime
    # Versions available for this generation
    versions: list["GenerationVersionResponse"] | None = None
    active_version_id: str | None = None

    class Config:
        from_attributes = True


class StoryDetailResponse(BaseModel):
    """Response model for story with items."""

    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    items: list[StoryItemDetail] = []

    class Config:
        from_attributes = True


class StoryItemCreate(BaseModel):
    """Request model for adding a generation to a story."""

    generation_id: str
    start_time_ms: int | None = None  # If not provided, will be calculated automatically
    track: int | None = 0  # Track number (0 = main track)


class StoryItemUpdateTime(BaseModel):
    """Request model for updating a story item's timecode."""

    generation_id: str
    start_time_ms: int = Field(..., ge=0)


class StoryItemBatchUpdate(BaseModel):
    """Request model for batch updating story item timecodes."""

    updates: list[StoryItemUpdateTime]


class StoryItemReorder(BaseModel):
    """Request model for reordering story items."""

    generation_ids: list[str] = Field(..., min_length=1)


class StoryItemMove(BaseModel):
    """Request model for moving a story item (position and/or track)."""

    start_time_ms: int = Field(..., ge=0)
    track: int = 0


class StoryItemTrim(BaseModel):
    """Request model for trimming a story item."""

    trim_start_ms: int = Field(..., ge=0)
    trim_end_ms: int = Field(..., ge=0)


class StoryItemSplit(BaseModel):
    """Request model for splitting a story item."""

    split_time_ms: int = Field(..., ge=0)  # Time within the clip to split at (relative to clip start)


class StoryItemVersionUpdate(BaseModel):
    """Request model for setting a story item's pinned version."""

    version_id: str | None = None  # null = use generation default


class StoryItemVolumeUpdate(BaseModel):
    """Request model for adjusting a story item's playback volume.

    Linear gain. ``1.0`` is the original level, ``0.0`` is silent. Capped
    above 1.0 so a too-aggressive boost can't blow out the mix or clip
    the export.
    """

    volume: float = Field(..., ge=0.0, le=2.0)


class EffectConfig(BaseModel):
    """A single effect in an effects chain."""

    type: str
    enabled: bool = True
    params: dict = Field(default_factory=dict)


class EffectsChain(BaseModel):
    """An ordered list of effects to apply."""

    effects: list[EffectConfig] = Field(default_factory=list)


class EffectPresetCreate(BaseModel):
    """Request model for creating an effect preset."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    effects_chain: list[EffectConfig]


class EffectPresetUpdate(BaseModel):
    """Request model for updating an effect preset."""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    effects_chain: list[EffectConfig] | None = None


class EffectPresetResponse(BaseModel):
    """Response model for effect preset."""

    id: str
    name: str
    description: str | None = None
    effects_chain: list[EffectConfig]
    is_builtin: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class GenerationVersionResponse(BaseModel):
    """Response model for a generation version."""

    id: str
    generation_id: str
    label: str
    audio_path: str
    effects_chain: list[EffectConfig] | None = None
    source_version_id: str | None = None
    is_default: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ApplyEffectsRequest(BaseModel):
    """Request to apply effects to an existing generation."""

    effects_chain: list[EffectConfig]
    source_version_id: str | None = Field(
        None, description="Version to use as source audio (defaults to clean/original)"
    )
    label: str | None = Field(None, max_length=100, description="Label for this version (auto-generated if omitted)")
    set_as_default: bool = Field(default=True, description="Set this version as the default")


class ProfileEffectsUpdate(BaseModel):
    """Request to update the default effects chain on a profile."""

    effects_chain: list[EffectConfig] | None = Field(None, description="Effects chain (null to remove)")


class AvailableEffectParam(BaseModel):
    """Description of a single effect parameter."""

    default: float
    min: float
    max: float
    step: float
    description: str


class AvailableEffect(BaseModel):
    """Description of an available effect type."""

    type: str
    label: str
    description: str
    params: dict  # param_name -> AvailableEffectParam


class AvailableEffectsResponse(BaseModel):
    """Response listing all available effect types."""

    effects: list[AvailableEffect]


# ─── Cloud (backup & sync) ──────────────────────────────────────────────


class CloudLoginStartResponse(BaseModel):
    """Returned when the desktop kicks off browser login. The backend has
    already opened the browser; the URL is included for fallback/debugging."""

    authorize_url: str


class CloudStatusResponse(BaseModel):
    """Current link between this device and a Voicebox Cloud account."""

    connected: bool
    device_name: str | None = None
    account_user_id: str | None = None
    key_prefix: str | None = None
    connected_at: datetime | None = None
    dashboard_url: str


# ── Voice AI agent ────────────────────────────────────────────────────

_VA_LANGUAGE_PATTERN = "^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$"
_VA_ENGINE_PATTERN = "^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$"
_VA_MODE_PATTERN = "^(outbound_sales|customer_service|support)$"
_VA_OUTCOME_PATTERN = (
    "^(interested|not_interested|callback|opt_out|resolved|unresolved|"
    "ticket_created|handoff|no_answer|voicemail|voicemail_left|max_turns|error)$"
)
_VA_SNAKE_PATTERN = "^[a-z][a-z0-9_]{0,39}$"

DEFAULT_DISCLOSURE = "Just so you know, I'm an automated AI assistant and this call may be recorded."
DEFAULT_FILLER_PHRASES = ["One moment.", "Sure, let me check that for you.", "Okay, bear with me a second."]
DEFAULT_EMPATHETIC_STYLE = "calm, warm and apologetic"


class ScriptVariant(BaseModel):
    """An A/B variant: any field left null falls back to the agent's base script."""

    name: str = Field(..., min_length=1, max_length=60)
    weight: int = Field(default=1, ge=1, le=100)
    opening_line: str | None = Field(None, max_length=1000)
    brief: str | None = Field(None, max_length=6000)
    goal: str | None = Field(None, max_length=1000)


class AnalysisField(BaseModel):
    """One post-call question the LLM answers from the transcript."""

    key: str = Field(..., pattern=_VA_SNAKE_PATTERN)
    question: str = Field(..., min_length=1, max_length=300)
    type: str = Field(default="string", pattern="^(string|boolean|number|enum)$")
    options: list[str] | None = Field(None, max_length=20)


class _VoiceAgentFields(BaseModel):
    """Everything an operator configures. Shared by create (required
    fields) and update (all optional) via subclassing."""

    engine: str | None = Field(None, pattern=_VA_ENGINE_PATTERN)
    llm_model_size: str | None = Field(None, pattern="^(0\\.6B|1\\.7B|4B)$")
    voice_style: str | None = Field(None, max_length=200)
    empathetic_voice_style: str | None = Field(DEFAULT_EMPATHETIC_STYLE, max_length=200)
    objection_notes: str | None = Field(None, max_length=4000)
    persona: str | None = Field(None, max_length=2000, description="Tone / manner of speaking.")
    opening_line: str | None = Field(None, max_length=1000)
    escalation_promise: str | None = Field(None, max_length=500)
    variants: list[ScriptVariant] | None = Field(None, max_length=8)
    filler_phrases: list[str] = Field(default_factory=lambda: list(DEFAULT_FILLER_PHRASES), max_length=6)
    fast_first_audio: bool = True
    tools_enabled: bool = True
    booking_instructions: str | None = Field(None, max_length=1000)
    appointment_duration_min: int = Field(default=30, ge=5, le=480)
    analysis_schema: list[AnalysisField] | None = Field(None, max_length=20)
    webhook_url: str | None = Field(None, max_length=500)
    webhook_secret: str | None = Field(None, max_length=200)
    timezone: str = Field(default="UTC", max_length=64)
    calling_window_start: int = Field(default=9, ge=0, le=23)
    calling_window_end: int = Field(default=20, ge=1, le=24)
    calling_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    max_attempts: int = Field(default=3, ge=1, le=20)
    daily_call_cap: int = Field(default=200, ge=1, le=100000)
    retry_delay_hours: int = Field(default=24, ge=1, le=720)
    callback_delay_hours: int = Field(default=24, ge=1, le=720)
    require_consent: bool = False
    max_turns: int = Field(default=30, ge=2, le=200)
    handoff_after_negative_turns: int = Field(default=3, ge=1, le=20)
    redact_pii: bool = True
    max_concurrent_calls: int = Field(default=1, ge=1, le=20)
    schedule_start_at: datetime | None = None
    schedule_end_at: datetime | None = None
    provider: str = Field(default="local", pattern="^(local|twilio)$")
    from_number: str | None = Field(None, max_length=32)
    transfer_number: str | None = Field(None, max_length=32)
    voicemail_message: str | None = Field(None, max_length=1000)
    sms_followup_template: str | None = Field(None, max_length=640)
    sms_followup_outcomes: list[str] = Field(default_factory=lambda: ["interested"], max_length=13)

    @field_validator("filler_phrases")
    @classmethod
    def _filler_short(cls, phrases: list[str]) -> list[str]:
        cleaned = [p.strip() for p in phrases if p and p.strip()]
        for p in cleaned:
            if len(p) > 80:
                raise ValueError("Filler phrases must be at most 80 characters.")
        return cleaned

    @field_validator("webhook_url")
    @classmethod
    def _webhook_scheme(cls, url: str | None) -> str | None:
        if url and not url.lower().startswith(("http://", "https://")):
            raise ValueError("webhook_url must start with http:// or https://")
        return url


class VoiceAgentCreate(_VoiceAgentFields):
    """Body for ``POST /agents``."""

    name: str = Field(..., min_length=1, max_length=100)
    mode: str = Field(default="outbound_sales", pattern=_VA_MODE_PATTERN)
    profile: str = Field(..., description="Voice profile name or id the agent speaks in.")
    language: str = Field(default="en", pattern=_VA_LANGUAGE_PATTERN)
    agent_name: str = Field(..., min_length=1, max_length=100)
    company_name: str = Field(..., min_length=1, max_length=200)
    brief: str = Field(
        ...,
        min_length=1,
        max_length=6000,
        description="Facts the agent may state: the offer (sales) or the service scope (service/support).",
    )
    goal: str = Field(..., min_length=1, max_length=1000, description="What a successful call achieves.")
    disclosure: str = Field(default=DEFAULT_DISCLOSURE, min_length=1, max_length=500)


class VoiceAgentUpdate(BaseModel):
    """Body for ``PUT /agents/{id}`` — every field optional."""

    name: str | None = Field(None, min_length=1, max_length=100)
    mode: str | None = Field(None, pattern=_VA_MODE_PATTERN)
    profile: str | None = None
    engine: str | None = Field(None, pattern=_VA_ENGINE_PATTERN)
    language: str | None = Field(None, pattern=_VA_LANGUAGE_PATTERN)
    llm_model_size: str | None = Field(None, pattern="^(0\\.6B|1\\.7B|4B)$")
    voice_style: str | None = Field(None, max_length=200)
    empathetic_voice_style: str | None = Field(None, max_length=200)
    agent_name: str | None = Field(None, min_length=1, max_length=100)
    company_name: str | None = Field(None, min_length=1, max_length=200)
    brief: str | None = Field(None, min_length=1, max_length=6000)
    goal: str | None = Field(None, min_length=1, max_length=1000)
    objection_notes: str | None = Field(None, max_length=4000)
    persona: str | None = Field(None, max_length=2000)
    opening_line: str | None = Field(None, max_length=1000)
    disclosure: str | None = Field(None, min_length=1, max_length=500)
    escalation_promise: str | None = Field(None, max_length=500)
    variants: list[ScriptVariant] | None = Field(None, max_length=8)
    filler_phrases: list[str] | None = Field(None, max_length=6)
    fast_first_audio: bool | None = None
    tools_enabled: bool | None = None
    booking_instructions: str | None = Field(None, max_length=1000)
    appointment_duration_min: int | None = Field(None, ge=5, le=480)
    analysis_schema: list[AnalysisField] | None = Field(None, max_length=20)
    webhook_url: str | None = Field(None, max_length=500)
    webhook_secret: str | None = Field(None, max_length=200)
    timezone: str | None = Field(None, max_length=64)
    calling_window_start: int | None = Field(None, ge=0, le=23)
    calling_window_end: int | None = Field(None, ge=1, le=24)
    calling_days: list[int] | None = None
    max_attempts: int | None = Field(None, ge=1, le=20)
    daily_call_cap: int | None = Field(None, ge=1, le=100000)
    retry_delay_hours: int | None = Field(None, ge=1, le=720)
    callback_delay_hours: int | None = Field(None, ge=1, le=720)
    require_consent: bool | None = None
    max_turns: int | None = Field(None, ge=2, le=200)
    handoff_after_negative_turns: int | None = Field(None, ge=1, le=20)
    redact_pii: bool | None = None
    max_concurrent_calls: int | None = Field(None, ge=1, le=20)
    schedule_start_at: datetime | None = None
    schedule_end_at: datetime | None = None
    provider: str | None = Field(None, pattern="^(local|twilio)$")
    from_number: str | None = Field(None, max_length=32)
    transfer_number: str | None = Field(None, max_length=32)
    voicemail_message: str | None = Field(None, max_length=1000)
    sms_followup_template: str | None = Field(None, max_length=640)
    sms_followup_outcomes: list[str] | None = Field(None, max_length=13)

    @field_validator("webhook_url")
    @classmethod
    def _webhook_scheme(cls, url: str | None) -> str | None:
        if url and not url.lower().startswith(("http://", "https://")):
            raise ValueError("webhook_url must start with http:// or https://")
        return url


class VoiceAgentResponse(BaseModel):
    id: str
    name: str
    mode: str
    status: str
    version: int = 1
    profile_id: str
    engine: str | None = None
    language: str
    llm_model_size: str | None = None
    voice_style: str | None = None
    empathetic_voice_style: str | None = None
    agent_name: str
    company_name: str
    brief: str
    goal: str
    objection_notes: str | None = None
    persona: str | None = None
    opening_line: str | None = None
    disclosure: str
    escalation_promise: str | None = None
    variants: list[ScriptVariant] | None = None
    filler_phrases: list[str] = Field(default_factory=list)
    filler_audio: dict[str, str] | None = None
    fast_first_audio: bool = True
    tools_enabled: bool = True
    booking_instructions: str | None = None
    appointment_duration_min: int = 30
    analysis_schema: list[AnalysisField] | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None
    timezone: str
    calling_window_start: int
    calling_window_end: int
    calling_days: list[int]
    max_attempts: int
    daily_call_cap: int
    retry_delay_hours: int
    callback_delay_hours: int
    require_consent: bool
    max_turns: int
    handoff_after_negative_turns: int
    redact_pii: bool = True
    max_concurrent_calls: int = 1
    schedule_start_at: datetime | None = None
    schedule_end_at: datetime | None = None
    provider: str
    from_number: str | None = None
    transfer_number: str | None = None
    voicemail_message: str | None = None
    sms_followup_template: str | None = None
    sms_followup_outcomes: list[str] = Field(default_factory=list)
    running: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class VoiceAgentVersionResponse(BaseModel):
    id: str
    agent_id: str
    version: int
    note: str | None = None
    snapshot: dict
    created_at: datetime

    class Config:
        from_attributes = True


class VoiceAgentStats(BaseModel):
    agent_id: str
    mode: str
    status: str
    running: bool
    contacts_total: int
    contacts_by_status: dict[str, int]
    calls_total: int
    calls_today: int
    calls_by_outcome: dict[str, int]
    avg_turns: float
    resolution_rate: float
    open_tickets: int
    next_dialable: int
    appointments_upcoming: int = 0
    avg_score: float | None = None


class AnalyticsSeriesPoint(BaseModel):
    date: str
    calls: int
    goal: int
    by_outcome: dict[str, int]


class AnalyticsVariant(BaseModel):
    name: str
    calls: int
    goal: int
    goal_rate: float


class AnalyticsResponse(BaseModel):
    agent_id: str
    days: int
    series: list[AnalyticsSeriesPoint]
    funnel: dict[str, int]
    outcomes: dict[str, int]
    avg_turns: float
    avg_duration_s: float
    avg_sentiment: float | None = None
    avg_stt_ms: float | None = None
    avg_llm_ms: float | None = None
    avg_score: float | None = None
    variants: list[AnalyticsVariant] = Field(default_factory=list)
    analysis: dict[str, dict[str, int]] = Field(default_factory=dict)
    simulations: int = 0
    appointments: int = 0


class ContactCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    phone: str = Field(..., min_length=3, max_length=32)
    company: str | None = Field(None, max_length=200)
    notes: str | None = Field(None, max_length=2000)
    timezone: str | None = Field(None, max_length=64)
    language: str | None = Field(None, pattern=_VA_LANGUAGE_PATTERN)
    custom_fields: dict[str, str] | None = None
    consent: bool = False


class ContactBulkCreate(BaseModel):
    contacts: list[ContactCreate] = Field(..., min_length=1, max_length=10000)


class ContactUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    phone: str | None = Field(None, min_length=3, max_length=32)
    company: str | None = Field(None, max_length=200)
    notes: str | None = Field(None, max_length=2000)
    memory: str | None = Field(None, max_length=8000)
    timezone: str | None = Field(None, max_length=64)
    language: str | None = Field(None, pattern=_VA_LANGUAGE_PATTERN)
    custom_fields: dict[str, str] | None = None
    consent: bool | None = None
    status: str | None = Field(
        None,
        pattern="^(new|callback|contacted|interested|not_interested|resolved|unresolved|do_not_call|exhausted)$",
    )
    next_attempt_at: datetime | None = None


class ContactResponse(BaseModel):
    id: str
    agent_id: str
    name: str
    phone: str
    company: str | None = None
    notes: str | None = None
    memory: str | None = None
    timezone: str | None = None
    language: str | None = None
    custom_fields: dict[str, str] | None = None
    is_test: bool = False
    consent: bool
    status: str
    attempts: int
    last_attempt_at: datetime | None = None
    next_attempt_at: datetime | None = None
    last_outcome: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ContactImportResult(BaseModel):
    imported: int
    skipped: int
    skipped_reasons: dict[str, int]


class KnowledgeArticleCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=20000)
    tags: list[str] | None = None


class KnowledgeArticleUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=200)
    content: str | None = Field(None, min_length=1, max_length=20000)
    tags: list[str] | None = None


class KnowledgeArticleResponse(BaseModel):
    id: str
    agent_id: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    created_at: datetime
    updated_at: datetime


class KnowledgeImportUrlRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)
    tags: list[str] | None = None


class KnowledgeSearchResult(BaseModel):
    article: KnowledgeArticleResponse
    score: float


class ToolParam(BaseModel):
    name: str = Field(..., pattern=_VA_SNAKE_PATTERN)
    type: str = Field(default="string", pattern="^(string|number|boolean)$")
    description: str | None = Field(None, max_length=300)
    required: bool = True


class AgentToolCreate(BaseModel):
    name: str = Field(..., pattern=_VA_SNAKE_PATTERN)
    description: str = Field(..., min_length=1, max_length=1000)
    method: str = Field(default="GET", pattern="^(GET|POST|PUT|PATCH|DELETE)$")
    url: str = Field(..., min_length=8, max_length=2000)
    headers: dict[str, str] | None = None
    params: list[ToolParam] | None = Field(None, max_length=20)
    timeout_s: int = Field(default=10, ge=1, le=60)
    enabled: bool = True

    @field_validator("url")
    @classmethod
    def _url_scheme(cls, url: str) -> str:
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return url


class AgentToolUpdate(BaseModel):
    name: str | None = Field(None, pattern=_VA_SNAKE_PATTERN)
    description: str | None = Field(None, min_length=1, max_length=1000)
    method: str | None = Field(None, pattern="^(GET|POST|PUT|PATCH|DELETE)$")
    url: str | None = Field(None, min_length=8, max_length=2000)
    headers: dict[str, str] | None = None
    params: list[ToolParam] | None = Field(None, max_length=20)
    timeout_s: int | None = Field(None, ge=1, le=60)
    enabled: bool | None = None


class AgentToolResponse(BaseModel):
    id: str
    agent_id: str
    name: str
    description: str
    method: str
    url: str
    headers: dict[str, str] | None = None
    params: list[ToolParam] | None = None
    timeout_s: int
    enabled: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CallTurnResponse(BaseModel):
    id: str
    call_id: str
    role: str
    text: str
    source: str = "llm"
    sentiment: float | None = None
    interrupted: bool = False
    stt_ms: int | None = None
    llm_ms: int | None = None
    tool_name: str | None = None
    meta: dict | None = None
    generation_id: str | None = None
    generation_ids: list[str] | None = None
    capture_id: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class CallResponse(BaseModel):
    id: str
    agent_id: str
    contact_id: str
    direction: str
    status: str
    stage: str
    outcome: str | None = None
    summary: str | None = None
    variant: str | None = None
    ai_paused: bool = False
    analysis: dict | None = None
    score: int | None = None
    score_reason: str | None = None
    flags: list[str] | None = None
    webhook_status: str | None = None
    provider: str
    provider_call_id: str | None = None
    turn_count: int
    started_at: datetime
    ended_at: datetime | None = None
    turns: list[CallTurnResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class InboundCallRequest(BaseModel):
    """Start an inbound conversation (customer service / support)."""

    phone: str = Field(..., min_length=3, max_length=32)
    name: str | None = Field(None, max_length=200)


class CustomerTurnRequest(BaseModel):
    """What the customer said, as text. Audio goes to ``/turn/audio``."""

    text: str = Field(..., min_length=1, max_length=4000)


class OperatorSayRequest(BaseModel):
    """A supervisor speaking as the agent (take-over)."""

    text: str = Field(..., min_length=1, max_length=2000)


class InterruptRequest(BaseModel):
    turn_id: str | None = None


class AgentTurnResponse(BaseModel):
    """The agent's reply plus what it decided about the call."""

    call_id: str
    text: str
    generation_id: str | None = None
    generation_ids: list[str] = Field(default_factory=list)
    poll_url: str | None = None
    ended: bool
    outcome: str | None = None
    ticket_id: str | None = None
    appointment_id: str | None = None
    customer_text: str | None = None
    sentiment: float | None = None
    awaiting_operator: bool = False
    tool_calls: list[dict] = Field(default_factory=list)
    stt_ms: int | None = None
    llm_ms: int | None = None


class EndCallRequest(BaseModel):
    outcome: str = Field(..., pattern=_VA_OUTCOME_PATTERN)
    summary: str | None = Field(None, max_length=2000)


class SimulateRequest(BaseModel):
    """Run a scripted test call: the LLM plays a customer with this persona."""

    persona: str = Field(
        default="A busy but polite homeowner who is mildly sceptical and asks one practical question before deciding.",
        min_length=1,
        max_length=1000,
    )
    max_turns: int = Field(default=12, ge=2, le=40)
    variant: str | None = Field(None, max_length=60)


class AppointmentUpdate(BaseModel):
    status: str | None = Field(None, pattern="^(booked|confirmed|cancelled|completed)$")
    notes: str | None = Field(None, max_length=2000)
    starts_at: datetime | None = None
    ends_at: datetime | None = None


class AppointmentResponse(BaseModel):
    id: str
    agent_id: str
    contact_id: str
    call_id: str | None = None
    starts_at: datetime
    ends_at: datetime
    timezone: str | None = None
    notes: str | None = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: str
    agent_id: str
    contact_id: str
    call_id: str | None = None
    channel: str
    to_number: str
    body: str
    status: str
    provider_message_id: str | None = None
    error: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class WebhookDeliveryResponse(BaseModel):
    id: str
    agent_id: str
    call_id: str | None = None
    event: str
    url: str
    status: str
    attempts: int
    response_code: int | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TicketUpdate(BaseModel):
    status: str | None = Field(None, pattern="^(open|in_progress|resolved|closed)$")
    priority: str | None = Field(None, pattern="^(low|normal|high|urgent)$")
    description: str | None = Field(None, max_length=8000)


class TicketResponse(BaseModel):
    id: str
    agent_id: str
    contact_id: str
    call_id: str | None = None
    kind: str
    priority: str
    status: str
    subject: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DoNotCallCreate(BaseModel):
    phone: str = Field(..., min_length=3, max_length=32)
    reason: str | None = Field(None, max_length=500)


class DoNotCallImportResult(BaseModel):
    imported: int
    skipped: int


class DoNotCallResponse(BaseModel):
    phone: str
    reason: str | None = None
    source: str
    created_at: datetime

    class Config:
        from_attributes = True
