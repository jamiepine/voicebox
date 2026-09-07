"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

from .utils.capture_chords import (
    default_push_to_talk_chord,
    default_toggle_to_talk_chord,
)


class VoiceProfileCreate(BaseModel):
    """Request model for creating a voice profile."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    language: str = Field(
        default="en", pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$"
    )
    voice_type: Optional[str] = Field(default="cloned", pattern="^(cloned|preset|designed)$")
    preset_engine: Optional[str] = Field(None, max_length=50)
    preset_voice_id: Optional[str] = Field(None, max_length=100)
    design_prompt: Optional[str] = Field(None, max_length=2000)
    default_engine: Optional[str] = Field(None, max_length=50)
    personality: Optional[str] = Field(None, max_length=2000)


class VoiceProfileResponse(BaseModel):
    """Response model for voice profile."""

    id: str
    name: str
    description: Optional[str]
    language: str
    avatar_path: Optional[str] = None
    effects_chain: Optional[List["EffectConfig"]] = None
    voice_type: str = "cloned"
    preset_engine: Optional[str] = None
    preset_voice_id: Optional[str] = None
    design_prompt: Optional[str] = None
    default_engine: Optional[str] = None
    personality: Optional[str] = None
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
    language: str = Field(default="en", pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$")
    seed: Optional[int] = Field(None, ge=0)
    model_size: Optional[str] = Field(default="1.7B", pattern="^(1\\.7B|0\\.6B|1B|3B)$")
    instruct: Optional[str] = Field(None, max_length=500)
    engine: Optional[str] = Field(default="qwen", pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$")
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
    effects_chain: Optional[List["EffectConfig"]] = Field(
        None, description="Effects chain to apply after generation (overrides profile default)"
    )


class GenerationResponse(BaseModel):
    """Response model for voice generation."""

    id: str
    profile_id: str
    text: str
    language: str
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    seed: Optional[int] = None
    instruct: Optional[str] = None
    engine: Optional[str] = "qwen"
    model_size: Optional[str] = None
    status: str = "completed"
    error: Optional[str] = None
    is_favorited: bool = False
    source: str = "manual"
    created_at: datetime
    versions: Optional[List["GenerationVersionResponse"]] = None
    active_version_id: Optional[str] = None

    class Config:
        from_attributes = True


class HistoryQuery(BaseModel):
    """Query model for generation history."""

    profile_id: Optional[str] = None
    search: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class HistoryResponse(BaseModel):
    """Response model for history entry (includes profile name)."""

    id: str
    profile_id: str
    profile_name: str
    text: str
    language: str
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    seed: Optional[int] = None
    instruct: Optional[str] = None
    engine: Optional[str] = "qwen"
    model_size: Optional[str] = None
    status: str = "completed"
    error: Optional[str] = None
    is_favorited: bool = False
    created_at: datetime
    versions: Optional[List["GenerationVersionResponse"]] = None
    active_version_id: Optional[str] = None

    class Config:
        from_attributes = True


class HistoryListResponse(BaseModel):
    """Response model for history list."""

    items: List[HistoryResponse]
    total: int


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""

    language: Optional[str] = Field(None, pattern="^(en|zh|ja|ko|de|fr|ru|pt|es|it)$")
    model: Optional[str] = Field(None, pattern="^(base|small|medium|large|turbo)$")


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
    language: Optional[str] = None
    duration_ms: Optional[int] = None
    transcript_raw: str
    transcript_refined: Optional[str] = None
    stt_model: Optional[str] = None
    llm_model: Optional[str] = None
    refinement_flags: Optional[RefinementFlagsModel] = None
    created_at: datetime

    class Config:
        from_attributes = True


class CaptureListResponse(BaseModel):
    """Response model for paginated capture list."""

    items: List[CaptureResponse]
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

    flags: Optional[RefinementFlagsModel] = None
    model_size: Optional[str] = Field(default=None, pattern="^(0\\.6B|1\\.7B|4B)$")


class CaptureRetranscribeRequest(BaseModel):
    """Request to re-run STT on a capture's audio with a different model."""

    model: Optional[str] = Field(None, pattern="^(base|small|medium|large|turbo)$")
    language: Optional[str] = Field(None, pattern="^(en|zh|ja|ko|de|fr|ru|pt|es|it)$")


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
    default_playback_voice_id: Optional[str] = None
    hotkey_enabled: bool = False
    chord_push_to_talk_keys: List[str] = Field(
        default_factory=default_push_to_talk_chord
    )
    chord_toggle_to_talk_keys: List[str] = Field(
        default_factory=default_toggle_to_talk_chord
    )

    class Config:
        from_attributes = True


class CaptureSettingsUpdate(BaseModel):
    """Partial update for capture settings — every field is optional."""

    stt_model: Optional[str] = Field(default=None, pattern="^(base|small|medium|large|turbo)$")
    language: Optional[str] = None
    auto_refine: Optional[bool] = None
    llm_model: Optional[str] = Field(default=None, pattern="^(0\\.6B|1\\.7B|4B)$")
    smart_cleanup: Optional[bool] = None
    self_correction: Optional[bool] = None
    preserve_technical: Optional[bool] = None
    allow_auto_paste: Optional[bool] = None
    default_playback_voice_id: Optional[str] = None
    hotkey_enabled: Optional[bool] = None
    chord_push_to_talk_keys: Optional[List[str]] = Field(default=None, min_length=1, max_length=6)
    chord_toggle_to_talk_keys: Optional[List[str]] = Field(default=None, min_length=1, max_length=6)


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

    max_chunk_chars: Optional[int] = Field(default=None, ge=100, le=5000)
    crossfade_ms: Optional[int] = Field(default=None, ge=0, le=500)
    normalize_audio: Optional[bool] = None
    autoplay_on_generate: Optional[bool] = None


class MCPClientBindingResponse(BaseModel):
    """Per-MCP-client voice binding — what voice / engine the server should
    use when a given client_id calls voicebox.speak without args, plus an
    opt-in personality-rewrite default."""

    client_id: str
    label: Optional[str] = None
    profile_id: Optional[str] = None
    default_engine: Optional[str] = Field(
        None,
        pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$",
    )
    default_personality: bool = False
    last_seen_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MCPClientBindingUpsert(BaseModel):
    """Create or update a binding. Matched by ``client_id``."""

    client_id: str = Field(..., min_length=1, max_length=64)
    label: Optional[str] = Field(None, max_length=128)
    profile_id: Optional[str] = None
    default_engine: Optional[str] = Field(
        None,
        pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$",
    )
    default_personality: bool = False


class MCPClientBindingListResponse(BaseModel):
    items: List[MCPClientBindingResponse]


class SpeakRequest(BaseModel):
    """Body for POST /speak — non-MCP REST surface that mirrors voicebox.speak."""

    text: str = Field(..., min_length=1, max_length=10000)
    profile: Optional[str] = Field(
        None,
        description="Voice profile name or id. Falls back to per-client binding, then default.",
    )
    engine: Optional[str] = Field(
        None,
        pattern="^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$",
    )
    personality: Optional[bool] = Field(
        None,
        description="When true and the profile has a personality prompt, the input text is rewritten in-character before TTS. When null, the per-client binding's default_personality flag decides.",
    )
    language: Optional[str] = Field(
        None,
        pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$",
    )


class LLMGenerateRequest(BaseModel):
    """Request model for LLM text generation."""

    prompt: str = Field(..., min_length=1, max_length=50000)
    system: Optional[str] = Field(None, max_length=4000)
    model_size: Optional[str] = Field(default="0.6B", pattern="^(0\\.6B|1\\.7B|4B)$")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Few-shot (user, assistant) pairs prepended as real chat turns.
    # Used by the refinement service to pin tricky rules (imperatives
    # staying imperatives, technical-term punctuation) that small models
    # lose when the examples live inline in the system prompt.
    examples: Optional[List[List[str]]] = Field(default=None, max_length=8)


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
    size_mb: Optional[int] = None


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
    model_downloaded: Optional[bool] = None  # Whether model is cached/downloaded
    model_size: Optional[str] = None  # Current model size if loaded
    gpu_available: bool
    gpu_type: Optional[str] = None  # GPU type (CUDA, MPS, or None)
    vram_used_mb: Optional[float] = None
    backend_type: Optional[str] = None  # Backend type (mlx or pytorch)
    backend_variant: Optional[str] = None  # Binary variant (cpu, cuda, or rocm)
    supports_rocm: bool = False  # AMD GPU on Windows — the ROCm backend is applicable
    gpu_compatibility_warning: Optional[str] = None  # Warning if GPU arch unsupported


class DirectoryCheck(BaseModel):
    """Health status for a single directory."""

    path: str
    exists: bool
    writable: bool
    error: Optional[str] = None


class FilesystemHealthResponse(BaseModel):
    """Response model for filesystem health check."""

    healthy: bool
    disk_free_mb: Optional[float] = None
    disk_total_mb: Optional[float] = None
    directories: List[DirectoryCheck]


class ModelStatus(BaseModel):
    """Response model for model status."""

    model_name: str
    display_name: str
    hf_repo_id: Optional[str] = None  # HuggingFace repository ID
    downloaded: bool
    downloading: bool = False  # True if download is in progress
    size_mb: Optional[float] = None
    loaded: bool = False


class ModelStatusListResponse(BaseModel):
    """Response model for model status list."""

    models: List[ModelStatus]


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
    error: Optional[str] = None
    progress: Optional[float] = None  # 0-100 percentage
    current: Optional[int] = None  # bytes downloaded
    total: Optional[int] = None  # total bytes
    filename: Optional[str] = None  # current file being downloaded


class ActiveGenerationTask(BaseModel):
    """Response model for active generation task."""

    task_id: str
    profile_id: str
    text_preview: str
    started_at: datetime


class ActiveTasksResponse(BaseModel):
    """Response model for active tasks."""

    downloads: List[ActiveDownloadTask]
    generations: List[ActiveGenerationTask]


class AudioChannelCreate(BaseModel):
    """Request model for creating an audio channel."""

    name: str = Field(..., min_length=1, max_length=100)
    device_ids: List[str] = Field(default_factory=list)


class AudioChannelUpdate(BaseModel):
    """Request model for updating an audio channel."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    device_ids: Optional[List[str]] = None


class AudioChannelResponse(BaseModel):
    """Response model for audio channel."""

    id: str
    name: str
    is_default: bool
    device_ids: List[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ChannelVoiceAssignment(BaseModel):
    """Request model for assigning voices to a channel."""

    profile_ids: List[str]


class ProfileChannelAssignment(BaseModel):
    """Request model for assigning channels to a profile."""

    channel_ids: List[str]


class StoryCreate(BaseModel):
    """Request model for creating a story."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class StoryResponse(BaseModel):
    """Response model for story (list view)."""

    id: str
    name: str
    description: Optional[str]
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
    version_id: Optional[str] = None
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
    seed: Optional[int]
    instruct: Optional[str]
    engine: Optional[str] = None
    volume: float = 1.0
    generation_created_at: datetime
    # Versions available for this generation
    versions: Optional[List["GenerationVersionResponse"]] = None
    active_version_id: Optional[str] = None

    class Config:
        from_attributes = True


class StoryDetailResponse(BaseModel):
    """Response model for story with items."""

    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    items: List[StoryItemDetail] = []

    class Config:
        from_attributes = True


class StoryItemCreate(BaseModel):
    """Request model for adding a generation to a story."""

    generation_id: str
    start_time_ms: Optional[int] = None  # If not provided, will be calculated automatically
    track: Optional[int] = 0  # Track number (0 = main track)


class StoryItemUpdateTime(BaseModel):
    """Request model for updating a story item's timecode."""

    generation_id: str
    start_time_ms: int = Field(..., ge=0)


class StoryItemBatchUpdate(BaseModel):
    """Request model for batch updating story item timecodes."""

    updates: List[StoryItemUpdateTime]


class StoryItemReorder(BaseModel):
    """Request model for reordering story items."""

    generation_ids: List[str] = Field(..., min_length=1)


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

    version_id: Optional[str] = None  # null = use generation default


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

    effects: List[EffectConfig] = Field(default_factory=list)


class EffectPresetCreate(BaseModel):
    """Request model for creating an effect preset."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    effects_chain: List[EffectConfig]


class EffectPresetUpdate(BaseModel):
    """Request model for updating an effect preset."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    effects_chain: Optional[List[EffectConfig]] = None


class EffectPresetResponse(BaseModel):
    """Response model for effect preset."""

    id: str
    name: str
    description: Optional[str] = None
    effects_chain: List[EffectConfig]
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
    effects_chain: Optional[List[EffectConfig]] = None
    source_version_id: Optional[str] = None
    is_default: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ApplyEffectsRequest(BaseModel):
    """Request to apply effects to an existing generation."""

    effects_chain: List[EffectConfig]
    source_version_id: Optional[str] = Field(
        None, description="Version to use as source audio (defaults to clean/original)"
    )
    label: Optional[str] = Field(None, max_length=100, description="Label for this version (auto-generated if omitted)")
    set_as_default: bool = Field(default=True, description="Set this version as the default")


class ProfileEffectsUpdate(BaseModel):
    """Request to update the default effects chain on a profile."""

    effects_chain: Optional[List[EffectConfig]] = Field(None, description="Effects chain (null to remove)")


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

    effects: List[AvailableEffect]


# ─── Cloud (backup & sync) ──────────────────────────────────────────────


class CloudLoginStartResponse(BaseModel):
    """Returned when the desktop kicks off browser login. The backend has
    already opened the browser; the URL is included for fallback/debugging."""

    authorize_url: str


class CloudStatusResponse(BaseModel):
    """Current link between this device and a Voicebox Cloud account."""

    connected: bool
    device_name: Optional[str] = None
    account_user_id: Optional[str] = None
    key_prefix: Optional[str] = None
    connected_at: Optional[datetime] = None
    dashboard_url: str


# ── Voice AI agent ────────────────────────────────────────────────────

_VA_LANGUAGE_PATTERN = "^(zh|en|ja|ko|de|fr|ru|pt|es|it|he|ar|da|el|fi|hi|ms|nl|no|pl|sv|sw|tr)$"
_VA_ENGINE_PATTERN = "^(qwen|qwen_custom_voice|luxtts|chatterbox|chatterbox_turbo|tada|kokoro)$"
_VA_MODE_PATTERN = "^(outbound_sales|customer_service|support)$"
_VA_OUTCOME_PATTERN = (
    "^(interested|not_interested|callback|opt_out|resolved|unresolved|"
    "ticket_created|handoff|no_answer|voicemail|max_turns|error)$"
)

DEFAULT_DISCLOSURE = "Just so you know, I'm an automated AI assistant and this call may be recorded."


class VoiceAgentCreate(BaseModel):
    """Body for ``POST /agents``."""

    name: str = Field(..., min_length=1, max_length=100)
    mode: str = Field(default="outbound_sales", pattern=_VA_MODE_PATTERN)
    profile: str = Field(..., description="Voice profile name or id the agent speaks in.")
    engine: Optional[str] = Field(None, pattern=_VA_ENGINE_PATTERN)
    language: str = Field(default="en", pattern=_VA_LANGUAGE_PATTERN)
    llm_model_size: Optional[str] = Field(None, pattern="^(0\\.6B|1\\.7B|4B)$")
    agent_name: str = Field(..., min_length=1, max_length=100)
    company_name: str = Field(..., min_length=1, max_length=200)
    brief: str = Field(
        ...,
        min_length=1,
        max_length=6000,
        description="Facts the agent may state: the offer (sales) or the service scope (service/support).",
    )
    goal: str = Field(..., min_length=1, max_length=1000, description="What a successful call achieves.")
    objection_notes: Optional[str] = Field(None, max_length=4000)
    persona: Optional[str] = Field(None, max_length=2000, description="Tone / manner of speaking.")
    opening_line: Optional[str] = Field(None, max_length=1000)
    disclosure: str = Field(default=DEFAULT_DISCLOSURE, min_length=1, max_length=500)
    escalation_promise: Optional[str] = Field(None, max_length=500)
    timezone: str = Field(default="UTC", max_length=64)
    calling_window_start: int = Field(default=9, ge=0, le=23)
    calling_window_end: int = Field(default=20, ge=1, le=24)
    calling_days: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    max_attempts: int = Field(default=3, ge=1, le=20)
    daily_call_cap: int = Field(default=200, ge=1, le=100000)
    retry_delay_hours: int = Field(default=24, ge=1, le=720)
    callback_delay_hours: int = Field(default=24, ge=1, le=720)
    require_consent: bool = Field(
        default=False,
        description="When true, only contacts flagged consent=true are ever dialled.",
    )
    max_turns: int = Field(default=30, ge=2, le=200)
    handoff_after_negative_turns: int = Field(default=3, ge=1, le=20)
    provider: str = Field(default="local", pattern="^(local|twilio)$")
    from_number: Optional[str] = Field(None, max_length=32)


class VoiceAgentUpdate(BaseModel):
    """Body for ``PUT /agents/{id}`` — every field optional."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    mode: Optional[str] = Field(None, pattern=_VA_MODE_PATTERN)
    profile: Optional[str] = None
    engine: Optional[str] = Field(None, pattern=_VA_ENGINE_PATTERN)
    language: Optional[str] = Field(None, pattern=_VA_LANGUAGE_PATTERN)
    llm_model_size: Optional[str] = Field(None, pattern="^(0\\.6B|1\\.7B|4B)$")
    agent_name: Optional[str] = Field(None, min_length=1, max_length=100)
    company_name: Optional[str] = Field(None, min_length=1, max_length=200)
    brief: Optional[str] = Field(None, min_length=1, max_length=6000)
    goal: Optional[str] = Field(None, min_length=1, max_length=1000)
    objection_notes: Optional[str] = Field(None, max_length=4000)
    persona: Optional[str] = Field(None, max_length=2000)
    opening_line: Optional[str] = Field(None, max_length=1000)
    disclosure: Optional[str] = Field(None, min_length=1, max_length=500)
    escalation_promise: Optional[str] = Field(None, max_length=500)
    timezone: Optional[str] = Field(None, max_length=64)
    calling_window_start: Optional[int] = Field(None, ge=0, le=23)
    calling_window_end: Optional[int] = Field(None, ge=1, le=24)
    calling_days: Optional[List[int]] = None
    max_attempts: Optional[int] = Field(None, ge=1, le=20)
    daily_call_cap: Optional[int] = Field(None, ge=1, le=100000)
    retry_delay_hours: Optional[int] = Field(None, ge=1, le=720)
    callback_delay_hours: Optional[int] = Field(None, ge=1, le=720)
    require_consent: Optional[bool] = None
    max_turns: Optional[int] = Field(None, ge=2, le=200)
    handoff_after_negative_turns: Optional[int] = Field(None, ge=1, le=20)
    provider: Optional[str] = Field(None, pattern="^(local|twilio)$")
    from_number: Optional[str] = Field(None, max_length=32)


class VoiceAgentResponse(BaseModel):
    id: str
    name: str
    mode: str
    status: str
    profile_id: str
    engine: Optional[str] = None
    language: str
    llm_model_size: Optional[str] = None
    agent_name: str
    company_name: str
    brief: str
    goal: str
    objection_notes: Optional[str] = None
    persona: Optional[str] = None
    opening_line: Optional[str] = None
    disclosure: str
    escalation_promise: Optional[str] = None
    timezone: str
    calling_window_start: int
    calling_window_end: int
    calling_days: List[int]
    max_attempts: int
    daily_call_cap: int
    retry_delay_hours: int
    callback_delay_hours: int
    require_consent: bool
    max_turns: int
    handoff_after_negative_turns: int
    provider: str
    from_number: Optional[str] = None
    running: bool = False
    created_at: datetime
    updated_at: datetime

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


class ContactCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    phone: str = Field(..., min_length=3, max_length=32)
    company: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)
    timezone: Optional[str] = Field(None, max_length=64)
    consent: bool = False


class ContactBulkCreate(BaseModel):
    contacts: List[ContactCreate] = Field(..., min_length=1, max_length=10000)


class ContactUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    phone: Optional[str] = Field(None, min_length=3, max_length=32)
    company: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)
    memory: Optional[str] = Field(None, max_length=8000)
    timezone: Optional[str] = Field(None, max_length=64)
    consent: Optional[bool] = None
    status: Optional[str] = Field(
        None,
        pattern="^(new|callback|contacted|interested|not_interested|resolved|unresolved|do_not_call|exhausted)$",
    )
    next_attempt_at: Optional[datetime] = None


class ContactResponse(BaseModel):
    id: str
    agent_id: str
    name: str
    phone: str
    company: Optional[str] = None
    notes: Optional[str] = None
    memory: Optional[str] = None
    timezone: Optional[str] = None
    consent: bool
    status: str
    attempts: int
    last_attempt_at: Optional[datetime] = None
    next_attempt_at: Optional[datetime] = None
    last_outcome: Optional[str] = None
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
    tags: Optional[List[str]] = None


class KnowledgeArticleUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1, max_length=20000)
    tags: Optional[List[str]] = None


class KnowledgeArticleResponse(BaseModel):
    id: str
    agent_id: str
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CallTurnResponse(BaseModel):
    id: str
    call_id: str
    role: str
    text: str
    sentiment: Optional[float] = None
    generation_id: Optional[str] = None
    capture_id: Optional[str] = None
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
    outcome: Optional[str] = None
    summary: Optional[str] = None
    provider: str
    provider_call_id: Optional[str] = None
    turn_count: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    turns: List[CallTurnResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class InboundCallRequest(BaseModel):
    """Start an inbound conversation (customer service / support)."""

    phone: str = Field(..., min_length=3, max_length=32)
    name: Optional[str] = Field(None, max_length=200)


class CustomerTurnRequest(BaseModel):
    """What the customer said, as text. Audio goes to ``/turn/audio``."""

    text: str = Field(..., min_length=1, max_length=4000)


class AgentTurnResponse(BaseModel):
    """The agent's reply plus what it decided about the call."""

    call_id: str
    text: str
    generation_id: Optional[str] = None
    poll_url: Optional[str] = None
    ended: bool
    outcome: Optional[str] = None
    ticket_id: Optional[str] = None
    customer_text: Optional[str] = None
    sentiment: Optional[float] = None


class EndCallRequest(BaseModel):
    outcome: str = Field(..., pattern=_VA_OUTCOME_PATTERN)
    summary: Optional[str] = Field(None, max_length=2000)


class TicketUpdate(BaseModel):
    status: Optional[str] = Field(None, pattern="^(open|in_progress|resolved|closed)$")
    priority: Optional[str] = Field(None, pattern="^(low|normal|high|urgent)$")
    description: Optional[str] = Field(None, max_length=8000)


class TicketResponse(BaseModel):
    id: str
    agent_id: str
    contact_id: str
    call_id: Optional[str] = None
    kind: str
    priority: str
    status: str
    subject: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DoNotCallCreate(BaseModel):
    phone: str = Field(..., min_length=3, max_length=32)
    reason: Optional[str] = Field(None, max_length=500)


class DoNotCallResponse(BaseModel):
    phone: str
    reason: Optional[str] = None
    source: str
    created_at: datetime

    class Config:
        from_attributes = True
