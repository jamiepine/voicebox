"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


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
    backend_variant: Optional[str] = None  # Binary variant (cpu or cuda)
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
