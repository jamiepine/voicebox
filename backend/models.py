"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from datetime import datetime


class VoiceProfileCreate(BaseModel):
    """Request model for creating a voice profile."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    language: str = Field(default="en", pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he)$")


class VoiceProfileResponse(BaseModel):
    """Response model for voice profile."""
    id: str
    name: str
    description: Optional[str]
    language: str
    avatar_path: Optional[str] = None
    has_finetune: bool = False
    has_active_adapter: bool = False
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _compute_has_active_adapter(cls, data):
        """Convert active_adapter_path from DB into a boolean for the API."""
        if hasattr(data, "__getattr__"):
            # SQLAlchemy model object
            path = getattr(data, "active_adapter_path", None)
            data_dict = {c.key: getattr(data, c.key) for c in data.__table__.columns}
            data_dict.pop("active_adapter_path", None)
            data_dict["has_active_adapter"] = bool(path)
            return data_dict
        if isinstance(data, dict):
            path = data.pop("active_adapter_path", None)
            data.setdefault("has_active_adapter", bool(path))
        return data

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
    text: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="en", pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he)$")
    seed: Optional[int] = Field(None, ge=0)
    model_size: Optional[str] = Field(default="1.7B", pattern="^(1\\.7B|0\\.6B)$")
    instruct: Optional[str] = Field(None, max_length=500)
    adapter_job_id: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        description="Specific adapter job ID to use. If not provided, uses the profile's active adapter.",
    )


class GenerationResponse(BaseModel):
    """Response model for voice generation."""
    id: str
    profile_id: str
    text: str
    language: str
    audio_path: str
    duration: float
    seed: Optional[int]
    instruct: Optional[str]
    created_at: datetime

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
    audio_path: str
    duration: float
    seed: Optional[int]
    instruct: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryListResponse(BaseModel):
    """Response model for history list."""
    items: List[HistoryResponse]
    total: int


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    language: Optional[str] = Field(None, pattern="^(zh|en|ja|ko|de|fr|ru|pt|es|it|he)$")


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


class ModelStatus(BaseModel):
    """Response model for model status."""
    model_name: str
    display_name: str
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


class ActiveDownloadTask(BaseModel):
    """Response model for active download task."""
    model_name: str
    status: str
    started_at: datetime


class ActiveGenerationTask(BaseModel):
    """Response model for active generation task."""
    task_id: str
    profile_id: str
    text_preview: str
    started_at: datetime


class ActiveFinetuneTask(BaseModel):
    """Response model for active finetune task."""
    job_id: str
    profile_id: str
    status: str
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    current_loss: Optional[float]
    started_at: datetime


class ActiveTasksResponse(BaseModel):
    """Response model for active tasks."""
    downloads: List[ActiveDownloadTask]
    generations: List[ActiveGenerationTask]
    finetunes: List[ActiveFinetuneTask] = []


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


# ============================================
# FINETUNE MODELS
# ============================================

class FinetuneSampleResponse(BaseModel):
    """Response model for a finetune training sample."""
    id: str
    profile_id: str
    transcript: str
    duration_seconds: float
    is_ref_audio: bool
    created_at: datetime

    class Config:
        from_attributes = True


class FinetuneJobResponse(BaseModel):
    """Response model for a finetune training job."""
    id: str
    profile_id: str
    status: str
    num_samples: int
    total_audio_duration_seconds: float
    epochs: int
    learning_rate: float
    batch_size: int
    lora_rank: int
    current_epoch: int
    current_step: int
    total_steps: int
    current_loss: Optional[float]
    has_adapter: bool = False
    label: Optional[str] = None
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    @model_validator(mode="before")
    @classmethod
    def _compute_has_adapter(cls, data):
        """Convert adapter_path from DB into a boolean for the API."""
        if hasattr(data, "__getattr__"):
            data_dict = {c.key: getattr(data, c.key) for c in data.__table__.columns}
            path = data_dict.pop("adapter_path", None)
            data_dict["has_adapter"] = bool(path)
            return data_dict
        if isinstance(data, dict):
            path = data.pop("adapter_path", None)
            data.setdefault("has_adapter", bool(path))
        return data

    class Config:
        from_attributes = True


class FinetuneStatusResponse(BaseModel):
    """Response model for finetune status overview."""
    has_finetune: bool
    has_active_adapter: bool = False
    active_job: Optional[FinetuneJobResponse]
    sample_count: int
    total_duration_seconds: float


class FinetuneStartRequest(BaseModel):
    """Request model for starting fine-tuning."""
    epochs: int = Field(default=3, ge=1, le=50)
    learning_rate: float = Field(default=2e-5, gt=0, le=1e-2)
    batch_size: int = Field(default=1, ge=1, le=8)
    lora_rank: int = Field(default=16, ge=4, le=128)
    label: Optional[str] = Field(default=None, max_length=100)


class FinetuneImportRequest(BaseModel):
    """Request model for importing profile samples into finetune."""
    sample_ids: Optional[List[str]] = None  # If None, import all


class AdapterInfo(BaseModel):
    """Response model for a trained adapter."""
    job_id: str
    label: Optional[str] = None
    epochs: int
    lora_rank: int
    learning_rate: float
    num_samples: int
    completed_at: Optional[datetime] = None
    is_active: bool = False


class SetActiveAdapterRequest(BaseModel):
    """Request model for setting the active adapter for a profile."""
    job_id: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )  # None = deactivate (use base model)


class UpdateAdapterLabelRequest(BaseModel):
    """Request model for updating an adapter's label."""
    label: str = Field(..., min_length=1, max_length=100)
