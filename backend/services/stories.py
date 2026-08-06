"""
Story management module.
"""

from typing import List, Optional
from datetime import datetime
import logging
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import func

from .. import config
from ..models import (
    StoryCreate,
    StoryResponse,
    StoryDetailResponse,
    StoryItemDetail,
    StoryItemCreate,
    StoryItemBatchUpdate,
    StoryItemMove,
    StoryItemTrim,
    StoryItemVolumeUpdate,
    StoryItemFadeUpdate,
    StoryItemSpeedUpdate,
    StoryItemSplit,
    StoryItemVersionUpdate,
    StoryTrackResponse,
    StoryTrackUpsert,
)
from ..database import (
    Story as DBStory,
    StoryItem as DBStoryItem,
    Generation as DBGeneration,
    VoiceProfile as DBVoiceProfile,
)
from ..database.models import StoryTrack as DBStoryTrack
from .history import _get_versions_for_generation
from ..utils.audio import encode_audio
import librosa
import numpy as np

# Mixdown never exceeds this even if a source is higher — 48 kHz is the
# practical ceiling for delivery, and resampling a 96 kHz bed up there costs
# memory for no audible gain.
MAX_PROJECT_SAMPLE_RATE = 48000

# Used when a story's sources give us nothing to go on (all unreadable).
FALLBACK_SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)


def _build_item_detail(
    item: DBStoryItem,
    generation: DBGeneration,
    profile_name: str,
    db: Session,
) -> StoryItemDetail:
    """Build a StoryItemDetail with version info from a story item and its generation."""
    versions, active_version_id = _get_versions_for_generation(generation.id, db)

    # Resolve the audio path: if version_id is set, use that version's audio
    audio_path = generation.audio_path
    if item.version_id and versions:
        for v in versions:
            if v.id == item.version_id:
                audio_path = v.audio_path
                break

    return StoryItemDetail(
        id=item.id,
        story_id=item.story_id,
        generation_id=item.generation_id,
        version_id=getattr(item, "version_id", None),
        start_time_ms=item.start_time_ms,
        track=item.track,
        trim_start_ms=getattr(item, "trim_start_ms", 0),
        trim_end_ms=getattr(item, "trim_end_ms", 0),
        created_at=item.created_at,
        profile_id=generation.profile_id,
        profile_name=profile_name,
        text=generation.text,
        language=generation.language,
        audio_path=audio_path,
        duration=generation.duration,
        seed=generation.seed,
        instruct=generation.instruct,
        engine=generation.engine,
        volume=getattr(item, "volume", 1.0),
        fade_in_ms=getattr(item, "fade_in_ms", 0) or 0,
        fade_out_ms=getattr(item, "fade_out_ms", 0) or 0,
        speed=getattr(item, "speed", 1.0) or 1.0,
        generation_created_at=generation.created_at,
        versions=versions,
        active_version_id=active_version_id,
    )


async def create_story(
    data: StoryCreate,
    db: Session,
) -> StoryResponse:
    """
    Create a new story.

    Args:
        data: Story creation data
        db: Database session

    Returns:
        Created story
    """
    db_story = DBStory(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(db_story)
    db.commit()
    db.refresh(db_story)

    item_count = db.query(func.count(DBStoryItem.id)).filter(DBStoryItem.story_id == db_story.id).scalar()

    response = StoryResponse.model_validate(db_story)
    response.item_count = item_count
    return response


async def list_stories(
    db: Session,
) -> List[StoryResponse]:
    """
    List all stories.

    Args:
        db: Database session

    Returns:
        List of stories with item counts
    """
    stories = db.query(DBStory).order_by(DBStory.updated_at.desc()).all()

    if not stories:
        return []

    # Batch-fetch all story item counts in one query to avoid an N+1 pattern
    # (previously there was one COUNT query per story in the loop below).
    story_ids = [s.id for s in stories]
    count_rows = (
        db.query(DBStoryItem.story_id, func.count(DBStoryItem.id).label("cnt"))
        .filter(DBStoryItem.story_id.in_(story_ids))
        .group_by(DBStoryItem.story_id)
        .all()
    )
    item_counts = {row.story_id: row.cnt for row in count_rows}

    result = []
    for story in stories:
        response = StoryResponse.model_validate(story)
        response.item_count = item_counts.get(story.id, 0)
        result.append(response)

    return result


async def get_story(
    story_id: str,
    db: Session,
) -> Optional[StoryDetailResponse]:
    """
    Get a story with all its items.

    Args:
        story_id: Story ID
        db: Database session

    Returns:
        Story with items or None if not found
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    items = (
        db.query(DBStoryItem, DBGeneration, DBVoiceProfile.name.label("profile_name"))
        .join(DBGeneration, DBStoryItem.generation_id == DBGeneration.id)
        .join(DBVoiceProfile, DBGeneration.profile_id == DBVoiceProfile.id)
        .filter(DBStoryItem.story_id == story_id)
        .order_by(DBStoryItem.start_time_ms)
        .all()
    )

    item_details = []
    for item, generation, profile_name in items:
        item_details.append(_build_item_detail(item, generation, profile_name, db))

    response = StoryDetailResponse.model_validate(story)
    response.items = item_details
    return response


async def update_story(
    story_id: str,
    data: StoryCreate,
    db: Session,
) -> Optional[StoryResponse]:
    """
    Update a story.

    Args:
        story_id: Story ID
        data: Update data
        db: Database session

    Returns:
        Updated story or None if not found
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    story.name = data.name
    story.description = data.description
    story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(story)

    item_count = db.query(func.count(DBStoryItem.id)).filter(DBStoryItem.story_id == story.id).scalar()

    response = StoryResponse.model_validate(story)
    response.item_count = item_count
    return response


async def delete_story(
    story_id: str,
    db: Session,
) -> bool:
    """
    Delete a story and all its items.

    Args:
        story_id: Story ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return False

    # Delete all items
    db.query(DBStoryItem).filter_by(story_id=story_id).delete()

    # Delete per-lane mixer settings. They are keyed by story_id but have no
    # FK cascade, so without this they outlive the story as unreachable rows.
    db.query(DBStoryTrack).filter_by(story_id=story_id).delete()

    # Delete story
    db.delete(story)
    db.commit()

    return True


async def add_item_to_story(
    story_id: str,
    data: StoryItemCreate,
    db: Session,
) -> Optional[StoryItemDetail]:
    """
    Add a generation to a story.

    Args:
        story_id: Story ID
        data: Item creation data
        db: Database session

    Returns:
        Created item detail or None if story/generation not found
    """
    # Verify story exists
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    # Verify generation exists
    generation = db.query(DBGeneration).filter_by(id=data.generation_id).first()
    if not generation:
        return None

    # Check if generation is already in story
    existing = db.query(DBStoryItem).filter_by(story_id=story_id, generation_id=data.generation_id).first()
    if existing:
        # Return existing item
        profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
        return _build_item_detail(existing, generation, profile.name if profile else "Unknown", db)

    # Imported audio is a bed, not another line of dialogue: default it to its
    # own empty lane starting at zero so it plays *under* the narration.
    # Appending it to track 0 like a TTS clip put the music after the voice,
    # which is never what someone dropping in a music file wants.
    profile_for_default = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
    is_imported = getattr(profile_for_default, "voice_type", None) == "import"

    if data.track is not None:
        track = data.track
    elif is_imported:
        track = _next_free_track(story_id, db)
    else:
        track = 0

    # Calculate start_time_ms if not provided
    if data.start_time_ms is not None:
        start_time_ms = data.start_time_ms
    elif is_imported:
        start_time_ms = 0
    else:
        existing_items = (
            db.query(DBStoryItem, DBGeneration)
            .join(DBGeneration, DBStoryItem.generation_id == DBGeneration.id)
            .filter(
                DBStoryItem.story_id == story_id,
                DBStoryItem.track == track,
            )
            .all()
        )

        if not existing_items:
            start_time_ms = 0
        else:
            max_end_time_ms = 0
            for item, gen in existing_items:
                item_end_ms = item.start_time_ms + int(gen.duration * 1000)
                max_end_time_ms = max(max_end_time_ms, item_end_ms)

            # Add 200ms gap after the last item
            start_time_ms = max_end_time_ms + 200

    # Create item
    item = DBStoryItem(
        id=str(uuid.uuid4()),
        story_id=story_id,
        generation_id=data.generation_id,
        start_time_ms=start_time_ms,
        track=track,
        created_at=datetime.utcnow(),
    )

    db.add(item)

    # Update story updated_at
    story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    # Get profile name
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


async def move_story_item(
    story_id: str,
    item_id: str,
    data: StoryItemMove,
    db: Session,
) -> Optional[StoryItemDetail]:
    """
    Move a story item (update position and/or track).

    Args:
        story_id: Story ID
        item_id: Story item ID
        data: New position and track data
        db: Database session

    Returns:
        Updated item detail or None if not found
    """
    # Get the item
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not item:
        return None

    # Get the generation
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    # Update position and track
    item.start_time_ms = data.start_time_ms
    item.track = data.track

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    # Get profile name
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


async def remove_item_from_story(
    story_id: str,
    item_id: str,
    db: Session,
) -> bool:
    """
    Remove a story item from a story.

    Args:
        story_id: Story ID
        item_id: Story item ID to remove
        db: Database session

    Returns:
        True if removed, False if not found
    """
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not item:
        return False

    # Delete item
    db.delete(item)

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    return True


async def trim_story_item(
    story_id: str,
    item_id: str,
    data: StoryItemTrim,
    db: Session,
) -> Optional[StoryItemDetail]:
    """
    Trim a story item (update trim_start_ms and trim_end_ms).

    Args:
        story_id: Story ID
        item_id: Story item ID
        data: Trim data (trim_start_ms, trim_end_ms)
        db: Database session

    Returns:
        Updated item detail or None if not found
    """
    # Get the item
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not item:
        return None

    # Get the generation
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    # Validate trim values don't exceed duration
    max_duration_ms = int(generation.duration * 1000)
    if data.trim_start_ms + data.trim_end_ms >= max_duration_ms:
        return None  # Invalid trim - would result in zero or negative duration

    # Update trim values
    item.trim_start_ms = data.trim_start_ms
    item.trim_end_ms = data.trim_end_ms

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    # Get profile name
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


async def update_story_item_volume(
    story_id: str,
    item_id: str,
    data: StoryItemVolumeUpdate,
    db: Session,
) -> Optional[StoryItemDetail]:
    """Update a story item's playback volume (per-clip linear gain)."""
    item = (
        db.query(DBStoryItem)
        .filter_by(id=item_id, story_id=story_id)
        .first()
    )
    if not item:
        return None
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    item.volume = data.volume

    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


def _next_free_track(story_id: str, db: Session) -> int:
    """Lowest lane index at or above 0 holding no clips.

    Lanes are sparse integers rather than a dense list, and negative indices
    are legitimate (the editor shows [1, 0, -1] by default), so this scans
    upward from 0 rather than taking a max.
    """
    used = {row[0] for row in db.query(DBStoryItem.track).filter_by(story_id=story_id).distinct()}
    index = 0
    while index in used:
        index += 1
    return index


async def _update_story_item_fields(
    story_id: str,
    item_id: str,
    db: Session,
    **fields,
) -> Optional[StoryItemDetail]:
    """Set fields on a story item and return the refreshed detail.

    Shared by the fade and speed endpoints, which differ only in what they
    assign — the lookup, story timestamp bump and detail rebuild are identical.
    """
    item = db.query(DBStoryItem).filter_by(id=item_id, story_id=story_id).first()
    if not item:
        return None
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    for key, value in fields.items():
        setattr(item, key, value)

    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


async def update_story_item_fades(
    story_id: str,
    item_id: str,
    data: StoryItemFadeUpdate,
    db: Session,
) -> Optional[StoryItemDetail]:
    """Set a story item's fade in/out lengths."""
    return await _update_story_item_fields(
        story_id,
        item_id,
        db,
        fade_in_ms=data.fade_in_ms,
        fade_out_ms=data.fade_out_ms,
    )


async def update_story_item_speed(
    story_id: str,
    item_id: str,
    data: StoryItemSpeedUpdate,
    db: Session,
) -> Optional[StoryItemDetail]:
    """Set a story item's playback rate."""
    return await _update_story_item_fields(story_id, item_id, db, speed=data.speed)


# ── Track mixer settings ─────────────────────────────────────────────


async def list_story_tracks(story_id: str, db: Session) -> List[StoryTrackResponse]:
    """Mixer settings for every lane that has them.

    Lanes without a row simply mix at unity gain, so the list is often
    shorter than the number of lanes on screen.
    """
    rows = (
        db.query(DBStoryTrack)
        .filter_by(story_id=story_id)
        .order_by(DBStoryTrack.index)
        .all()
    )
    return [StoryTrackResponse.model_validate(r) for r in rows]


async def upsert_story_track(
    story_id: str,
    index: int,
    data: StoryTrackUpsert,
    db: Session,
) -> Optional[StoryTrackResponse]:
    """Create or update one lane's mixer settings."""
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    row = db.query(DBStoryTrack).filter_by(story_id=story_id, index=index).first()
    if row is None:
        row = DBStoryTrack(story_id=story_id, index=index)
        db.add(row)

    row.name = data.name
    row.volume = data.volume
    row.muted = data.muted
    row.soloed = data.soloed
    row.duck_under_track = data.duck_under_track
    row.updated_at = datetime.utcnow()

    story.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return StoryTrackResponse.model_validate(row)


async def delete_story_track(story_id: str, index: int, db: Session) -> bool:
    """Reset a lane to defaults.

    Only the settings row goes — clips on that lane are untouched, and the
    lane keeps rendering at unity gain.
    """
    row = db.query(DBStoryTrack).filter_by(story_id=story_id, index=index).first()
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


async def split_story_item(
    story_id: str,
    item_id: str,
    data: StoryItemSplit,
    db: Session,
) -> Optional[List[StoryItemDetail]]:
    """
    Split a story item at a given time, creating two clips.

    Args:
        story_id: Story ID
        item_id: Story item ID to split
        data: Split data (split_time_ms - time within clip to split at)
        db: Database session

    Returns:
        List of two updated item details (original and new) or None if not found/invalid
    """
    # Get the item with a row lock to prevent concurrent splits on the
    # same clip (e.g. from rapid double-clicks racing each other).
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .with_for_update()
        .first()
    )
    if not item:
        return None

    # Get the generation
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    # Calculate effective duration and validate split point
    current_trim_start = getattr(item, "trim_start_ms", 0)
    current_trim_end = getattr(item, "trim_end_ms", 0)
    original_duration_ms = int(generation.duration * 1000)
    effective_duration_ms = original_duration_ms - current_trim_start - current_trim_end

    # Validate split_time_ms is within the effective duration
    if data.split_time_ms <= 0 or data.split_time_ms >= effective_duration_ms:
        return None  # Invalid split point

    # Calculate the absolute time in the original audio where we're splitting
    absolute_split_ms = current_trim_start + data.split_time_ms

    # Update original clip: trim from the end
    item.trim_end_ms = original_duration_ms - absolute_split_ms

    # Fades split with the audio: the head keeps its fade-in, the tail keeps
    # the fade-out. Leaving both on both halves would insert an audible dip at
    # the seam of what the user hears as one continuous clip.
    tail_fade_out = getattr(item, "fade_out_ms", 0) or 0
    item.fade_out_ms = 0

    # Create new clip: starts after the split, trimmed from the start
    new_item = DBStoryItem(
        id=str(uuid.uuid4()),
        story_id=story_id,
        generation_id=item.generation_id,  # Same generation, different trim
        version_id=getattr(item, "version_id", None),  # Preserve pinned version
        start_time_ms=item.start_time_ms + data.split_time_ms,
        track=item.track,
        trim_start_ms=absolute_split_ms,
        trim_end_ms=current_trim_end,
        volume=getattr(item, "volume", 1.0),
        fade_in_ms=0,
        fade_out_ms=tail_fade_out,
        speed=getattr(item, "speed", 1.0) or 1.0,
        created_at=datetime.utcnow(),
    )

    db.add(new_item)

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)
    db.refresh(new_item)

    # Get profile name
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
    profile_name = profile.name if profile else "Unknown"

    return [
        _build_item_detail(item, generation, profile_name, db),
        _build_item_detail(new_item, generation, profile_name, db),
    ]


async def duplicate_story_item(
    story_id: str,
    item_id: str,
    db: Session,
) -> Optional[StoryItemDetail]:
    """
    Duplicate a story item, creating a copy with all properties.

    Args:
        story_id: Story ID
        item_id: Story item ID to duplicate
        db: Database session

    Returns:
        New item detail or None if not found
    """
    # Get the original item
    original_item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not original_item:
        return None

    # Get the generation
    generation = db.query(DBGeneration).filter_by(id=original_item.generation_id).first()
    if not generation:
        return None

    # Calculate effective duration
    current_trim_start = getattr(original_item, "trim_start_ms", 0)
    current_trim_end = getattr(original_item, "trim_end_ms", 0)
    original_duration_ms = int(generation.duration * 1000)
    effective_duration_ms = original_duration_ms - current_trim_start - current_trim_end

    # Create duplicate item - place it right after the original
    new_item = DBStoryItem(
        id=str(uuid.uuid4()),
        story_id=story_id,
        generation_id=original_item.generation_id,  # Same generation as original
        version_id=getattr(original_item, "version_id", None),  # Preserve pinned version
        start_time_ms=original_item.start_time_ms + effective_duration_ms + 200,  # 200ms gap
        track=original_item.track,
        trim_start_ms=current_trim_start,
        trim_end_ms=current_trim_end,
        volume=getattr(original_item, "volume", 1.0),
        created_at=datetime.utcnow(),
    )

    db.add(new_item)

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(new_item)

    # Get profile name
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(new_item, generation, profile.name if profile else "Unknown", db)


async def update_story_item_times(
    story_id: str,
    data: StoryItemBatchUpdate,
    db: Session,
) -> bool:
    """
    Update story item timecodes.

    Args:
        story_id: Story ID
        data: Batch update data with timecodes
        db: Database session

    Returns:
        True if updated, False if story not found or invalid
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return False

    # Get all items for this story
    items = db.query(DBStoryItem).filter_by(story_id=story_id).all()
    item_map = {item.generation_id: item for item in items}

    # Verify all generation IDs belong to this story and update timecodes
    for update in data.updates:
        if update.generation_id not in item_map:
            return False
        item_map[update.generation_id].start_time_ms = update.start_time_ms

    # Update story updated_at
    story.updated_at = datetime.utcnow()

    db.commit()
    return True


async def reorder_story_items(
    story_id: str,
    generation_ids: List[str],
    db: Session,
    gap_ms: int = 200,
) -> Optional[List[StoryItemDetail]]:
    """
    Reorder story items and recalculate timecodes.

    Args:
        story_id: Story ID
        generation_ids: List of generation IDs in the desired order
        db: Database session
        gap_ms: Gap in milliseconds between items (default 200ms)

    Returns:
        Updated list of story items with new timecodes, or None if invalid
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    # Get all items for this story with their generation data
    items_with_gen = (
        db.query(DBStoryItem, DBGeneration, DBVoiceProfile.name.label("profile_name"))
        .join(DBGeneration, DBStoryItem.generation_id == DBGeneration.id)
        .join(DBVoiceProfile, DBGeneration.profile_id == DBVoiceProfile.id)
        .filter(DBStoryItem.story_id == story_id)
        .all()
    )

    # Create maps for quick lookup
    item_map = {item.generation_id: (item, gen, profile_name) for item, gen, profile_name in items_with_gen}

    # Verify all generation IDs belong to this story
    if set(generation_ids) != set(item_map.keys()):
        return None

    # Recalculate timecodes based on new order
    current_time_ms = 0
    updated_items = []

    for gen_id in generation_ids:
        item, generation, profile_name = item_map[gen_id]

        # Update the item's start time
        item.start_time_ms = current_time_ms

        # Calculate the duration in ms
        duration_ms = int(generation.duration * 1000)

        # Move to next position (current end + gap)
        current_time_ms += duration_ms + gap_ms

        # Build the response item
        updated_items.append(_build_item_detail(item, generation, profile_name, db))

    # Update story updated_at
    story.updated_at = datetime.utcnow()

    db.commit()
    return updated_items


async def set_story_item_version(
    story_id: str,
    item_id: str,
    data: StoryItemVersionUpdate,
    db: Session,
) -> Optional[StoryItemDetail]:
    """
    Pin a story item to a specific generation version.

    Args:
        story_id: Story ID
        item_id: Story item ID
        data: Version update data (version_id or null for default)
        db: Database session

    Returns:
        Updated item detail or None if not found
    """
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not item:
        return None

    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    # Validate version_id belongs to this generation if provided
    if data.version_id:
        from ..database import GenerationVersion as DBGenerationVersion

        version = (
            db.query(DBGenerationVersion)
            .filter_by(
                id=data.version_id,
                generation_id=item.generation_id,
            )
            .first()
        )
        if not version:
            return None

    item.version_id = data.version_id

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    """Normalise any loaded clip to a ``(2, samples)`` float32 array.

    librosa hands back ``(samples,)`` for mono and ``(channels, samples)``
    otherwise. Mono is duplicated rather than panned so a voice clip sits
    centred; anything above stereo is folded down to the first two channels.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return np.stack([audio, audio])
    if audio.shape[0] == 1:
        return np.repeat(audio, 2, axis=0)
    return audio[:2]


def _apply_fades(audio: np.ndarray, sample_rate: int, fade_in_ms: int, fade_out_ms: int) -> np.ndarray:
    """Apply linear fades to a ``(channels, samples)`` clip, in place-safe form.

    The two fades are scaled down together if they would overlap, so a short
    clip with long fades still ends up monotonic rather than re-brightening in
    the middle.
    """
    n = audio.shape[1]
    if n == 0 or (fade_in_ms <= 0 and fade_out_ms <= 0):
        return audio

    fade_in = int(sample_rate * max(fade_in_ms, 0) / 1000)
    fade_out = int(sample_rate * max(fade_out_ms, 0) / 1000)

    total = fade_in + fade_out
    if total > n and total > 0:
        scale = n / total
        fade_in = int(fade_in * scale)
        fade_out = int(fade_out * scale)

    audio = audio.copy()
    if fade_in > 0:
        audio[:, :fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 0:
        audio[:, n - fade_out :] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return audio


def _duck_envelope(
    source: np.ndarray,
    sample_rate: int,
    depth: float = 0.75,
    attack_ms: int = 80,
    release_ms: int = 400,
) -> np.ndarray:
    """Gain curve that pulls a bed down while ``source`` is loud.

    A plain RMS follower with asymmetric smoothing: duck quickly when speech
    starts, recover slowly so the bed doesn't pump between words.
    """
    mono = source.mean(axis=0)
    frame = max(1, sample_rate // 100)  # 10 ms

    padded = np.pad(mono, (0, (-len(mono)) % frame))
    rms = np.sqrt((padded.reshape(-1, frame) ** 2).mean(axis=1))

    peak = rms.max()
    if peak <= 1e-6:
        return np.ones(source.shape[1], dtype=np.float32)

    # 0 where silent, 1 where at peak, then invert into a gain reduction.
    activity = np.clip(rms / peak, 0.0, 1.0)
    gain = 1.0 - depth * activity

    attack = max(1, int(attack_ms / 10))
    release = max(1, int(release_ms / 10))
    smoothed = np.empty_like(gain)
    current = 1.0
    for i, target in enumerate(gain):
        coeff = 1.0 / (attack if target < current else release)
        current += (target - current) * coeff
        smoothed[i] = current

    envelope = np.repeat(smoothed, frame)[: source.shape[1]]
    return envelope.astype(np.float32)


async def export_story_audio(
    story_id: str,
    db: Session,
    fmt: str = "wav",
) -> Optional[bytes]:
    """
    Export story as single mixed audio file with timecode-based mixing.

    Mixes in stereo at the highest sample rate any source actually uses
    (capped at 48 kHz) rather than flattening everything to 24 kHz mono, so an
    imported music bed keeps its bandwidth and stereo image.

    Each lane is rendered to its own buffer first. That is what makes ducking
    possible — a bed can be attenuated by the *finished* speech lane — and it
    is also where track volume, mute and solo apply.

    Args:
        story_id: Story ID
        db: Database session
        fmt: Output container; see ``utils.audio.EXPORT_FORMATS``.

    Returns:
        Audio file bytes or None if story not found
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    # Get all items ordered by start_time_ms
    items = (
        db.query(DBStoryItem, DBGeneration)
        .join(DBGeneration, DBStoryItem.generation_id == DBGeneration.id)
        .filter(DBStoryItem.story_id == story_id)
        .order_by(DBStoryItem.start_time_ms)
        .all()
    )

    if not items:
        return None

    tracks = {t.index: t for t in db.query(DBStoryTrack).filter_by(story_id=story_id).all()}
    any_soloed = any(t.soloed for t in tracks.values())

    # --- decode once, at native rate ---------------------------------------
    # Decoding at each file's own rate lets us pick the project rate from what
    # the sources actually are, instead of forcing 24 kHz on a 48 kHz bed.
    loaded = []
    for item, generation in items:
        resolved_audio_path = generation.audio_path
        if getattr(item, "version_id", None):
            from ..database import GenerationVersion as DBGenerationVersion

            version = db.query(DBGenerationVersion).filter_by(id=item.version_id).first()
            if version:
                resolved_audio_path = version.audio_path

        audio_path = config.resolve_storage_path(resolved_audio_path)
        if audio_path is None or not audio_path.exists():
            logger.warning("Story %s: skipping item %s, audio missing", story_id, item.id)
            continue

        try:
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
        except Exception as exc:
            logger.warning("Story %s: skipping item %s, decode failed: %s", story_id, item.id, exc)
            continue

        loaded.append((item, _to_stereo(audio), int(sr)))

    if not loaded:
        return None

    project_sr = min(max(sr for _item, _audio, sr in loaded), MAX_PROJECT_SAMPLE_RATE)

    # --- per-lane submixes --------------------------------------------------
    lanes: dict[int, np.ndarray] = {}
    placements = []

    for item, audio, sr in loaded:
        if sr != project_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=project_sr)

        # Duration comes from the array we actually decoded, not from
        # generation.duration: a pinned version can be a different length, and
        # a NULL duration used to raise inside a bare except and silently drop
        # the clip from the export.
        trim_start = int(project_sr * max(getattr(item, "trim_start_ms", 0), 0) / 1000)
        trim_end = int(project_sr * max(getattr(item, "trim_end_ms", 0), 0) / 1000)
        audio = audio[:, trim_start : audio.shape[1] - trim_end if trim_end else None]
        if audio.shape[1] == 0:
            continue

        speed = float(getattr(item, "speed", 1.0) or 1.0)
        if speed != 1.0:
            # Phase vocoder, so pitch survives the tempo change.
            audio = np.stack([librosa.effects.time_stretch(ch, rate=speed) for ch in audio])

        audio = _apply_fades(
            audio,
            project_sr,
            int(getattr(item, "fade_in_ms", 0) or 0),
            int(getattr(item, "fade_out_ms", 0) or 0),
        )

        volume = float(getattr(item, "volume", 1.0) or 1.0)
        if volume != 1.0:
            audio = audio * volume

        placements.append((item.track, int(item.start_time_ms), audio))

    if not placements:
        return None

    total_samples = max(
        int(project_sr * start_ms / 1000) + audio.shape[1] for _track, start_ms, audio in placements
    )

    for track_index, start_ms, audio in placements:
        lane = lanes.get(track_index)
        if lane is None:
            lane = np.zeros((2, total_samples), dtype=np.float32)
            lanes[track_index] = lane

        start = int(project_sr * start_ms / 1000)
        end = min(start + audio.shape[1], total_samples)
        if start < total_samples:
            lane[:, start:end] += audio[:, : end - start]

    # --- track gain, mute and solo -----------------------------------------
    for index, lane in lanes.items():
        # A lane with no settings row means *defaults*, not *exempt* — it still
        # has to be silenced when another lane is soloed. Skipping it here let
        # the un-configured lane (usually the voice on track 0) play through a
        # solo of the music bed.
        track = tracks.get(index)
        muted = bool(track.muted) if track else False
        soloed = bool(track.soloed) if track else False
        volume = float(track.volume) if track else 1.0

        # Solo is a property of the whole story: once anything is soloed,
        # everything else is silent regardless of its own mute flag.
        if muted or (any_soloed and not soloed):
            lane[:] = 0.0
            continue
        if volume != 1.0:
            lane *= volume

    # --- ducking ------------------------------------------------------------
    # Runs after gain so the envelope reflects what will actually be heard,
    # and after mute/solo so a silenced lane ducks nothing.
    for index, lane in lanes.items():
        track = tracks.get(index)
        if track is None or track.duck_under_track is None:
            continue
        source = lanes.get(track.duck_under_track)
        if source is None:
            continue
        lane *= _duck_envelope(source, project_sr)

    final_audio = np.zeros((2, total_samples), dtype=np.float32)
    for lane in lanes.values():
        final_audio += lane

    peak = np.abs(final_audio).max()
    if peak > 1.0:
        final_audio /= peak

    return encode_audio(final_audio, project_sr, fmt)
