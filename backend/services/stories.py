"""
Story management module.
"""

from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import os
import re
import shutil
import subprocess
import uuid
import tempfile
from pathlib import Path
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
    StoryItemSplit,
    StoryItemVersionUpdate,
)
from ..database import (
    Story as DBStory,
    StoryItem as DBStoryItem,
    Generation as DBGeneration,
    VoiceProfile as DBVoiceProfile,
)
from .history import _get_versions_for_generation
from ..utils.audio import load_audio, save_audio
import numpy as np


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

    result = []
    for story in stories:
        item_count = db.query(func.count(DBStoryItem.id)).filter(DBStoryItem.story_id == story.id).scalar()

        response = StoryResponse.model_validate(story)
        response.item_count = item_count
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

    # Get track from data or default to 0
    track = data.track if data.track is not None else 0

    # Calculate start_time_ms if not provided
    if data.start_time_ms is not None:
        start_time_ms = data.start_time_ms
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


@dataclass
class _Chapter:
    """Single chapter boundary used when exporting a story to m4b/mp3."""

    start_ms: int
    end_ms: int
    title: str


# Latin sentences end with [.!?] + whitespace; CJK sentences end with [。！？]
# and conventionally have no following whitespace, so the boundary is the mark
# itself. Both lookbehinds are zero-width — split() just consumes whitespace
# when present.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")


def _chapter_title_from_text(text: Optional[str], max_len: int = 80) -> str:
    """Derive a chapter title from the linked generation's text.

    Takes the first sentence (or the leading slice if the sentence is long).
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return "Chapter"
    first = _SENTENCE_BREAK.split(cleaned, maxsplit=1)[0].strip()
    if not first:
        return "Chapter"
    if len(first) > max_len:
        return first[:max_len].rstrip() + "…"
    return first


def _derive_chapters_auto(
    segments: List[dict],
    total_duration_ms: int,
) -> List[_Chapter]:
    """One chapter per story segment, ordered by start_time_ms.

    Each segment dict must carry ``start_time_ms`` and ``text``. The final
    chapter runs to ``total_duration_ms``. Segments with identical
    ``start_time_ms`` are deduped (only the first contributes a chapter
    boundary) so multi-track items don't produce zero-length chapters.
    """
    ordered = sorted(segments, key=lambda s: s["start_time_ms"])
    chapters: List[_Chapter] = []
    for seg in ordered:
        start = int(seg["start_time_ms"])
        if chapters and chapters[-1].start_ms == start:
            continue
        chapters.append(
            _Chapter(start_ms=start, end_ms=0, title=_chapter_title_from_text(seg.get("text")))
        )
    for i, ch in enumerate(chapters):
        ch.end_ms = chapters[i + 1].start_ms if i + 1 < len(chapters) else total_duration_ms
    # Skip degenerate trailing chapter that would have zero duration.
    return [ch for ch in chapters if ch.end_ms > ch.start_ms]


def _escape_ffmetadata(value: str) -> str:
    """Escape a metadata value for an FFMETADATA1 file.

    Per the spec (https://ffmpeg.org/ffmpeg-formats.html#Metadata-1) ``=``,
    ``;``, ``#``, ``\\``, and literal newlines must be backslash-escaped, or
    ffmpeg will reject (or silently mangle) the chapter entry.
    """
    return (
        value.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\n", "\\n")
    )


def _write_ffmetadata(chapters: List[_Chapter], path: Path) -> None:
    """Serialize chapter list to an FFMETADATA1 file ffmpeg can ingest via ``-i``."""
    lines = [";FFMETADATA1"]
    for ch in chapters:
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={ch.start_ms}",
                f"END={ch.end_ms}",
                f"title={_escape_ffmetadata(ch.title)}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ffmpeg_encode(
    wav_path: Path,
    out_path: Path,
    fmt: str,
    chapters: Optional[List[_Chapter]],
) -> None:
    """Transcode WAV → fmt with optional embedded chapter metadata.

    Raises RuntimeError on missing ffmpeg or non-zero exit.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for m4b/mp3 story export — install it or request format=wav"
        )

    cmd: List[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path)]
    meta_path: Optional[Path] = None
    try:
        if chapters:
            meta_path = wav_path.with_suffix(".chapters.txt")
            _write_ffmetadata(chapters, meta_path)
            cmd.extend(["-i", str(meta_path), "-map_metadata", "1", "-map_chapters", "1"])

        if fmt == "m4b":
            # The 'ipod' muxer is ffmpeg's name for the .m4b container.
            cmd.extend(["-map", "0:a", "-c:a", "aac", "-b:a", "128k", "-f", "ipod"])
        elif fmt == "mp3":
            cmd.extend(["-map", "0:a", "-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"])
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

        cmd.append(str(out_path))
        try:
            # Hard ceiling — a stuck ffmpeg must not pin server resources
            # indefinitely. 10 minutes is generous enough for a multi-hour
            # audiobook on slow hardware but still bounded.
            result = subprocess.run(cmd, capture_output=True, check=False, timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ffmpeg timed out during story export") from exc
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg exited {result.returncode}: {stderr}")
    finally:
        if meta_path is not None:
            meta_path.unlink(missing_ok=True)


def _make_tempfile(suffix: str) -> Path:
    """Create a closed-on-return temp file path with the given suffix.

    Uses ``tempfile.mkstemp`` rather than ``NamedTemporaryFile(delete=False).name``
    so the OS file descriptor is released immediately instead of lingering
    until garbage collection (Ruff SIM115).
    """
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(name)


async def export_story_audio(
    story_id: str,
    db: Session,
    fmt: str = "wav",
    chapters_mode: str = "none",
) -> Optional[bytes]:
    """
    Export story as a single mixed audio file with timecode-based mixing.

    Args:
        story_id: Story ID
        db: Database session
        fmt: Output container — "wav" (default), "m4b", or "mp3".
        chapters_mode: "none" (default) leaves chapter metadata off; "auto"
            derives one chapter per story item, titled from its generation
            text. WAV ignores this — chapters are an m4b/mp3 feature.

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

    # Load all audio files and calculate total duration
    audio_data = []
    sample_rate = 24000  # Default sample rate

    for item, generation in items:
        # Resolve audio path: use pinned version if set, otherwise generation default
        resolved_audio_path = generation.audio_path
        if getattr(item, "version_id", None):
            from ..database import GenerationVersion as DBGenerationVersion

            version = db.query(DBGenerationVersion).filter_by(id=item.version_id).first()
            if version:
                resolved_audio_path = version.audio_path

        audio_path = config.resolve_storage_path(resolved_audio_path)
        if audio_path is None or not audio_path.exists():
            continue

        try:
            audio, sr = load_audio(str(audio_path), sample_rate=sample_rate)
            sample_rate = sr  # Use actual sample rate from first file

            # Get trim values
            trim_start_ms = getattr(item, "trim_start_ms", 0)
            trim_end_ms = getattr(item, "trim_end_ms", 0)

            # Calculate effective duration
            original_duration_ms = int(generation.duration * 1000)
            effective_duration_ms = original_duration_ms - trim_start_ms - trim_end_ms

            # Slice audio based on trim values
            trim_start_sample = int((trim_start_ms / 1000.0) * sample_rate)
            trim_end_sample = int((trim_end_ms / 1000.0) * sample_rate)

            # Extract the trimmed portion
            if trim_end_ms > 0:
                trimmed_audio = (
                    audio[trim_start_sample:-trim_end_sample] if trim_end_sample > 0 else audio[trim_start_sample:]
                )
            else:
                trimmed_audio = audio[trim_start_sample:]

            # Apply per-clip volume to the export mix.
            volume = float(getattr(item, "volume", 1.0) or 1.0)
            if volume != 1.0:
                trimmed_audio = trimmed_audio * volume

            # Store audio with its timecode info
            start_time_ms = item.start_time_ms

            audio_data.append(
                {
                    "audio": trimmed_audio,
                    "start_time_ms": start_time_ms,
                    "duration_ms": effective_duration_ms,
                    "text": generation.text,
                }
            )
        except Exception:
            # Skip files that can't be loaded
            continue

    if not audio_data:
        return None

    # Calculate total duration: max(start_time_ms + duration_ms)
    max_end_time_ms = max((data["start_time_ms"] + data["duration_ms"] for data in audio_data), default=0)

    # Convert to samples
    total_samples = int((max_end_time_ms / 1000.0) * sample_rate)

    # Create output buffer initialized to zeros
    final_audio = np.zeros(total_samples, dtype=np.float32)

    # Mix each audio segment at its timecode position
    for data in audio_data:
        audio = data["audio"]
        start_time_ms = data["start_time_ms"]

        # Calculate start sample index
        start_sample = int((start_time_ms / 1000.0) * sample_rate)

        # Ensure we don't exceed buffer bounds
        audio_length = len(audio)
        end_sample = min(start_sample + audio_length, total_samples)

        if start_sample < total_samples:
            # Trim audio if it extends beyond buffer
            audio_to_mix = audio[: end_sample - start_sample]

            # Mix: add audio to existing buffer (overlapping audio will sum)
            # Normalize to prevent clipping (simple approach: divide by max)
            final_audio[start_sample:end_sample] += audio_to_mix

    # Normalize to prevent clipping
    max_val = np.abs(final_audio).max()
    if max_val > 1.0:
        final_audio = final_audio / max_val

    fmt = (fmt or "wav").lower()
    if fmt not in ("wav", "m4b", "mp3"):
        raise ValueError(f"Unsupported export format: {fmt}")

    chapters: Optional[List[_Chapter]] = None
    if fmt != "wav" and chapters_mode == "auto":
        chapters = _derive_chapters_auto(audio_data, max_end_time_ms) or None

    wav_path = _make_tempfile(suffix=".wav")
    out_path: Optional[Path] = None
    try:
        save_audio(final_audio, str(wav_path), sample_rate)
        if fmt == "wav":
            return wav_path.read_bytes()

        out_suffix = ".m4b" if fmt == "m4b" else ".mp3"
        out_path = _make_tempfile(suffix=out_suffix)
        # ffmpeg is CPU-bound and can run for several seconds on a real
        # audiobook — offload to a worker thread so it doesn't block the
        # FastAPI event loop while it runs.
        await asyncio.to_thread(_ffmpeg_encode, wav_path, out_path, fmt, chapters)
        return out_path.read_bytes()
    finally:
        wav_path.unlink(missing_ok=True)
        if out_path is not None:
            out_path.unlink(missing_ok=True)
