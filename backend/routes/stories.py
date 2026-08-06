"""Story endpoints."""

import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import database, models
from ..services import stories
from ..app import safe_content_disposition
from ..database import get_db
from ..utils import ffmpeg
from ..utils.audio import EXPORT_FORMATS

router = APIRouter()


@router.get("/stories", response_model=list[models.StoryResponse])
async def list_stories(db: Session = Depends(get_db)):
    """List all stories."""
    return await stories.list_stories(db)


@router.post("/stories", response_model=models.StoryResponse)
async def create_story(
    data: models.StoryCreate,
    db: Session = Depends(get_db),
):
    """Create a new story."""
    try:
        return await stories.create_story(data, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stories/{story_id}", response_model=models.StoryDetailResponse)
async def get_story(
    story_id: str,
    db: Session = Depends(get_db),
):
    """Get a story with all its items."""
    story = await stories.get_story(story_id, db)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    return story


@router.put("/stories/{story_id}", response_model=models.StoryResponse)
async def update_story(
    story_id: str,
    data: models.StoryCreate,
    db: Session = Depends(get_db),
):
    """Update a story."""
    story = await stories.update_story(story_id, data, db)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    return story


@router.delete("/stories/{story_id}")
async def delete_story(
    story_id: str,
    db: Session = Depends(get_db),
):
    """Delete a story."""
    success = await stories.delete_story(story_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Story not found")
    return {"message": "Story deleted successfully"}


@router.post("/stories/{story_id}/items", response_model=models.StoryItemDetail)
async def add_story_item(
    story_id: str,
    data: models.StoryItemCreate,
    db: Session = Depends(get_db),
):
    """Add a generation to a story."""
    item = await stories.add_item_to_story(story_id, data, db)
    if not item:
        raise HTTPException(status_code=404, detail="Story or generation not found")
    return item


@router.delete("/stories/{story_id}/items/{item_id}")
async def remove_story_item(
    story_id: str,
    item_id: str,
    db: Session = Depends(get_db),
):
    """Remove a story item from a story."""
    success = await stories.remove_item_from_story(story_id, item_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Story item not found")
    return {"message": "Item removed successfully"}


@router.put("/stories/{story_id}/items/times")
async def update_story_item_times(
    story_id: str,
    data: models.StoryItemBatchUpdate,
    db: Session = Depends(get_db),
):
    """Update story item timecodes."""
    success = await stories.update_story_item_times(story_id, data, db)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid timecode update request")
    return {"message": "Item timecodes updated successfully"}


@router.put("/stories/{story_id}/items/reorder", response_model=list[models.StoryItemDetail])
async def reorder_story_items(
    story_id: str,
    data: models.StoryItemReorder,
    db: Session = Depends(get_db),
):
    """Reorder story items and recalculate timecodes."""
    items = await stories.reorder_story_items(story_id, data.generation_ids, db)
    if items is None:
        raise HTTPException(
            status_code=400, detail="Invalid reorder request - ensure all generation IDs belong to this story"
        )
    return items


@router.put("/stories/{story_id}/items/{item_id}/move", response_model=models.StoryItemDetail)
async def move_story_item(
    story_id: str,
    item_id: str,
    data: models.StoryItemMove,
    db: Session = Depends(get_db),
):
    """Move a story item (update position and/or track)."""
    item = await stories.move_story_item(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found")
    return item


@router.put("/stories/{story_id}/items/{item_id}/trim", response_model=models.StoryItemDetail)
async def trim_story_item(
    story_id: str,
    item_id: str,
    data: models.StoryItemTrim,
    db: Session = Depends(get_db),
):
    """Trim a story item."""
    item = await stories.trim_story_item(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found or invalid trim values")
    return item


@router.put("/stories/{story_id}/items/{item_id}/volume", response_model=models.StoryItemDetail)
async def update_story_item_volume(
    story_id: str,
    item_id: str,
    data: models.StoryItemVolumeUpdate,
    db: Session = Depends(get_db),
):
    """Set a story item's per-clip volume (linear gain, 0.0–2.0)."""
    item = await stories.update_story_item_volume(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found")
    return item


@router.put("/stories/{story_id}/items/{item_id}/fades", response_model=models.StoryItemDetail)
async def update_story_item_fades(
    story_id: str,
    item_id: str,
    data: models.StoryItemFadeUpdate,
    db: Session = Depends(get_db),
):
    """Set a story item's fade in/out lengths (ms)."""
    item = await stories.update_story_item_fades(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found")
    return item


@router.put("/stories/{story_id}/items/{item_id}/speed", response_model=models.StoryItemDetail)
async def update_story_item_speed(
    story_id: str,
    item_id: str,
    data: models.StoryItemSpeedUpdate,
    db: Session = Depends(get_db),
):
    """Set a story item's playback rate (pitch-preserving)."""
    item = await stories.update_story_item_speed(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found")
    return item


# ── Track mixer settings ─────────────────────────────────────────────


@router.get("/stories/{story_id}/tracks", response_model=list[models.StoryTrackResponse])
async def list_story_tracks(story_id: str, db: Session = Depends(get_db)):
    """Mixer settings for lanes that have them; others render at unity gain."""
    return await stories.list_story_tracks(story_id, db)


@router.put("/stories/{story_id}/tracks/{index}", response_model=models.StoryTrackResponse)
async def upsert_story_track(
    story_id: str,
    index: int,
    data: models.StoryTrackUpsert,
    db: Session = Depends(get_db),
):
    """Create or update one lane's mixer settings."""
    track = await stories.upsert_story_track(story_id, index, data, db)
    if track is None:
        raise HTTPException(status_code=404, detail="Story not found")
    return track


@router.delete("/stories/{story_id}/tracks/{index}")
async def delete_story_track(story_id: str, index: int, db: Session = Depends(get_db)):
    """Reset a lane to defaults. Clips on the lane are kept."""
    ok = await stories.delete_story_track(story_id, index, db)
    if not ok:
        raise HTTPException(status_code=404, detail="Track settings not found")
    return {"deleted": index}


@router.post("/stories/{story_id}/items/{item_id}/split", response_model=list[models.StoryItemDetail])
async def split_story_item(
    story_id: str,
    item_id: str,
    data: models.StoryItemSplit,
    db: Session = Depends(get_db),
):
    """Split a story item at a given time, creating two clips."""
    items = await stories.split_story_item(story_id, item_id, data, db)
    if items is None:
        raise HTTPException(status_code=404, detail="Story item not found or invalid split point")
    return items


@router.post("/stories/{story_id}/items/{item_id}/duplicate", response_model=models.StoryItemDetail)
async def duplicate_story_item(
    story_id: str,
    item_id: str,
    db: Session = Depends(get_db),
):
    """Duplicate a story item."""
    item = await stories.duplicate_story_item(story_id, item_id, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item not found")
    return item


@router.put("/stories/{story_id}/items/{item_id}/version", response_model=models.StoryItemDetail)
async def set_story_item_version(
    story_id: str,
    item_id: str,
    data: models.StoryItemVersionUpdate,
    db: Session = Depends(get_db),
):
    """Pin a story item to a specific generation version."""
    item = await stories.set_story_item_version(story_id, item_id, data, db)
    if item is None:
        raise HTTPException(status_code=404, detail="Story item or version not found")
    return item


@router.get("/stories/{story_id}/export-audio")
async def export_story_audio(
    story_id: str,
    format: str = "wav",
    normalize_loudness: bool = False,
    db: Session = Depends(get_db),
):
    """Export story as a single mixed audio file.

    ``format`` defaults to wav so existing callers are unaffected; every
    supported container is handled by the bundled libsndfile, no ffmpeg.

    ``normalize_loudness`` applies EBU R128 normalisation and needs ffmpeg. It
    is a no-op when ffmpeg is absent rather than an error — the export still
    succeeds with the mixer's own peak normalisation.
    """
    spec = EXPORT_FORMATS.get(format.lower())
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported export format '{format}'. Supported: {sorted(EXPORT_FORMATS)}",
        )

    try:
        story = db.query(database.Story).filter_by(id=story_id).first()
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        audio_bytes = await stories.export_story_audio(story_id, db, fmt=format.lower())
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Story has no audio items")

        if normalize_loudness:
            normalized = ffmpeg.normalize_loudness(audio_bytes, suffix=spec["ext"])
            if normalized is not None:
                audio_bytes = normalized

        safe_name = "".join(c for c in story.name if c.isalnum() or c in (" ", "-", "_")).strip()
        if not safe_name:
            safe_name = "story"
        filename = f"{safe_name}{spec['ext']}"

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=spec["mime"],
            headers={"Content-Disposition": safe_content_disposition("attachment", filename)},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
