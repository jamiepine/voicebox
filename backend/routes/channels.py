"""Audio channel endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..services import channels
from ..database import get_db

router = APIRouter()


@router.get("/channels", response_model=list[models.AudioChannelResponse])
async def list_channels(db: Session = Depends(get_db)):
    """List all audio channels."""
    return await channels.list_channels(db)


@router.post("/channels", response_model=models.AudioChannelResponse)
async def create_channel(
    data: models.AudioChannelCreate,
    db: Session = Depends(get_db),
):
    """Create a new audio channel."""
    try:
        return await channels.create_channel(data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/channels/{channel_id}", response_model=models.AudioChannelResponse)
async def get_channel(
    channel_id: str,
    db: Session = Depends(get_db),
):
    """Get an audio channel by ID."""
    channel = await channels.get_channel(channel_id, db)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return channel


@router.put("/channels/{channel_id}", response_model=models.AudioChannelResponse)
async def update_channel(
    channel_id: str,
    data: models.AudioChannelUpdate,
    db: Session = Depends(get_db),
):
    """Update an audio channel."""
    try:
        channel = await channels.update_channel(channel_id, data, db)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
        return channel
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/channels/{channel_id}")
async def delete_channel(
    channel_id: str,
    db: Session = Depends(get_db),
):
    """Delete an audio channel."""
    try:
        success = await channels.delete_channel(channel_id, db)
        if not success:
            raise HTTPException(status_code=404, detail="Channel not found")
        return {"message": "Channel deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/channels/{channel_id}/voices")
async def get_channel_voices(
    channel_id: str,
    db: Session = Depends(get_db),
):
    """Get list of profile IDs assigned to a channel."""
    try:
        profile_ids = await channels.get_channel_voices(channel_id, db)
        return {"profile_ids": profile_ids}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/channels/{channel_id}/voices")
async def set_channel_voices(
    channel_id: str,
    data: models.ChannelVoiceAssignment,
    db: Session = Depends(get_db),
):
    """Set which voices are assigned to a channel."""
    try:
        await channels.set_channel_voices(channel_id, data, db)
        return {"message": "Channel voices updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
