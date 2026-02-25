"""
Audio channel management module.
"""

from typing import List, Optional
from datetime import datetime, timezone
import uuid
from sqlalchemy.orm import Session

from .models import (
    AudioChannelCreate,
    AudioChannelUpdate,
    AudioChannelResponse,
    ChannelVoiceAssignment,
    ProfileChannelAssignment,
)
from .database import (
    AudioChannel as DBAudioChannel,
    ChannelDeviceMapping as DBChannelDeviceMapping,
    ProfileChannelMapping as DBProfileChannelMapping,
    VoiceProfile as DBVoiceProfile,
)


async def list_channels(db: Session) -> List[AudioChannelResponse]:
    """List all audio channels."""
    channels = db.query(DBAudioChannel).all()
    result = []
    
    for channel in channels:
        # Get device IDs for this channel
        device_mappings = db.query(DBChannelDeviceMapping).filter_by(
            channel_id=channel.id
        ).all()
        device_ids = [m.device_id for m in device_mappings]
        
        result.append(AudioChannelResponse(
            id=channel.id,
            name=channel.name,
            is_default=channel.is_default,
            device_ids=device_ids,
            created_at=channel.created_at,
        ))
    
    return result


async def get_channel(channel_id: str, db: Session) -> Optional[AudioChannelResponse]:
    """Get a channel by ID."""
    channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
    if not channel:
        return None
    
    # Get device IDs
    device_mappings = db.query(DBChannelDeviceMapping).filter_by(
        channel_id=channel.id
    ).all()
    device_ids = [m.device_id for m in device_mappings]
    
    return AudioChannelResponse(
        id=channel.id,
        name=channel.name,
        is_default=channel.is_default,
        device_ids=device_ids,
        created_at=channel.created_at,
    )


async def create_channel(
    data: AudioChannelCreate,
    db: Session,
) -> AudioChannelResponse:
    """Create a new audio channel."""
    # Check if name already exists
    existing = db.query(DBAudioChannel).filter_by(name=data.name).first()
    if existing:
        raise ValueError(f"Channel with name '{data.name}' already exists")
    
    # Create channel
    channel = DBAudioChannel(
        id=str(uuid.uuid4()),
        name=data.name,
        is_default=False,
        created_at=datetime.now(timezone.utc),
    )
    db.add(channel)
    db.flush()
    
    # Add device mappings
    for device_id in data.device_ids:
        mapping = DBChannelDeviceMapping(
            id=str(uuid.uuid4()),
            channel_id=channel.id,
            device_id=device_id,
        )
        db.add(mapping)
    
    db.commit()
    db.refresh(channel)
    
    return AudioChannelResponse(
        id=channel.id,
        name=channel.name,
        is_default=channel.is_default,
        device_ids=data.device_ids,
        created_at=channel.created_at,
    )


async def update_channel(
    channel_id: str,
    data: AudioChannelUpdate,
    db: Session,
) -> Optional[AudioChannelResponse]:
    """Update an audio channel."""
    channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
    if not channel:
        return None
    
    if channel.is_default:
        raise ValueError("Cannot modify the default channel")
    
    # Update name if provided
    if data.name is not None:
        # Check if name already exists (excluding current channel)
        existing = db.query(DBAudioChannel).filter(
            DBAudioChannel.name == data.name,
            DBAudioChannel.id != channel_id
        ).first()
        if existing:
            raise ValueError(f"Channel with name '{data.name}' already exists")
        channel.name = data.name
    
    # Update device mappings if provided
    if data.device_ids is not None:
        # Delete existing mappings
        db.query(DBChannelDeviceMapping).filter_by(channel_id=channel_id).delete()
        
        # Add new mappings
        for device_id in data.device_ids:
            mapping = DBChannelDeviceMapping(
                id=str(uuid.uuid4()),
                channel_id=channel.id,
                device_id=device_id,
            )
            db.add(mapping)
    
    db.commit()
    db.refresh(channel)
    
    # Get updated device IDs
    device_mappings = db.query(DBChannelDeviceMapping).filter_by(
        channel_id=channel.id
    ).all()
    device_ids = [m.device_id for m in device_mappings]
    
    return AudioChannelResponse(
        id=channel.id,
        name=channel.name,
        is_default=channel.is_default,
        device_ids=device_ids,
        created_at=channel.created_at,
    )


async def delete_channel(channel_id: str, db: Session) -> bool:
    """Delete an audio channel."""
    channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
    if not channel:
        return False
    
    if channel.is_default:
        raise ValueError("Cannot delete the default channel")
    
    # Delete device mappings
    db.query(DBChannelDeviceMapping).filter_by(channel_id=channel_id).delete()
    
    # Delete profile-channel mappings
    db.query(DBProfileChannelMapping).filter_by(channel_id=channel_id).delete()
    
    # Delete channel
    db.delete(channel)
    db.commit()
    
    return True


async def get_channel_voices(channel_id: str, db: Session) -> List[str]:
    """Get list of profile IDs assigned to a channel."""
    mappings = db.query(DBProfileChannelMapping).filter_by(
        channel_id=channel_id
    ).all()
    return [m.profile_id for m in mappings]


async def set_channel_voices(
    channel_id: str,
    data: ChannelVoiceAssignment,
    db: Session,
) -> None:
    """Set which voices are assigned to a channel."""
    # Verify channel exists
    channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
    if not channel:
        raise ValueError(f"Channel {channel_id} not found")
    
    # Verify all profiles exist
    for profile_id in data.profile_ids:
        profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
    
    # Delete existing mappings for this channel
    db.query(DBProfileChannelMapping).filter_by(channel_id=channel_id).delete()
    
    # Add new mappings
    for profile_id in data.profile_ids:
        mapping = DBProfileChannelMapping(
            profile_id=profile_id,
            channel_id=channel_id,
        )
        db.add(mapping)
    
    db.commit()


async def get_profile_channels(profile_id: str, db: Session) -> List[str]:
    """Get list of channel IDs assigned to a profile."""
    mappings = db.query(DBProfileChannelMapping).filter_by(
        profile_id=profile_id
    ).all()
    return [m.channel_id for m in mappings]


async def set_profile_channels(
    profile_id: str,
    data: ProfileChannelAssignment,
    db: Session,
) -> None:
    """Set which channels a profile is assigned to."""
    # Verify profile exists
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")
    
    # Verify all channels exist
    for channel_id in data.channel_ids:
        channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
        if not channel:
            raise ValueError(f"Channel {channel_id} not found")
    
    # Delete existing mappings for this profile
    db.query(DBProfileChannelMapping).filter_by(profile_id=profile_id).delete()
    
    # Add new mappings
    for channel_id in data.channel_ids:
        mapping = DBProfileChannelMapping(
            profile_id=profile_id,
            channel_id=channel_id,
        )
        db.add(mapping)
    
    db.commit()
