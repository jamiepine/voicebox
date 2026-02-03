"""
Settings API endpoints for managing application configuration.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Optional

from .database import get_db, Settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingUpdate(BaseModel):
    value: str


class SettingsResponse(BaseModel):
    settings: Dict[str, str]


@router.get("", response_model=SettingsResponse)
def get_settings(db: Session = Depends(get_db)):
    """Get all settings."""
    settings = db.query(Settings).all()
    return {
        "settings": {s.key: s.value for s in settings}
    }


@router.get("/{key}")
def get_setting(key: str, db: Session = Depends(get_db)):
    """Get a specific setting."""
    setting = db.query(Settings).filter(Settings.key == key).first()
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return {"key": key, "value": setting.value}


@router.put("/{key}")
def update_setting(key: str, data: SettingUpdate, db: Session = Depends(get_db)):
    """Update a setting value."""
    setting = db.query(Settings).filter(Settings.key == key).first()
    
    if not setting:
        # Create new setting if it doesn't exist
        setting = Settings(key=key, value=data.value)
        db.add(setting)
    else:
        setting.value = data.value
    
    db.commit()
    db.refresh(setting)
    
    # If TTS backend changed, need to reload the backend
    if key == "tts_backend":
        from .backends import reload_backend
        try:
            reload_backend()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Setting updated but failed to reload backend: {str(e)}"
            )
    
    return {"key": key, "value": setting.value, "reload_required": key == "tts_backend"}


@router.get("/backend/options")
def get_backend_options():
    """Get available TTS backend options."""
    return {
        "options": [
            {
                "value": "chatterbox_turbo",
                "label": "Chatterbox Turbo",
                "description": "Fast, high-quality TTS (4GB download on first use)"
            },
            {
                "value": "qwen",
                "label": "Qwen TTS", 
                "description": "Alternative TTS model"
            }
        ]
    }
