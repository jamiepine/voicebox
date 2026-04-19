"""
Effect presets CRUD operations.
"""

from __future__ import annotations

import json
import uuid
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ..utils.effects import validate_effects_chain

from ..database import EffectPreset as DBEffectPreset
from ..models import EffectPresetResponse, EffectPresetCreate, EffectPresetUpdate, EffectConfig


def _preset_response(p: DBEffectPreset) -> EffectPresetResponse:
    """Convert a DB preset row to a Pydantic response."""
    effects_chain = [EffectConfig(**e) for e in json.loads(p.effects_chain)]
    return EffectPresetResponse(
        id=p.id,
        name=p.name,
        description=p.description,
        effects_chain=effects_chain,
        is_builtin=p.is_builtin or False,
        created_at=p.created_at,
    )


def list_presets(db: Session) -> List[EffectPresetResponse]:
    """List all effect presets (built-in + user-created)."""
    presets = db.query(DBEffectPreset).order_by(DBEffectPreset.sort_order, DBEffectPreset.name).all()
    return [_preset_response(p) for p in presets]


def get_preset(preset_id: str, db: Session) -> Optional[EffectPresetResponse]:
    """Get a preset by ID."""
    p = db.query(DBEffectPreset).filter_by(id=preset_id).first()
    if not p:
        return None
    return _preset_response(p)


def get_preset_by_name(name: str, db: Session) -> Optional[EffectPresetResponse]:
    """Get a preset by name."""
    p = db.query(DBEffectPreset).filter_by(name=name).first()
    if not p:
        return None
    return _preset_response(p)


def create_preset(data: EffectPresetCreate, db: Session) -> EffectPresetResponse:
    """Create a new user effect preset."""

    chain_dicts = [e.model_dump() for e in data.effects_chain]
    error = validate_effects_chain(chain_dicts)
    if error:
        raise ValueError(error)

    # Check for duplicate name before insert
    existing = db.query(DBEffectPreset).filter_by(name=data.name).first()
    if existing:
        raise ValueError(f"A preset named '{data.name}' already exists")

    preset = DBEffectPreset(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        effects_chain=json.dumps(chain_dicts),
        is_builtin=False,
    )
    db.add(preset)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ValueError(f"A preset named '{data.name}' already exists")
    db.refresh(preset)
    return _preset_response(preset)


def update_preset(preset_id: str, data: EffectPresetUpdate, db: Session) -> Optional[EffectPresetResponse]:
    """Update a user effect preset. Cannot modify built-in presets."""
    preset = db.query(DBEffectPreset).filter_by(id=preset_id).first()
    if not preset:
        return None
    if preset.is_builtin:
        raise ValueError("Cannot modify built-in presets")

    if data.name is not None:
        preset.name = data.name
    if data.description is not None:
        preset.description = data.description
    if data.effects_chain is not None:

        chain_dicts = [e.model_dump() for e in data.effects_chain]
        error = validate_effects_chain(chain_dicts)
        if error:
            raise ValueError(error)
        preset.effects_chain = json.dumps(chain_dicts)

    db.commit()
    db.refresh(preset)
    return _preset_response(preset)


def delete_preset(preset_id: str, db: Session) -> bool:
    """Delete a user effect preset. Cannot delete built-in presets."""
    preset = db.query(DBEffectPreset).filter_by(id=preset_id).first()
    if not preset:
        return False
    if preset.is_builtin:
        raise ValueError("Cannot delete built-in presets")

    db.delete(preset)
    db.commit()
    return True
