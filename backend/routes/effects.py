"""Effects presets and generation version endpoints."""

import asyncio
import io
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..services import history
from ..database import Generation as DBGeneration, get_db

router = APIRouter()


@router.post("/effects/preview/{generation_id}")
async def preview_effects(
    generation_id: str,
    data: models.ApplyEffectsRequest,
    db: Session = Depends(get_db),
):
    """Apply effects to a generation's clean audio and stream back without saving."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation is not completed")

    from ..services import versions as versions_mod
    from ..utils.effects import apply_effects, validate_effects_chain
    from ..utils.audio import load_audio

    chain_dicts = [e.model_dump() for e in data.effects_chain]
    error = validate_effects_chain(chain_dicts)
    if error:
        raise HTTPException(status_code=400, detail=error)

    all_versions = versions_mod.list_versions(generation_id, db)
    clean_version = next((v for v in all_versions if v.effects_chain is None), None)
    source_path = clean_version.audio_path if clean_version else gen.audio_path
    resolved_source_path = config.resolve_storage_path(source_path)
    if resolved_source_path is None or not resolved_source_path.exists():
        raise HTTPException(status_code=404, detail="Source audio file not found")

    audio, sample_rate = await asyncio.to_thread(load_audio, str(resolved_source_path))
    processed = await asyncio.to_thread(apply_effects, audio, sample_rate, chain_dicts)

    import soundfile as sf

    buf = io.BytesIO()
    await asyncio.to_thread(lambda: sf.write(buf, processed, sample_rate, format="WAV"))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="preview_{generation_id}.wav"',
            "Cache-Control": "no-cache, no-store",
        },
    )


@router.get("/effects/available", response_model=models.AvailableEffectsResponse)
async def get_available_effects():
    """List all available effect types with parameter definitions."""
    from ..utils.effects import get_available_effects as _get_effects

    return models.AvailableEffectsResponse(effects=[models.AvailableEffect(**e) for e in _get_effects()])


@router.get("/effects/presets", response_model=list[models.EffectPresetResponse])
async def list_effect_presets(db: Session = Depends(get_db)):
    """List all effect presets (built-in + user-created)."""
    from ..services import effects as effects_mod

    return effects_mod.list_presets(db)


@router.get("/effects/presets/{preset_id}", response_model=models.EffectPresetResponse)
async def get_effect_preset(preset_id: str, db: Session = Depends(get_db)):
    """Get a specific effect preset."""
    from ..services import effects as effects_mod

    preset = effects_mod.get_preset(preset_id, db)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset


@router.post("/effects/presets", response_model=models.EffectPresetResponse)
async def create_effect_preset(
    data: models.EffectPresetCreate,
    db: Session = Depends(get_db),
):
    """Create a new effect preset."""
    from ..services import effects as effects_mod

    try:
        return effects_mod.create_preset(data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/effects/presets/{preset_id}", response_model=models.EffectPresetResponse)
async def update_effect_preset(
    preset_id: str,
    data: models.EffectPresetUpdate,
    db: Session = Depends(get_db),
):
    """Update an effect preset."""
    from ..services import effects as effects_mod

    try:
        result = effects_mod.update_preset(preset_id, data, db)
        if not result:
            raise HTTPException(status_code=404, detail="Preset not found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/effects/presets/{preset_id}")
async def delete_effect_preset(preset_id: str, db: Session = Depends(get_db)):
    """Delete a user effect preset."""
    from ..services import effects as effects_mod

    try:
        if not effects_mod.delete_preset(preset_id, db):
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/generations/{generation_id}/versions",
    response_model=list[models.GenerationVersionResponse],
)
async def list_generation_versions(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """List all versions for a generation."""
    gen = await history.get_generation(generation_id, db)
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    from ..services import versions as versions_mod

    return versions_mod.list_versions(generation_id, db)


@router.post(
    "/generations/{generation_id}/versions/apply-effects",
    response_model=models.GenerationVersionResponse,
)
async def apply_effects_to_generation(
    generation_id: str,
    data: models.ApplyEffectsRequest,
    db: Session = Depends(get_db),
):
    """Apply an effects chain to an existing generation, creating a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation is not completed")

    from ..services import versions as versions_mod
    from ..utils.effects import apply_effects, validate_effects_chain
    from ..utils.audio import load_audio, save_audio

    chain_dicts = [e.model_dump() for e in data.effects_chain]
    error = validate_effects_chain(chain_dicts)
    if error:
        raise HTTPException(status_code=400, detail=error)

    all_versions = versions_mod.list_versions(generation_id, db)
    source_version_id = data.source_version_id
    if source_version_id:
        source_version = next((v for v in all_versions if v.id == source_version_id), None)
        if not source_version:
            raise HTTPException(status_code=404, detail="Source version not found")
        source_path = source_version.audio_path
    else:
        clean_version = next((v for v in all_versions if v.effects_chain is None), None)
        if not clean_version:
            source_path = gen.audio_path
        else:
            source_path = clean_version.audio_path
            source_version_id = clean_version.id

    resolved_source_path = config.resolve_storage_path(source_path)
    if resolved_source_path is None or not resolved_source_path.exists():
        raise HTTPException(status_code=404, detail="Source audio file not found")

    audio, sample_rate = await asyncio.to_thread(load_audio, str(resolved_source_path))
    processed_audio = await asyncio.to_thread(apply_effects, audio, sample_rate, chain_dicts)

    version_id = str(uuid.uuid4())
    processed_path = config.get_generations_dir() / f"{generation_id}_{version_id[:8]}.wav"
    await asyncio.to_thread(save_audio, processed_audio, str(processed_path), sample_rate)

    label = data.label or f"version-{len(all_versions) + 1}"

    version = versions_mod.create_version(
        generation_id=generation_id,
        label=label,
        audio_path=config.to_storage_path(processed_path),
        db=db,
        effects_chain=chain_dicts,
        is_default=data.set_as_default,
        source_version_id=source_version_id,
    )

    return version


@router.put(
    "/generations/{generation_id}/versions/{version_id}/set-default",
    response_model=models.GenerationVersionResponse,
)
async def set_default_version(
    generation_id: str,
    version_id: str,
    db: Session = Depends(get_db),
):
    """Set a specific version as the default for a generation."""
    from ..services import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version or version.generation_id != generation_id:
        raise HTTPException(status_code=404, detail="Version not found")

    result = versions_mod.set_default_version(version_id, db)
    if not result:
        raise HTTPException(status_code=404, detail="Version not found")
    return result


@router.delete("/generations/{generation_id}/versions/{version_id}")
async def delete_generation_version(
    generation_id: str,
    version_id: str,
    db: Session = Depends(get_db),
):
    """Delete a version. Cannot delete the last remaining version."""
    from ..services import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version or version.generation_id != generation_id:
        raise HTTPException(status_code=404, detail="Version not found")

    if not versions_mod.delete_version(version_id, db):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the last remaining version",
        )
    return {"status": "deleted"}
