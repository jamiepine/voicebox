"""Post-migration data seeding and backfills."""

import json
import logging
import uuid

from .. import config

logger = logging.getLogger(__name__)


def backfill_generation_versions(SessionLocal, Generation, GenerationVersion) -> None:
    """Create 'clean' version entries for generations that predate the versions feature."""
    db = SessionLocal()
    try:
        existing_version_gen_ids = {
            row[0] for row in db.query(GenerationVersion.generation_id).all()
        }
        generations = db.query(Generation).filter(
            Generation.status == "completed",
            Generation.audio_path.isnot(None),
            Generation.audio_path != "",
        ).all()

        count = 0
        for gen in generations:
            if gen.id in existing_version_gen_ids:
                continue
            resolved_audio_path = config.resolve_storage_path(gen.audio_path)
            if resolved_audio_path is None or not resolved_audio_path.exists():
                continue
            version = GenerationVersion(
                id=str(uuid.uuid4()),
                generation_id=gen.id,
                label="clean",
                audio_path=gen.audio_path,
                effects_chain=None,
                is_default=True,
            )
            db.add(version)
            count += 1

        if count > 0:
            db.commit()
            logger.info("Backfilled %d generation version entries", count)
    finally:
        db.close()


def seed_builtin_presets(SessionLocal, EffectPreset) -> None:
    """Ensure built-in effect presets exist in the database."""
    from ..utils.effects import BUILTIN_PRESETS

    db = SessionLocal()
    try:
        for idx, (_key, preset_data) in enumerate(BUILTIN_PRESETS.items()):
            sort_order = preset_data.get("sort_order", idx)
            existing = db.query(EffectPreset).filter_by(name=preset_data["name"]).first()
            if not existing:
                preset = EffectPreset(
                    id=str(uuid.uuid4()),
                    name=preset_data["name"],
                    description=preset_data.get("description"),
                    effects_chain=json.dumps(preset_data["effects_chain"]),
                    is_builtin=True,
                    sort_order=sort_order,
                )
                db.add(preset)
            elif existing.sort_order != sort_order:
                existing.sort_order = sort_order
        db.commit()
    finally:
        db.close()
