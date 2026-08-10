"""Voice profile management module."""

import json as _json
import logging
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import config
from ..database import Generation as DBGeneration, ProfileSample as DBProfileSample, VoiceProfile as DBVoiceProfile
from ..database.models import Folder as DBFolder
from ..models import (
    EffectConfig,
    ProfileSampleResponse,
    VoiceProfileCreate,
    VoiceProfileResponse,
)
from ..utils.audio import save_audio, validate_and_load_reference_audio
from ..utils.cache import _get_cache_dir, clear_profile_cache
from ..utils.images import process_avatar, validate_image

logger = logging.getLogger(__name__)

CLONING_ENGINES = {"qwen", "luxtts", "chatterbox", "chatterbox_turbo", "tada"}


def _profile_to_response(
    profile: DBVoiceProfile,
    generation_count: int = 0,
    sample_count: int = 0,
) -> VoiceProfileResponse:
    """Convert a DB profile to a VoiceProfileResponse, deserializing effects_chain."""
    effects_chain = None
    if profile.effects_chain:
        try:
            raw = _json.loads(profile.effects_chain)
            effects_chain = [EffectConfig(**e) for e in raw]
        except Exception as e:
            import logging

            logging.warning(f"Failed to parse effects_chain for profile {profile.id}: {e}")
    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        language=profile.language,
        avatar_path=profile.avatar_path,
        effects_chain=effects_chain,
        voice_type=getattr(profile, "voice_type", None) or "cloned",
        preset_engine=getattr(profile, "preset_engine", None),
        preset_voice_id=getattr(profile, "preset_voice_id", None),
        design_prompt=getattr(profile, "design_prompt", None),
        default_engine=getattr(profile, "default_engine", None),
        personality=getattr(profile, "personality", None),
        folder_id=getattr(profile, "folder_id", None),
        generation_count=generation_count,
        sample_count=sample_count,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _get_preset_voice_ids(engine: str) -> set[str]:
    if engine == "kokoro":
        from ..backends.kokoro_backend import KOKORO_VOICES

        return {voice_id for voice_id, _name, _gender, _lang in KOKORO_VOICES}

    if engine == "qwen_custom_voice":
        from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

        return {voice_id for voice_id, _name, _gender, _lang, _desc in QWEN_CUSTOM_VOICES}

    return set()


def _validate_profile_fields(
    *,
    voice_type: str,
    preset_engine: str | None,
    preset_voice_id: str | None,
    design_prompt: str | None,
    default_engine: str | None,
) -> str | None:
    if voice_type == "preset":
        if not preset_engine or not preset_voice_id:
            return "Preset profiles require both preset_engine and preset_voice_id"
        if default_engine and default_engine != preset_engine:
            return "Preset profiles must use their preset_engine as default_engine"

        available_voice_ids = _get_preset_voice_ids(preset_engine)
        if available_voice_ids and preset_voice_id not in available_voice_ids:
            return f"Preset voice '{preset_voice_id}' is not valid for engine '{preset_engine}'"
        return None

    if voice_type == "designed":
        if not design_prompt or not design_prompt.strip():
            return "Designed profiles require a design_prompt"
        if preset_engine or preset_voice_id:
            return "Designed profiles cannot set preset_engine or preset_voice_id"
        return None

    if preset_engine or preset_voice_id:
        return "Cloned profiles cannot set preset_engine or preset_voice_id"
    if design_prompt:
        return "Cloned profiles cannot set design_prompt"
    if default_engine and default_engine not in CLONING_ENGINES:
        return f"Cloned profiles cannot use default engine '{default_engine}'"
    return None


def validate_profile_engine(profile, engine: str) -> None:
    voice_type = getattr(profile, "voice_type", None) or "cloned"

    if voice_type == "preset":
        preset_engine = getattr(profile, "preset_engine", None)
        preset_voice_id = getattr(profile, "preset_voice_id", None)
        if not preset_engine or not preset_voice_id:
            raise ValueError(f"Preset profile {profile.id} is missing preset engine metadata")
        if preset_engine != engine:
            raise ValueError(
                f"Preset profile {profile.id} only supports engine '{preset_engine}', not '{engine}'"
            )
        return

    if voice_type == "designed":
        design_prompt = getattr(profile, "design_prompt", None)
        if not design_prompt or not design_prompt.strip():
            raise ValueError(f"Designed profile {profile.id} is missing design_prompt")
        return

    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles")


# How many "name (n)" variants to try before giving up. Only reached under
# genuine contention -- a single caller finds a free name on the first miss.
_NAME_ALLOCATION_ATTEMPTS = 50


def get_unique_profile_name(name: str, db: Session) -> str:
    """Return ``name``, or the first free "name (n)" variant.

    ``profiles.name`` is UNIQUE, so anything that creates a profile from an
    existing one -- import, duplicate -- has to resolve collisions first.

    This only *reads*, so it is a check-then-act: two concurrent callers can
    both be handed the same name and the second insert then fails the unique
    constraint. Prefer :func:`insert_profile_with_unique_name`, which settles
    the name by inserting it. This remains for callers that need a candidate
    name before they have a row to insert.
    """
    base_name = name
    counter = 1

    while True:
        existing = db.query(DBVoiceProfile).filter_by(name=name).first()
        if not existing:
            return name

        name = f"{base_name} ({counter})"
        counter += 1


def insert_profile_with_unique_name(
    base_name: str,
    db: Session,
    build_row: Callable[[str], DBVoiceProfile],
) -> DBVoiceProfile:
    """Insert a profile under the first free variant of *base_name*.

    The name is settled by the insert rather than by a preceding SELECT, so
    concurrent callers cannot both take it -- the unique constraint arbitrates
    and the loser retries with the next suffix instead of surfacing a 500.

    ``build_row`` is called per attempt because a rolled-back commit expunges
    the instance, so each try needs a fresh one.
    """
    name = base_name
    for counter in range(1, _NAME_ALLOCATION_ATTEMPTS + 1):
        row = build_row(name)
        db.add(row)
        try:
            db.commit()
            db.refresh(row)
            return row
        except IntegrityError:
            db.rollback()
            name = f"{base_name} ({counter})"

    raise ValueError(
        f"Could not find a free name for {base_name!r} after "
        f"{_NAME_ALLOCATION_ATTEMPTS} attempts"
    )


async def create_profile(
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse:
    """
    Create a new voice profile.

    Args:
        data: Profile creation data
        db: Database session

    Returns:
        Created profile

    Raises:
        ValueError: If a profile with the same name already exists
    """
    existing_profile = db.query(DBVoiceProfile).filter_by(name=data.name).first()
    if existing_profile:
        raise ValueError(f"A profile with the name '{data.name}' already exists. Please choose a different name.")

    # Auto-set default_engine for preset profiles
    default_engine = data.default_engine
    voice_type = data.voice_type or "cloned"
    if voice_type == "preset" and data.preset_engine and not default_engine:
        default_engine = data.preset_engine

    validation_error = _validate_profile_fields(
        voice_type=voice_type,
        preset_engine=data.preset_engine,
        preset_voice_id=data.preset_voice_id,
        design_prompt=data.design_prompt,
        default_engine=default_engine,
    )
    if validation_error:
        raise ValueError(validation_error)

    # The folder-assignment endpoint enforces that a voice only lands in a
    # voice folder; creation has to enforce the same contract, or a client can
    # file a new profile straight into a clip folder and bypass it.
    if data.folder_id is not None:
        folder = db.query(DBFolder).filter_by(id=data.folder_id).first()
        if folder is None:
            raise ValueError(f"Folder not found: {data.folder_id}")
        if folder.kind != "voice":
            raise ValueError("Target folder does not hold voices")

    db_profile = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        language=data.language,
        voice_type=voice_type,
        preset_engine=data.preset_engine,
        preset_voice_id=data.preset_voice_id,
        design_prompt=data.design_prompt,
        default_engine=default_engine,
        personality=data.personality,
        folder_id=data.folder_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)

    profile_dir = config.get_profiles_dir() / db_profile.id
    profile_dir.mkdir(parents=True, exist_ok=True)

    return _profile_to_response(db_profile)


async def add_profile_sample(
    profile_id: str,
    audio_path: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse:
    """
    Add a sample to a voice profile.

    Args:
        profile_id: Profile ID
        audio_path: Path to temporary audio file
        reference_text: Transcript of audio
        db: Database session

    Returns:
        Created sample
    """
    import asyncio

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    # Validate and load audio in a single pass, off the event loop
    is_valid, error_msg, audio, sr = await asyncio.to_thread(
        validate_and_load_reference_audio, audio_path
    )
    if not is_valid:
        raise ValueError(f"Invalid reference audio: {error_msg}")

    sample_id = str(uuid.uuid4())
    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    dest_path = profile_dir / f"{sample_id}.wav"
    await asyncio.to_thread(save_audio, audio, str(dest_path), sr)

    db_sample = DBProfileSample(
        id=sample_id,
        profile_id=profile_id,
        audio_path=config.to_storage_path(dest_path),
        reference_text=reference_text,
    )

    db.add(db_sample)

    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(db_sample)

    # Invalidate combined audio cache for this profile
    # Since a new sample was added, any cached combined audio is now stale
    clear_profile_cache(profile_id)

    return ProfileSampleResponse.model_validate(db_sample)


async def get_profile(
    profile_id: str,
    db: Session,
) -> VoiceProfileResponse | None:
    """
    Get a voice profile by ID.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        Profile or None if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    return _profile_to_response(profile)


def get_profile_orm_by_name_or_id(
    name_or_id: str,
    db: Session,
) -> DBVoiceProfile | None:
    """Resolve a profile from a user-supplied string that may be either id or name.

    Id is tried first (fast path, matches UUIDs). Name fallback is
    case-insensitive so agents can say "Morgan" regardless of casing.
    """
    if not name_or_id:
        return None
    row = db.query(DBVoiceProfile).filter(DBVoiceProfile.id == name_or_id).first()
    if row is not None:
        return row
    return (
        db.query(DBVoiceProfile)
        .filter(func.lower(DBVoiceProfile.name) == name_or_id.lower())
        .first()
    )


async def get_profile_samples(
    profile_id: str,
    db: Session,
) -> list[ProfileSampleResponse]:
    """
    Get all samples for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        List of samples
    """
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    return [ProfileSampleResponse.model_validate(s) for s in samples]


async def list_profiles(db: Session) -> list[VoiceProfileResponse]:
    """
    List all voice profiles with generation and sample counts.

    Args:
        db: Database session

    Returns:
        List of profiles
    """
    profiles = db.query(DBVoiceProfile).order_by(DBVoiceProfile.created_at.desc()).all()

    if not profiles:
        return []

    # Batch-fetch generation counts
    gen_counts_rows = (
        db.query(DBGeneration.profile_id, func.count(DBGeneration.id)).group_by(DBGeneration.profile_id).all()
    )
    gen_counts = {row[0]: row[1] for row in gen_counts_rows}

    # Batch-fetch sample counts
    sample_counts_rows = (
        db.query(DBProfileSample.profile_id, func.count(DBProfileSample.id)).group_by(DBProfileSample.profile_id).all()
    )
    sample_counts = {row[0]: row[1] for row in sample_counts_rows}

    return [
        _profile_to_response(
            p,
            generation_count=gen_counts.get(p.id, 0),
            sample_count=sample_counts.get(p.id, 0),
        )
        for p in profiles
    ]


async def update_profile(
    profile_id: str,
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse | None:
    """
    Update a voice profile.

    Args:
        profile_id: Profile ID
        data: Updated profile data
        db: Database session

    Returns:
        Updated profile or None if not found

    Raises:
        ValueError: If a profile with the same name already exists (different profile)
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    if profile.name != data.name:
        existing_profile = db.query(DBVoiceProfile).filter_by(name=data.name).first()
        if existing_profile:
            raise ValueError(f"A profile with the name '{data.name}' already exists. Please choose a different name.")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    preset_engine = getattr(profile, "preset_engine", None)
    preset_voice_id = getattr(profile, "preset_voice_id", None)
    design_prompt = getattr(profile, "design_prompt", None)
    default_engine = data.default_engine if data.default_engine is not None else getattr(profile, "default_engine", None)

    validation_error = _validate_profile_fields(
        voice_type=voice_type,
        preset_engine=preset_engine,
        preset_voice_id=preset_voice_id,
        design_prompt=design_prompt,
        default_engine=default_engine,
    )
    if validation_error:
        raise ValueError(validation_error)

    profile.name = data.name
    profile.description = data.description
    profile.language = data.language
    profile.personality = data.personality
    if data.default_engine is not None:
        profile.default_engine = data.default_engine or None  # empty string → NULL
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


async def delete_profile(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete a voice profile and all associated data.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return False

    db.query(DBProfileSample).filter_by(profile_id=profile_id).delete()

    db.delete(profile)
    db.commit()

    profile_dir = config.get_profiles_dir() / profile_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir)

    # Clean up combined audio cache files for this profile
    clear_profile_cache(profile_id)

    return True


async def delete_profile_sample(
    sample_id: str,
    db: Session,
) -> bool:
    """
    Delete a profile sample.

    Args:
        sample_id: Sample ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return False

    # Store profile_id before deleting
    profile_id = sample.profile_id

    audio_path = config.resolve_storage_path(sample.audio_path)
    if audio_path is not None and audio_path.exists():
        audio_path.unlink()

    db.delete(sample)
    db.commit()

    # Invalidate combined audio cache for this profile
    # Since the sample set changed, any cached combined audio is now stale
    clear_profile_cache(profile_id)

    return True


async def update_profile_sample(
    sample_id: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse | None:
    """
    Update a profile sample's reference text.

    Args:
        sample_id: Sample ID
        reference_text: Updated reference text
        db: Database session

    Returns:
        Updated sample or None if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return None

    # Store profile_id before updating
    profile_id = sample.profile_id

    sample.reference_text = reference_text
    db.commit()
    db.refresh(sample)

    # Invalidate combined audio cache for this profile
    # Since the reference text changed, cache keys and combined text are now stale
    clear_profile_cache(profile_id)

    return ProfileSampleResponse.model_validate(sample)


async def create_voice_prompt_for_profile(
    profile_id: str,
    db: Session,
    use_cache: bool = True,
    engine: str = "qwen",
) -> dict:
    """
    Create a voice prompt from a profile.

    For cloned profiles: combines all audio samples into a voice prompt.
    For preset profiles: returns the engine-specific preset voice reference.
    For designed profiles: returns the text design prompt (future).

    Args:
        profile_id: Profile ID
        db: Database session
        use_cache: Whether to use cached prompts
        engine: TTS engine to create prompt for

    Returns:
        Voice prompt dictionary
    """
    from ..backends import get_tts_backend_for_engine

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile not found: {profile_id}")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    validate_profile_engine(profile, engine)

    # ── Preset profiles: return engine-specific voice reference ──
    if voice_type == "preset":
        if not profile.preset_engine or not profile.preset_voice_id:
            raise ValueError(f"Preset profile {profile_id} is missing preset engine metadata")
        if profile.preset_engine != engine:
            raise ValueError(
                f"Preset profile {profile_id} only supports engine '{profile.preset_engine}', not '{engine}'"
            )
        return {
            "voice_type": "preset",
            "preset_engine": profile.preset_engine,
            "preset_voice_id": profile.preset_voice_id,
        }

    # ── Designed profiles: return text description (future) ──
    if voice_type == "designed":
        if not profile.design_prompt or not profile.design_prompt.strip():
            raise ValueError(f"Designed profile {profile_id} is missing design_prompt")
        return {
            "voice_type": "designed",
            "design_prompt": profile.design_prompt,
        }

    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles")

    # ── Cloned profiles: create from audio samples ──
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()

    if not samples:
        raise ValueError(f"No samples found for profile {profile_id}")

    tts_model = get_tts_backend_for_engine(engine)

    if len(samples) == 1:
        sample = samples[0]
        sample_audio_path = config.resolve_storage_path(sample.audio_path)
        if sample_audio_path is None:
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        voice_prompt, _ = await tts_model.create_voice_prompt(
            str(sample_audio_path),
            sample.reference_text,
            use_cache=use_cache,
        )
        return voice_prompt

    audio_paths = []
    for sample in samples:
        sample_audio_path = config.resolve_storage_path(sample.audio_path)
        if sample_audio_path is None:
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        audio_paths.append(str(sample_audio_path))
    reference_texts = [s.reference_text for s in samples]

    combined_audio, combined_text = await tts_model.combine_voice_prompts(
        audio_paths,
        reference_texts,
    )

    # Save combined audio to cache directory (persistent)
    # Create a hash of sample IDs to identify this specific combination
    import hashlib

    sample_ids_str = "-".join(sorted([s.id for s in samples]))
    combination_hash = hashlib.md5(sample_ids_str.encode()).hexdigest()[:12]

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined_path = cache_dir / f"combined_{profile_id}_{combination_hash}.wav"

    save_audio(combined_audio, str(combined_path), 24000)

    voice_prompt, _ = await tts_model.create_voice_prompt(
        str(combined_path),
        combined_text,
        use_cache=use_cache,
    )
    return voice_prompt


async def upload_avatar(
    profile_id: str,
    image_path: str,
    db: Session,
) -> VoiceProfileResponse:
    """
    Upload and process avatar image for a profile.

    Args:
        profile_id: Profile ID
        image_path: Path to uploaded image file
        db: Database session

    Returns:
        Updated profile
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    is_valid, error_msg = validate_image(image_path)
    if not is_valid:
        raise ValueError(error_msg)

    if profile.avatar_path:
        old_avatar = config.resolve_storage_path(profile.avatar_path)
        if old_avatar is not None and old_avatar.exists():
            old_avatar.unlink()

    # Determine file extension from uploaded file
    from PIL import Image

    with Image.open(image_path) as img:
        # Normalize JPEG variants (MPO is multi-picture format from some cameras)
        img_format = img.format
        if img_format in ("MPO", "JPG"):
            img_format = "JPEG"

        ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
        ext = ext_map.get(img_format, ".png")

    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_path = profile_dir / f"avatar{ext}"

    process_avatar(image_path, str(output_path))

    profile.avatar_path = config.to_storage_path(output_path)
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


async def delete_avatar(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete avatar image for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found or no avatar
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile or not profile.avatar_path:
        return False

    avatar_path = config.resolve_storage_path(profile.avatar_path)
    if avatar_path is not None and avatar_path.exists():
        avatar_path.unlink()

    profile.avatar_path = None
    profile.updated_at = datetime.utcnow()

    db.commit()

    return True


async def duplicate_profile(
    profile_id: str,
    db: Session,
    name: str | None = None,
) -> VoiceProfileResponse:
    """Copy a profile, its samples, and its avatar.

    Deliberately not implemented as export-then-import.  The transfer format
    carries only name/description/language (see export_import.py), so a
    round-trip silently drops personality, effects_chain, default_engine and
    every preset/designed field -- and refuses profiles with no samples,
    which is every preset voice.  Copying the row directly keeps all of it.

    Sample audio is copied byte-for-byte rather than re-encoded through
    add_profile_sample(): the source files were already validated when they
    were first added, and a duplicate should be identical, not resampled.
    """
    source = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not source:
        raise ValueError(f"Profile {profile_id} not found")

    new_id = str(uuid.uuid4())
    base_name = name.strip() if name else f"{source.name} (copy)"

    def _build(candidate: str) -> DBVoiceProfile:
        return DBVoiceProfile(
            id=new_id,
            name=candidate,
            description=source.description,
            language=source.language,
            effects_chain=source.effects_chain,
            voice_type=source.voice_type,
            preset_engine=source.preset_engine,
            preset_voice_id=source.preset_voice_id,
            design_prompt=source.design_prompt,
            default_engine=source.default_engine,
            personality=source.personality,
            folder_id=source.folder_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    # Reserve the name by inserting it, before any file work. Retrying after
    # the copies would mean undoing them; the directory is keyed by new_id, so
    # a name retry does not affect it.
    duplicate = insert_profile_with_unique_name(base_name, db, _build)

    new_dir = config.get_profiles_dir() / new_id
    new_dir.mkdir(parents=True, exist_ok=True)

    # Samples: copy each file under a fresh id, then point a new row at it.
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    for sample in samples:
        source_audio = config.resolve_storage_path(sample.audio_path)
        if source_audio is None or not source_audio.exists():
            # A profile can outlive its audio (moved data dir, manual
            # cleanup).  Skip the orphan rather than fail the whole copy.
            logger.warning(
                "Skipping sample %s while duplicating %s: audio missing at %s",
                sample.id,
                profile_id,
                sample.audio_path,
            )
            continue

        new_sample_id = str(uuid.uuid4())
        dest = new_dir / f"{new_sample_id}{source_audio.suffix}"
        shutil.copy2(source_audio, dest)
        db.add(
            DBProfileSample(
                id=new_sample_id,
                profile_id=new_id,
                audio_path=config.to_storage_path(dest),
                reference_text=sample.reference_text,
            )
        )

    if source.avatar_path:
        source_avatar = config.resolve_storage_path(source.avatar_path)
        if source_avatar is not None and source_avatar.exists():
            dest_avatar = new_dir / source_avatar.name
            shutil.copy2(source_avatar, dest_avatar)
            duplicate.avatar_path = config.to_storage_path(dest_avatar)

    db.commit()
    db.refresh(duplicate)

    sample_count = (
        db.query(func.count(DBProfileSample.id))
        .filter(DBProfileSample.profile_id == new_id)
        .scalar()
    )
    # A fresh copy has no generations of its own.
    return _profile_to_response(duplicate, generation_count=0, sample_count=sample_count)
