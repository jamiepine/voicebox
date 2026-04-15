"""
Unified TTS generation orchestration.

Replaces the three near-identical closures (_run_generation, _run_retry,
_run_regenerate) that lived in main.py with a single ``run_generation()``
function parameterized by *mode*.

Mode differences:
  - "generate"   : full pipeline -- save clean version, optionally apply
                    effects and create a processed version.
  - "retry"      : re-runs a failed generation with the same seed.
                    No effects, no version creation.
  - "regenerate" : re-runs with seed=None for variation.  Creates a new
                    version with an auto-incremented "take-N" label.
"""

from __future__ import annotations

import traceback
from typing import Literal, Optional

from .. import config
from . import history, profiles
from ..database import get_db
from ..utils.tasks import get_task_manager


async def run_generation(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int],
    normalize: bool = False,
    effects_chain: Optional[list] = None,
    instruct: Optional[str] = None,
    mode: Literal["generate", "retry", "regenerate"],
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
    version_id: Optional[str] = None,
    sampling_params: Optional[dict] = None,
    inject_breaths: bool = False,
    jitter_ms: int = 0,
    humanize_text: bool = False,
    humanize_intensity: Optional[str] = None,
) -> None:
    """Execute TTS inference and persist the result.

    This is the single entry point for all background generation work.
    It is designed to be enqueued via ``services.task_queue.enqueue_generation``.
    """
    from ..backends import load_engine_model, get_tts_backend_for_engine, engine_needs_trim
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import normalize_audio, save_audio, trim_tts_output

    task_manager = get_task_manager()
    bg_db = next(get_db())

    try:
        # LLM text preprocessing (before loading TTS model to leave RAM for Ollama)
        if humanize_text:
            from ..utils.text_preprocess import inject_disfluencies
            intensity = humanize_intensity or "light"
            text = await inject_disfluencies(text, language, intensity)

        tts_model = get_tts_backend_for_engine(engine)

        if not tts_model.is_loaded():
            await history.update_generation_status(generation_id, "loading_model", bg_db)

        # Free Whisper memory before loading TTS
        from ..backends import ensure_tts_memory
        ensure_tts_memory()

        await load_engine_model(engine, model_size)

        voice_prompt = await profiles.create_voice_prompt_for_profile(
            profile_id,
            bg_db,
            use_cache=True,
            engine=engine,
        )

        await history.update_generation_status(generation_id, "generating", bg_db)
        trim_fn = trim_tts_output if engine_needs_trim(engine) else None

        effective_seed = seed if mode != "regenerate" else None
        _max_chunk_chars = max_chunk_chars if max_chunk_chars is not None else 800
        _crossfade_ms = crossfade_ms if crossfade_ms is not None else 50

        from ..utils.tag_router import has_paralinguistic_tags
        if has_paralinguistic_tags(text) and engine in ("qwen",):
            from ..utils.hybrid_generate import generate_hybrid
            audio, sample_rate = await generate_hybrid(
                text,
                voice_prompt,
                language=language,
                seed=effective_seed,
                instruct=instruct,
                sampling_params=sampling_params,
                max_chunk_chars=_max_chunk_chars,
                crossfade_ms=_crossfade_ms,
                jitter_ms=jitter_ms,
                primary_engine=engine,
                primary_model_size=model_size,
            )
        else:
            gen_kwargs: dict = dict(
                language=language,
                seed=effective_seed,
                instruct=instruct,
                trim_fn=trim_fn,
                max_chunk_chars=_max_chunk_chars,
                crossfade_ms=_crossfade_ms,
            )
            if sampling_params:
                gen_kwargs["sampling_params"] = sampling_params
            if jitter_ms > 0:
                gen_kwargs["jitter_ms"] = jitter_ms

            audio, sample_rate = await generate_chunked(tts_model, text, voice_prompt, **gen_kwargs)

        from ..backends import touch_tts_model
        touch_tts_model(engine)

        # Trim voice cloning warm-up for Qwen engines
        if engine in ("qwen",) and voice_prompt.get("ref_audio"):
            from ..utils.audio import trim_leading_warmup
            audio = trim_leading_warmup(
                audio, sample_rate,
                ref_audio_duration=voice_prompt.get("ref_audio_duration", 0.0),
                ref_text=voice_prompt.get("ref_text", ""),
            )

        # Breath injection (if requested)
        if inject_breaths:
            from ..utils.breath_injection import inject_breaths as _inject_breaths
            audio = _inject_breaths(audio, sample_rate)

        # --- Normalize (generate and regenerate always; retry skips) -----
        if normalize or mode == "regenerate":
            audio = normalize_audio(audio)

        duration = len(audio) / sample_rate

        # --- Persist audio and update status -----------------------------
        if mode == "generate":
            final_path = _save_generate(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                effects_chain=effects_chain,
                save_audio=save_audio,
                db=bg_db,
            )
        elif mode == "retry":
            final_path = _save_retry(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
            )
        elif mode == "regenerate":
            final_path = _save_regenerate(
                generation_id=generation_id,
                version_id=version_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
                db=bg_db,
            )

        await history.update_generation_status(
            generation_id=generation_id,
            status="completed",
            db=bg_db,
            audio_path=final_path,
            duration=duration,
        )

    except Exception as e:
        traceback.print_exc()
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error=str(e),
        )
    finally:
        task_manager.complete_generation(generation_id)
        bg_db.close()


def _save_generate(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    effects_chain: Optional[list],
    save_audio,
    db,
) -> str:
    """Save clean version and optionally an effects-processed version.

    Returns the final audio path (processed if effects were applied,
    otherwise clean).
    """
    from . import versions as versions_mod

    clean_audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(clean_audio_path), sample_rate)

    has_effects = effects_chain and any(e.get("enabled", True) for e in effects_chain)

    versions_mod.create_version(
        generation_id=generation_id,
        label="original",
        audio_path=config.to_storage_path(clean_audio_path),
        db=db,
        effects_chain=None,
        is_default=not has_effects,
    )

    final_audio_path = str(clean_audio_path)

    if has_effects:
        from ..utils.effects import apply_effects, validate_effects_chain

        assert effects_chain is not None

        error_msg = validate_effects_chain(effects_chain)
        if error_msg:
            import logging
            logging.getLogger(__name__).warning("invalid effects chain, skipping: %s", error_msg)
            versions_mod.set_default_version(
                versions_mod.list_versions(generation_id, db)[0].id, db
            )
        else:
            processed_audio = apply_effects(audio, sample_rate, effects_chain)
            processed_path = config.get_generations_dir() / f"{generation_id}_processed.wav"
            save_audio(processed_audio, str(processed_path), sample_rate)
            final_audio_path = str(processed_path)
            versions_mod.create_version(
                generation_id=generation_id,
                label="version-2",
                audio_path=config.to_storage_path(processed_path),
                db=db,
                effects_chain=effects_chain,
                is_default=True,
            )

    return config.to_storage_path(final_audio_path)


def _save_retry(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    save_audio,
) -> str:
    """Save retry output -- single file, no versions.

    Returns the audio path.
    """
    audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(audio_path), sample_rate)
    return config.to_storage_path(audio_path)


def _save_regenerate(
    *,
    generation_id: str,
    version_id: Optional[str],
    audio,
    sample_rate: int,
    save_audio,
    db,
) -> str:
    """Save regeneration output as a new version with auto-label.

    Returns the audio path.
    """
    from . import versions as versions_mod

    import uuid as _uuid

    suffix = _uuid.uuid4().hex[:8]
    audio_path = config.get_generations_dir() / f"{generation_id}_{suffix}.wav"
    save_audio(audio, str(audio_path), sample_rate)

    # Count via DB query rather than list length to avoid TOCTOU race
    from ..database import GenerationVersion as DBGenerationVersion

    count = db.query(DBGenerationVersion).filter_by(generation_id=generation_id).count()
    label = f"take-{count + 1}"

    versions_mod.create_version(
        generation_id=generation_id,
        label=label,
        audio_path=config.to_storage_path(audio_path),
        db=db,
        effects_chain=None,
        is_default=True,
    )

    return config.to_storage_path(audio_path)
