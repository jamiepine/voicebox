"""SRT-driven dubbing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from .. import config, models
from ..app import safe_content_disposition
from ..database import DubbingProject as DBDubbingProject, DubbingSegment as DBDubbingSegment
from ..database import Generation as DBGeneration, get_db
from ..services import dubbing, history, profiles
from ..services.task_queue import cancel_generation as cancel_generation_job
from ..utils.tasks import get_task_manager

router = APIRouter(prefix="/dubbing", tags=["dubbing"])


@router.post("/release-memory")
async def release_dubbing_memory():
    """Free loaded TTS/STT backends when entering/leaving SRT2Voice-heavy work."""
    unloaded_tts = dubbing.release_dubbing_tts_memory("SRT2Voice explicit memory release")
    dubbing.release_dubbing_stt_memory("SRT2Voice explicit memory release")
    return {"message": "SRT2Voice memory release requested.", "unloaded_tts_backends": unloaded_tts}


def _serialize_segment(segment, db: Session) -> models.DubbingSegmentResponse:
    generation_audio_path = None
    generation_audio_absolute_path = None
    generation_error = None
    cut_audio_path = None
    cut_audio_absolute_path = None
    cut_duration_ms = None
    cut_source_start_ms = None
    cut_source_end_ms = None
    cut_source_type = None

    if segment.generation_id:
        generation = db.query(DBGeneration).filter_by(id=segment.generation_id).first()
        if generation is not None:
            generation_audio_path = generation.audio_path
            generation_error = generation.error
            resolved_path = (
                config.resolve_storage_path(generation.audio_path) if generation.audio_path else None
            )
            generation_audio_absolute_path = str(resolved_path) if resolved_path is not None else None

    cut_generation = dubbing.get_cut_generation(segment, db)
    if cut_generation is not None:
        cut_audio_path = cut_generation.audio_path
        cut_duration_ms = (
            int(round(cut_generation.duration * 1000))
            if cut_generation.duration is not None
            else None
        )
        resolved_cut_path = (
            config.resolve_storage_path(cut_generation.audio_path) if cut_generation.audio_path else None
        )
        cut_audio_absolute_path = str(resolved_cut_path) if resolved_cut_path is not None else None
        cut_bounds = dubbing.get_cut_source_bounds(segment.project_id, segment.id)
        if cut_bounds is not None:
            cut_source_start_ms = int(cut_bounds["cut_start_ms"])
            cut_source_end_ms = int(cut_bounds["cut_end_ms"])
            cut_source_type = str(cut_bounds["source_type"])

    return models.DubbingSegmentResponse(
        id=segment.id,
        project_id=segment.project_id,
        segment_order=segment.segment_order,
        srt_index=segment.srt_index,
        start_tc=segment.start_tc,
        end_tc=segment.end_tc,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        target_duration_ms=segment.target_duration_ms,
        text_lines=segment.text_lines,
        text=segment.text,
        pace_group_id=segment.pace_group_id,
        speaker=segment.speaker,
        generation_id=segment.generation_id,
        generation_audio_path=generation_audio_path,
        generation_audio_absolute_path=generation_audio_absolute_path,
        generation_error=generation_error,
        cut_generation_id=cut_generation.id if cut_generation is not None else None,
        cut_audio_path=cut_audio_path,
        cut_audio_absolute_path=cut_audio_absolute_path,
        cut_duration_ms=cut_duration_ms,
        cut_source_start_ms=cut_source_start_ms,
        cut_source_end_ms=cut_source_end_ms,
        cut_source_type=cut_source_type,
        actual_duration_ms=segment.actual_duration_ms,
        delta_ms=segment.delta_ms,
        fit_status=segment.fit_status,
        status=segment.status,
        created_at=segment.created_at,
        updated_at=segment.updated_at,
    )


def _serialize_project(project, db: Session) -> models.DubbingProjectResponse:
    segments = dubbing.list_project_segments(project.id, db)
    pace_groups = dubbing.build_pace_group_responses(project, segments)
    full_narration = dubbing.get_full_narration_generation(project.id, db)
    cut_count = len(dubbing.list_cut_generations(project.id, db))
    full_narration_generation_elapsed_ms = None
    full_narration_revision_ms = None
    if full_narration is not None and full_narration.status in {"completed", "failed"}:
        full_narration_generation_elapsed_ms = dubbing.read_full_narration_elapsed_ms(full_narration.id)
        if full_narration.audio_path:
            audio_path = config.resolve_storage_path(full_narration.audio_path)
            if audio_path is not None and audio_path.exists():
                full_narration_revision_ms = int(round(audio_path.stat().st_mtime * 1000))
    elif full_narration is not None and full_narration.created_at is not None:
        full_narration_revision_ms = int(round(full_narration.created_at.timestamp() * 1000))
    db.refresh(project)
    return models.DubbingProjectResponse(
        id=project.id,
        name=project.name,
        source_type=project.source_type,
        source_path=project.source_path,
        engine=project.engine,
        language=project.language,
        profile_id=project.profile_id,
        style_prompt=project.style_prompt,
        pace_override=project.pace_override,
        temperature=project.temperature,
        group_pace_overrides=dubbing.get_group_override_map(project),
        full_narration_generation_id=full_narration.id if full_narration is not None else None,
        full_narration_status=full_narration.status if full_narration is not None else None,
        full_narration_audio_path=full_narration.audio_path if full_narration is not None else None,
        full_narration_duration_ms=(
            int(round(full_narration.duration * 1000))
            if full_narration is not None and full_narration.duration is not None
            else None
        ),
        full_narration_revision_ms=full_narration_revision_ms,
        full_narration_generation_elapsed_ms=full_narration_generation_elapsed_ms,
        full_narration_error=full_narration.error if full_narration is not None else None,
        post_processed_segment_count=cut_count,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
        pace_groups=[models.DubbingPaceGroupResponse(**group) for group in pace_groups],
        segments=[_serialize_segment(segment, db) for segment in segments],
    )


def _serialize_project_list_item(project, db: Session) -> models.DubbingProjectListItemResponse:
    segments = dubbing.list_project_segments(project.id, db)
    exact_count = sum(1 for segment in segments if segment.fit_status == "exact")
    warning_count = sum(1 for segment in segments if segment.fit_status == "warning")
    failed_count = sum(1 for segment in segments if segment.status == "failed")
    pending_count = sum(1 for segment in segments if segment.status in {"pending", "generating"})
    return models.DubbingProjectListItemResponse(
        id=project.id,
        name=project.name,
        source_type=project.source_type,
        language=project.language,
        profile_id=project.profile_id,
        status=project.status,
        segment_count=len(segments),
        exact_count=exact_count,
        warning_count=warning_count,
        failed_count=failed_count,
        pending_count=pending_count,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.get("/projects", response_model=list[models.DubbingProjectListItemResponse])
async def list_projects(db: Session = Depends(get_db)):
    projects = db.query(DBDubbingProject).order_by(DBDubbingProject.updated_at.desc()).all()
    return [_serialize_project_list_item(project, db) for project in projects]


@router.post("/import-srt", response_model=models.DubbingProjectResponse)
async def import_srt(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not (file.filename or "").lower().endswith(".srt"):
        raise HTTPException(status_code=400, detail="Only .srt files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("cp1252")

    try:
        project = dubbing.create_project_from_srt(filename=file.filename or "import.srt", content=content, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _serialize_project(project, db)


@router.get("/projects/{project_id}", response_model=models.DubbingProjectResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    return _serialize_project(project, db)


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    await dubbing.delete_project(project, db)
    return {"message": "Dubbing project deleted."}


@router.post("/projects/{project_id}/segments/{segment_id}/generate", response_model=models.DubbingSegmentResponse)
async def generate_segment(
    project_id: str,
    segment_id: str,
    data: models.DubbingSegmentGenerateRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    segment = dubbing.get_segment_or_none(project_id, segment_id, db)
    if segment is None:
        raise HTTPException(status_code=404, detail="Dubbing segment not found.")

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    try:
        engine = dubbing.resolve_dubbing_engine_for_profile(profile, data.engine)
        profiles.validate_profile_engine(profile, engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await dubbing.queue_segment_generation(
        project=project,
        segment=segment,
        request=data,
        db=db,
        engine=engine,
    )
    db.refresh(segment)
    return _serialize_segment(segment, db)


@router.put("/projects/{project_id}/segments/{segment_id}", response_model=models.DubbingSegmentResponse)
async def update_segment(
    project_id: str,
    segment_id: str,
    data: models.DubbingSegmentUpdateRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    segment = dubbing.get_segment_or_none(project_id, segment_id, db)
    if segment is None:
        raise HTTPException(status_code=404, detail="Dubbing segment not found.")

    try:
        await dubbing.update_segment_text(segment, db, text=data.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if dubbing.update_project_status(project, db):
        db.commit()
    db.refresh(segment)
    return _serialize_segment(segment, db)


@router.put("/projects/{project_id}/segments/{segment_id}/timing", response_model=models.DubbingSegmentResponse)
async def update_segment_timing(
    project_id: str,
    segment_id: str,
    data: models.DubbingSegmentTimingUpdateRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    segment = dubbing.get_segment_or_none(project_id, segment_id, db)
    if segment is None:
        raise HTTPException(status_code=404, detail="Dubbing segment not found.")

    try:
        await dubbing.update_segment_timing(
            segment,
            db,
            start_ms=data.start_ms,
            end_ms=data.end_ms,
            preserve_audio=data.preserve_audio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if dubbing.update_project_status(project, db):
        db.commit()
    db.refresh(segment)
    return _serialize_segment(segment, db)


@router.delete("/projects/{project_id}/segments/{segment_id}", response_model=models.DubbingProjectResponse)
async def delete_segment(project_id: str, segment_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    segment = dubbing.get_segment_or_none(project_id, segment_id, db)
    if segment is None:
        raise HTTPException(status_code=404, detail="Dubbing segment not found.")

    try:
        await dubbing.delete_segment(segment, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.refresh(project)
    return _serialize_project(project, db)


@router.put("/projects/{project_id}/settings", response_model=models.DubbingProjectResponse)
async def update_project_settings(
    project_id: str,
    data: models.DubbingProjectSettingsUpdateRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    try:
        pace_override = data.pace_override if "pace_override" in data.model_fields_set else project.pace_override
        temperature = data.temperature if "temperature" in data.model_fields_set else project.temperature
        await dubbing.update_project_settings(
            project,
            db,
            pace_override=pace_override,
            temperature=temperature,
            name=data.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    db.refresh(project)
    return _serialize_project(project, db)


@router.put("/projects/{project_id}/groups/{group_id}/pace", response_model=models.DubbingProjectResponse)
async def update_group_pace(
    project_id: str,
    group_id: str,
    data: models.DubbingGroupPaceUpdateRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    try:
        await dubbing.update_group_pace_override(
            project,
            db,
            group_id=group_id,
            pace_override=data.pace_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db.refresh(project)
    return _serialize_project(project, db)


@router.post("/projects/{project_id}/segments/{segment_id}/auto-fit", response_model=models.DubbingSegmentResponse)
async def auto_fit_segment(
    project_id: str,
    segment_id: str,
    data: models.DubbingAutoFitRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    segment = dubbing.get_segment_or_none(project_id, segment_id, db)
    if segment is None:
        raise HTTPException(status_code=404, detail="Dubbing segment not found.")

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    try:
        engine = dubbing.resolve_dubbing_engine_for_profile(profile, data.engine)
        profiles.validate_profile_engine(profile, engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    segment.status = "generating"
    segment.fit_status = "unknown"
    project.status = "processing"
    project.profile_id = data.profile_id
    project.style_prompt = dubbing.sanitize_dubbing_instructions(data.instruct or data.style_prompt)
    project.language = data.language
    project.engine = engine
    db.commit()
    db.refresh(segment)

    dubbing.start_auto_fit_segment(project_id=project_id, segment_id=segment_id, request=data, engine=engine)
    return _serialize_segment(segment, db)


@router.post("/projects/{project_id}/generate-all", response_model=models.DubbingProjectResponse)
async def auto_fit_project(
    project_id: str,
    data: models.DubbingAutoFitRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    try:
        engine = dubbing.resolve_dubbing_engine_for_profile(profile, data.engine)
        profiles.validate_profile_engine(profile, engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    project.status = "processing"
    project.engine = engine
    project.profile_id = data.profile_id
    project.style_prompt = dubbing.sanitize_dubbing_instructions(data.instruct or data.style_prompt)
    project.language = data.language
    db.commit()

    dubbing.start_auto_fit_project(project_id=project_id, request=data, engine=engine)
    return _serialize_project(project, db)


@router.post("/projects/{project_id}/generate-full-narration", response_model=models.DubbingProjectResponse)
async def generate_full_narration(
    project_id: str,
    data: models.DubbingFullNarrationRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")

    try:
        engine = dubbing.resolve_dubbing_engine_for_profile(profile, data.engine)
        profiles.validate_profile_engine(profile, engine)
        await dubbing.queue_full_narration_generation(
            project=project,
            request=data,
            db=db,
            engine=engine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.refresh(project)
    return _serialize_project(project, db)


@router.post("/projects/{project_id}/post-process", response_model=models.DubbingProjectResponse)
async def post_process_project(project_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    try:
        await dubbing.post_process_full_narration_cuts(project, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.refresh(project)
    return _serialize_project(project, db)


@router.post("/projects/{project_id}/auto-cut", response_model=models.DubbingAutoCutResponse)
async def build_project_auto_cut(project_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    try:
        return await dubbing.build_auto_cut_timeline_clips(project, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        dubbing.release_dubbing_stt_memory("auto cut endpoint")


@router.post("/projects/{project_id}/tempo-suggestion", response_model=models.DubbingTempoSuggestionResponse)
async def suggest_project_tempo(project_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    try:
        return await dubbing.suggest_project_tempo(project, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        dubbing.release_dubbing_stt_memory("tempo suggestion endpoint")


@router.post("/projects/{project_id}/apply-tempo", response_model=models.DubbingApplyTempoResponse)
async def apply_project_tempo(
    project_id: str,
    data: models.DubbingApplyTempoRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    try:
        return await dubbing.apply_project_suggested_tempo(
            project,
            db,
            multiplier=data.multiplier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        dubbing.release_dubbing_stt_memory("apply tempo endpoint")


@router.post(
    "/projects/{project_id}/segments/{segment_id}/manual-cut",
    response_model=models.DubbingSegmentResponse,
)
async def create_manual_segment_cut(
    project_id: str,
    segment_id: str,
    data: models.DubbingManualCutRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    segment = dubbing.get_segment_or_none(project_id, segment_id, db)
    if segment is None:
        raise HTTPException(status_code=404, detail="Dubbing segment not found.")

    try:
        await dubbing.create_manual_cut_from_full_narration(
            project,
            segment,
            db,
            cut_start_ms=data.cut_start_ms,
            cut_end_ms=data.cut_end_ms,
            use_previous_cut_end=data.use_previous_cut_end,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.refresh(segment)
    return _serialize_segment(segment, db)


@router.delete(
    "/projects/{project_id}/segments/{segment_id}/generation",
    response_model=models.DubbingSegmentResponse,
)
async def delete_segment_generation(project_id: str, segment_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")
    segment = dubbing.get_segment_or_none(project_id, segment_id, db)
    if segment is None:
        raise HTTPException(status_code=404, detail="Dubbing segment not found.")
    if not segment.generation_id and dubbing.get_cut_generation(segment, db) is None:
        raise HTTPException(status_code=404, detail="This segment has no generation to delete.")

    deleted = await dubbing.delete_segment_generation(segment, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Linked generation not found.")

    if dubbing.update_project_status(project, db):
        db.commit()
    db.refresh(segment)
    return _serialize_segment(segment, db)


@router.get("/projects/{project_id}/export-audio")
async def export_project_audio(project_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    wav_bytes = await dubbing.build_project_timeline_wav(project_id, db)
    if not wav_bytes:
        raise HTTPException(
            status_code=400,
            detail="No generated segment audio is available to export for this project.",
        )

    safe_name = "".join(c for c in project.name[:50] if c.isalnum() or c in (" ", "-", "_")).strip()
    filename = f"{safe_name or 'dubbing-project'}.timeline.wav"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": safe_content_disposition("attachment", filename)},
    )


@router.post("/projects/{project_id}/export-audio")
async def export_project_visible_timeline_audio(
    project_id: str,
    data: models.DubbingTimelineExportRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    wav_bytes = await dubbing.build_project_visible_timeline_wav(project_id, db, clips=data.clips)
    if not wav_bytes:
        raise HTTPException(
            status_code=400,
            detail="No visible timeline audio is available to export for this project.",
        )

    safe_name = "".join(c for c in project.name[:50] if c.isalnum() or c in (" ", "-", "_")).strip()
    filename = f"{safe_name or 'dubbing-project'}.timeline.wav"
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": safe_content_disposition("attachment", filename)},
    )


@router.get("/projects/{project_id}/export-package")
async def export_project_package(project_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    package_bytes = await dubbing.build_project_export_package(project_id, db)
    if not package_bytes:
        raise HTTPException(
            status_code=400,
            detail="No dubbing package could be built for this project.",
        )

    safe_name = "".join(c for c in project.name[:50] if c.isalnum() or c in (" ", "-", "_")).strip()
    filename = f"{safe_name or 'dubbing-project'}.dubbing.zip"
    return Response(
        content=package_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": safe_content_disposition("attachment", filename)},
    )


@router.post("/projects/{project_id}/export-package")
async def export_project_visible_timeline_package(
    project_id: str,
    data: models.DubbingTimelineExportRequest,
    db: Session = Depends(get_db),
):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    timeline_wav = await dubbing.build_project_visible_timeline_wav(project_id, db, clips=data.clips)
    package_bytes = await dubbing.build_project_export_package(project_id, db, timeline_wav=timeline_wav)
    if not package_bytes:
        raise HTTPException(
            status_code=400,
            detail="No dubbing package could be built for this project.",
        )

    safe_name = "".join(c for c in project.name[:50] if c.isalnum() or c in (" ", "-", "_")).strip()
    filename = f"{safe_name or 'dubbing-project'}.dubbing.zip"
    return Response(
        content=package_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": safe_content_disposition("attachment", filename)},
    )


@router.post("/projects/{project_id}/cancel-all")
async def cancel_project_tasks(project_id: str, db: Session = Depends(get_db)):
    project = dubbing.get_project_or_none(project_id, db)
    if project is None:
        raise HTTPException(status_code=404, detail="Dubbing project not found.")

    task_manager = get_task_manager()
    cancelled = 0
    segments = (
        db.query(DBDubbingSegment)
        .filter_by(project_id=project_id)
        .order_by(DBDubbingSegment.segment_order.asc())
        .all()
    )

    for segment in segments:
        if not segment.generation_id:
            continue
        generation = db.query(DBGeneration).filter_by(id=segment.generation_id).first()
        if generation is None:
            segment.generation_id = None
            segment.status = "pending"
            segment.fit_status = "unknown"
            segment.actual_duration_ms = None
            segment.delta_ms = None
            continue

        generation_status = generation.status or "completed"
        if generation_status not in {"loading_model", "generating"}:
            continue

        cancellation_state = cancel_generation_job(generation.id)
        cancelled += 1
        if cancellation_state is not None:
            task_manager.complete_generation(generation.id)
        await history.update_generation_status(
            generation_id=generation.id,
            status="failed",
            db=db,
            error=(
                "Generation cancelled by user"
                if cancellation_state is not None
                else "Stale generation reset by user"
            ),
        )
        segment.generation_id = None
        segment.status = "pending"
        segment.fit_status = "unknown"
        segment.actual_duration_ms = None
        segment.delta_ms = None

    full_narration = dubbing.get_full_narration_generation(project_id, db)
    if full_narration is not None and (full_narration.status or "completed") in {"loading_model", "generating"}:
        cancellation_state = cancel_generation_job(full_narration.id)
        cancelled += 1
        if cancellation_state is not None:
            task_manager.complete_generation(full_narration.id)
        await history.update_generation_status(
            generation_id=full_narration.id,
            status="failed",
            db=db,
            error=(
                "Generation cancelled by user"
                if cancellation_state is not None
                else "Stale generation reset by user"
            ),
        )

    dubbing.update_project_status(project, db)
    db.commit()
    return {"message": f"Cancelled {cancelled} active task(s).", "cancelled": cancelled}
