"""
Fine-tuning orchestration for per-profile LoRA adapters.

Manages training samples, dataset preparation, and training subprocess lifecycle.
"""

import asyncio
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from sqlalchemy.orm import Session

from . import config
from .database import (
    FinetuneSample as DBFinetuneSample,
    FinetuneJob as DBFinetuneJob,
    VoiceProfile as DBVoiceProfile,
    ProfileSample as DBProfileSample,
)
from .models import FinetuneSampleResponse, FinetuneJobResponse
from .utils.tasks import get_task_manager
from .utils.progress import get_progress_manager


def _get_profile_finetune_dir(profile_id: str) -> Path:
    """Get the finetune directory for a specific profile."""
    path = config.get_finetune_dir() / profile_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_samples_dir(profile_id: str) -> Path:
    """Get the samples directory for a profile's finetune data."""
    path = _get_profile_finetune_dir(profile_id) / "samples"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_dataset_dir(profile_id: str) -> Path:
    """Get the dataset directory for a profile's finetune data."""
    path = _get_profile_finetune_dir(profile_id) / "dataset"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_adapters_dir(profile_id: str, job_id: str) -> Path:
    """Get the adapter directory for a specific training job."""
    path = _get_profile_finetune_dir(profile_id) / "adapters" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================
# SAMPLE MANAGEMENT
# ============================================

async def add_sample(
    profile_id: str,
    audio_path: str,
    transcript: str,
    duration_seconds: float,
    db: Session,
) -> FinetuneSampleResponse:
    """Add a training sample for fine-tuning."""
    sample_id = str(uuid.uuid4())
    samples_dir = _get_samples_dir(profile_id)

    # Copy audio to finetune samples directory
    dest_path = samples_dir / f"{sample_id}.wav"
    from .utils.audio import load_audio, save_audio
    audio, sr = load_audio(audio_path)
    save_audio(audio, str(dest_path), sr)

    # Check if this is the first sample (auto-set as ref audio)
    existing_count = db.query(DBFinetuneSample).filter_by(profile_id=profile_id).count()
    is_ref = existing_count == 0

    db_sample = DBFinetuneSample(
        id=sample_id,
        profile_id=profile_id,
        audio_path=str(dest_path),
        transcript=transcript,
        duration_seconds=duration_seconds,
        is_ref_audio=is_ref,
        created_at=datetime.now(timezone.utc),
    )

    db.add(db_sample)
    db.commit()
    db.refresh(db_sample)

    return FinetuneSampleResponse.model_validate(db_sample)


async def list_samples(
    profile_id: str,
    db: Session,
) -> List[FinetuneSampleResponse]:
    """List all finetune samples for a profile."""
    samples = (
        db.query(DBFinetuneSample)
        .filter_by(profile_id=profile_id)
        .order_by(DBFinetuneSample.created_at.asc())
        .all()
    )
    return [FinetuneSampleResponse.model_validate(s) for s in samples]


async def delete_sample(
    sample_id: str,
    db: Session,
    profile_id: Optional[str] = None,
) -> bool:
    """Delete a finetune sample.

    Args:
        sample_id: The sample ID to delete.
        db: Database session.
        profile_id: If provided, validates that the sample belongs to this profile
                     (prevents IDOR — deleting another profile's samples).
    """
    filters = {"id": sample_id}
    if profile_id:
        filters["profile_id"] = profile_id

    sample = db.query(DBFinetuneSample).filter_by(**filters).first()
    if not sample:
        return False

    # Delete audio file
    audio_path = Path(sample.audio_path)
    if audio_path.exists():
        audio_path.unlink()

    db.delete(sample)
    db.commit()
    return True


async def set_ref_audio(
    sample_id: str,
    profile_id: str,
    db: Session,
) -> bool:
    """Mark a sample as the reference audio for this profile's dataset."""
    # Unset all existing ref audio for this profile
    db.query(DBFinetuneSample).filter_by(
        profile_id=profile_id, is_ref_audio=True
    ).update({"is_ref_audio": False})

    # Set the new ref audio
    sample = db.query(DBFinetuneSample).filter_by(id=sample_id, profile_id=profile_id).first()
    if not sample:
        return False

    sample.is_ref_audio = True
    db.commit()
    return True


def _audio_file_hash(path: str) -> str:
    """Compute a short SHA-256 hash of an audio file's raw bytes for dedup."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


async def import_profile_samples(
    profile_id: str,
    sample_ids: Optional[List[str]],
    db: Session,
) -> List[FinetuneSampleResponse]:
    """Import existing profile samples into finetune training set."""
    if sample_ids:
        profile_samples = (
            db.query(DBProfileSample)
            .filter(
                DBProfileSample.profile_id == profile_id,
                DBProfileSample.id.in_(sample_ids),
            )
            .all()
        )
    else:
        profile_samples = (
            db.query(DBProfileSample)
            .filter_by(profile_id=profile_id)
            .all()
        )

    # Build a set of audio hashes already in the finetune samples for this profile
    # to prevent importing duplicates (more reliable than transcript comparison)
    existing_finetune_samples = (
        db.query(DBFinetuneSample)
        .filter_by(profile_id=profile_id)
        .all()
    )
    existing_hashes = set()
    for es in existing_finetune_samples:
        try:
            existing_hashes.add(_audio_file_hash(es.audio_path))
        except (OSError, IOError):
            continue

    imported = []
    for ps in profile_samples:
        # Check for duplicate by audio content hash
        try:
            source_hash = _audio_file_hash(ps.audio_path)
        except (OSError, IOError):
            continue
        if source_hash in existing_hashes:
            continue

        # Get duration
        try:
            from .utils.audio import load_audio
            audio, sr = load_audio(ps.audio_path)
            duration = len(audio) / sr
        except Exception:
            duration = 0.0

        result = await add_sample(
            profile_id=profile_id,
            audio_path=ps.audio_path,
            transcript=ps.reference_text,
            duration_seconds=duration,
            db=db,
        )
        imported.append(result)
        # Track the newly added hash to prevent dupes within the same import batch
        existing_hashes.add(source_hash)

    return imported


# ============================================
# TRAINING ORCHESTRATION
# ============================================

# Global reference to active training subprocess
_active_training_process: Optional[asyncio.subprocess.Process] = None
_active_training_job_id: Optional[str] = None
_training_lock = asyncio.Lock()


async def get_status(profile_id: str, db: Session) -> dict:
    """Get finetune status for a profile."""
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    # Get active job
    active_job = (
        db.query(DBFinetuneJob)
        .filter(
            DBFinetuneJob.profile_id == profile_id,
            DBFinetuneJob.status.in_(["pending", "preparing", "training"]),
        )
        .first()
    )

    # Sample stats
    samples = db.query(DBFinetuneSample).filter_by(profile_id=profile_id).all()
    total_duration = sum(s.duration_seconds for s in samples)

    return {
        "has_finetune": profile.has_finetune or False,
        "has_active_adapter": bool(profile.active_adapter_path),
        "active_job": FinetuneJobResponse.model_validate(active_job) if active_job else None,
        "sample_count": len(samples),
        "total_duration_seconds": total_duration,
    }


async def prepare_dataset(profile_id: str, job_id: str, db: Session) -> Path:
    """
    Prepare training dataset: build JSONL manifest and copy ref audio.

    Returns the path to the dataset directory.
    """
    dataset_dir = _get_dataset_dir(profile_id)

    # Get all samples
    samples = (
        db.query(DBFinetuneSample)
        .filter_by(profile_id=profile_id)
        .all()
    )

    if not samples:
        raise ValueError("No training samples found")

    # Find reference audio (marked sample, or longest)
    ref_sample = next((s for s in samples if s.is_ref_audio), None)
    if not ref_sample:
        ref_sample = max(samples, key=lambda s: s.duration_seconds)

    # Copy ref audio to dataset dir
    ref_audio_dest = dataset_dir / "ref_audio.wav"
    shutil.copy2(ref_sample.audio_path, str(ref_audio_dest))

    # Build JSONL manifest
    manifest_path = dataset_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for sample in samples:
            entry = {
                "audio": sample.audio_path,
                "text": sample.transcript,
                "ref_audio": str(ref_audio_dest),
            }
            f.write(json.dumps(entry) + "\n")

    return dataset_dir


async def start_training(
    profile_id: str,
    config_dict: dict,
    db: Session,
) -> str:
    """
    Start fine-tuning for a profile.

    Creates a job record, prepares the dataset, and launches the training subprocess.
    Returns the job ID.
    """
    global _active_training_process, _active_training_job_id

    task_manager = get_task_manager()

    # Acquire lock to prevent TOCTOU race between check and start
    async with _training_lock:
        # Check if training is already running
        if task_manager.is_finetune_active():
            raise ValueError("A fine-tuning job is already running. Cancel it first or wait for completion.")

        # Get samples
        samples = db.query(DBFinetuneSample).filter_by(profile_id=profile_id).all()
        if len(samples) < 10:
            raise ValueError(f"Need at least 10 training samples, got {len(samples)}")

        total_duration = sum(s.duration_seconds for s in samples)

        # Create job record
        job_id = str(uuid.uuid4())
        job = DBFinetuneJob(
            id=job_id,
            profile_id=profile_id,
            status="preparing",
            num_samples=len(samples),
            total_audio_duration_seconds=total_duration,
            epochs=config_dict.get("epochs", 3),
            learning_rate=config_dict.get("learning_rate", 2e-5),
            batch_size=config_dict.get("batch_size", 1),
            lora_rank=config_dict.get("lora_rank", 32),
            label=config_dict.get("label"),
            created_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Track in task manager (must be inside lock to prevent TOCTOU)
        task_manager.start_finetune(job_id, profile_id, total_epochs=job.epochs)

    # Launch async training pipeline (outside lock — runs independently)
    asyncio.create_task(_run_training_pipeline(profile_id, job_id, config_dict))

    return job_id


def _update_job_status(job_id: str, **fields):
    """Update a finetune job using a short-lived DB session.

    Each call opens and closes its own session so the training pipeline
    never holds a long-lived connection (training can run for hours).
    """
    from .database import SessionLocal

    db = SessionLocal()
    try:
        job = db.query(DBFinetuneJob).filter_by(id=job_id).first()
        if job:
            for k, v in fields.items():
                setattr(job, k, v)
            db.commit()
    finally:
        db.close()


def _update_profile_after_training(profile_id: str, adapter_dir: str):
    """Mark a profile as fine-tuned using a short-lived DB session."""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
        if profile:
            profile.has_finetune = True
            profile.active_adapter_path = adapter_dir
            profile.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


async def _run_training_pipeline(profile_id: str, job_id: str, training_config: dict):
    """Run the full training pipeline asynchronously.

    Uses short-lived DB sessions for each operation instead of holding
    a single session open for the entire (potentially hours-long) training.
    """
    global _active_training_process, _active_training_job_id

    from .database import SessionLocal
    task_manager = get_task_manager()

    try:
        # Verify job exists
        db = SessionLocal()
        try:
            job = db.query(DBFinetuneJob).filter_by(id=job_id).first()
            if not job:
                return
        finally:
            db.close()

        # Stage 1: Prepare dataset
        print(f"[Finetune] Preparing dataset for profile {profile_id}...")
        task_manager.update_finetune(job_id, status="preparing")
        _update_job_status(job_id, status="preparing")

        try:
            db = SessionLocal()
            try:
                dataset_dir = await prepare_dataset(profile_id, job_id, db)
            finally:
                db.close()
        except Exception as e:
            print(f"[Finetune] Dataset preparation failed: {e}")
            _update_job_status(
                job_id,
                status="failed",
                error_message=f"Dataset preparation failed: {e}",
                completed_at=datetime.now(timezone.utc),
            )
            task_manager.error_finetune(job_id, str(e))
            task_manager.complete_finetune(job_id)
            return

        # Stage 2: Unload TTS model to free GPU memory
        print("[Finetune] Unloading TTS model to free GPU memory...")
        try:
            from .tts import unload_tts_model
            unload_tts_model()
        except Exception as e:
            print(f"[Finetune] Warning: failed to unload TTS model: {e}")

        # Stage 3: Launch training subprocess
        print(f"[Finetune] Launching training subprocess...")
        _update_job_status(
            job_id,
            status="training",
            started_at=datetime.now(timezone.utc),
        )
        task_manager.update_finetune(job_id, status="training")

        # Write config file for worker
        finetune_dir = _get_profile_finetune_dir(profile_id)
        config_path = finetune_dir / "config.json"
        adapter_output_dir = str(_get_adapters_dir(profile_id, job_id))
        log_path = finetune_dir / "training_log.jsonl"

        worker_config = {
            "profile_id": profile_id,
            "job_id": job_id,
            "dataset_dir": str(dataset_dir),
            "manifest_path": str(dataset_dir / "manifest.jsonl"),
            "ref_audio_path": str(dataset_dir / "ref_audio.wav"),
            "output_dir": adapter_output_dir,
            "log_path": str(log_path),
            "epochs": training_config.get("epochs", 3),
            "learning_rate": training_config.get("learning_rate", 2e-5),
            "batch_size": training_config.get("batch_size", 1),
            "lora_rank": training_config.get("lora_rank", 32),
            "lora_alpha": training_config.get("lora_alpha", training_config.get("lora_rank", 32) * 2),
        }
        with open(config_path, "w") as f:
            json.dump(worker_config, f, indent=2)

        # Clear old log
        if log_path.exists():
            log_path.unlink()

        # Launch subprocess
        import sys
        worker_script = str(Path(__file__).parent / "finetune_worker.py")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, worker_script, str(config_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _active_training_process = process
            _active_training_job_id = job_id
        except Exception as e:
            print(f"[Finetune] Failed to launch training subprocess: {e}")
            _update_job_status(
                job_id,
                status="failed",
                error_message=f"Failed to launch training: {e}",
                completed_at=datetime.now(timezone.utc),
            )
            task_manager.error_finetune(job_id, str(e))
            task_manager.complete_finetune(job_id)
            return

        # Stage 4: Monitor progress by tailing the log file
        print(f"[Finetune] Monitoring training progress (PID: {process.pid})...")
        await _monitor_training(process, log_path, job_id, profile_id)

    except Exception as e:
        print(f"[Finetune] Unexpected error: {e}")
        _update_job_status(
            job_id,
            status="failed",
            error_message=str(e),
            completed_at=datetime.now(timezone.utc),
        )
        task_manager.error_finetune(job_id, str(e))
        task_manager.complete_finetune(job_id)
    finally:
        _active_training_process = None
        _active_training_job_id = None


async def _monitor_training(
    process: asyncio.subprocess.Process,
    log_path: Path,
    job_id: str,
    profile_id: str,
):
    """Monitor training subprocess by tailing the log file.

    Uses short-lived DB sessions — no session is held open during the
    (potentially hours-long) wait between log entries.
    """
    import time as _time

    task_manager = get_task_manager()
    progress_manager = get_progress_manager()
    progress_key = f"finetune-{job_id}"

    last_pos = 0
    last_log_activity = _time.time()
    STALE_THRESHOLD = 600  # 10 minutes with no log output = potentially stuck

    while True:
        # Check if process has finished
        retcode = process.returncode
        if retcode is not None:
            break

        # Read new log entries
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()

                if new_lines:
                    last_log_activity = _time.time()

                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        _process_log_entry(entry, job_id, profile_id, task_manager, progress_manager, progress_key)
                    except json.JSONDecodeError:
                        continue
            except Exception:
                pass

        # Warn if no log output for a long time (training may be stuck)
        elapsed = _time.time() - last_log_activity
        if elapsed > STALE_THRESHOLD:
            print(f"[Finetune] Warning: No training progress for {int(elapsed)}s, subprocess may be stuck")
            # Reset timer to avoid spamming
            last_log_activity = _time.time()

        # Wait before checking again
        await asyncio.sleep(2)

    # Process has exited — read remaining log entries
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                f.seek(last_pos)
                remaining = f.readlines()
            for line in remaining:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    _process_log_entry(entry, job_id, profile_id, task_manager, progress_manager, progress_key)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    # Check exit code and finalize
    retcode = process.returncode

    from .database import SessionLocal
    db = SessionLocal()
    try:
        job = db.query(DBFinetuneJob).filter_by(id=job_id).first()

        if retcode == 0 and job and job.status == "training":
            adapter_dir = _get_adapters_dir(profile_id, job_id)
            adapter_config = adapter_dir / "adapter_config.json"

            if adapter_config.exists():
                job.status = "completed"
                job.adapter_path = str(adapter_dir)
                job.completed_at = datetime.now(timezone.utc)
                db.commit()

                # Update profile in a separate short-lived session
                _update_profile_after_training(profile_id, str(adapter_dir))

                print(f"[Finetune] Training completed successfully for profile {profile_id}")
                task_manager.update_finetune(job_id, status="completed")
                progress_manager.update_progress(progress_key, 100, 100, "Training complete", "complete")
            else:
                job.status = "failed"
                job.error_message = "Training completed but no adapter files found"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                task_manager.error_finetune(job_id, "No adapter files")
        elif job and job.status not in ("completed", "cancelled"):
            stderr_data = b""
            try:
                stderr_data = await process.stderr.read()
            except Exception:
                pass
            error_msg = stderr_data.decode("utf-8", errors="replace")[-500:] if stderr_data else f"Exit code: {retcode}"

            job.status = "failed"
            job.error_message = error_msg
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            print(f"[Finetune] Training failed: {error_msg}")
            task_manager.error_finetune(job_id, error_msg)
    finally:
        db.close()

    task_manager.complete_finetune(job_id)


def _process_log_entry(
    entry: dict,
    job_id: str,
    profile_id: str,
    task_manager,
    progress_manager,
    progress_key: str,
):
    """Process a single training log entry using a short-lived DB session."""
    entry_type = entry.get("type", "progress")

    if entry_type == "progress":
        epoch = entry.get("epoch", 0)
        step = entry.get("step", 0)
        total_steps = entry.get("total_steps", 0)
        loss = entry.get("loss")

        # Update job in DB with short-lived session
        _update_job_status(
            job_id,
            current_epoch=epoch,
            current_step=step,
            total_steps=total_steps,
            **({"current_loss": loss} if loss is not None else {}),
        )

        # Update task manager (in-memory)
        task_manager.update_finetune(
            job_id,
            current_epoch=epoch,
            current_step=step,
            total_steps=total_steps,
            current_loss=loss,
        )

        # Update progress manager for SSE (in-memory)
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0
        eta_seconds = entry.get("eta_seconds")
        eta_str = f", ETA: {eta_seconds // 60}m{eta_seconds % 60}s" if eta_seconds else ""
        progress_manager.update_progress(
            progress_key,
            int(progress_pct),
            100,
            f"Epoch {epoch}, Step {step}/{total_steps}" + (f", Loss: {loss:.4f}" if loss else "") + eta_str,
            "training",
        )

    elif entry_type == "error":
        error_msg = entry.get("message", "Unknown error")
        _update_job_status(
            job_id,
            status="failed",
            error_message=error_msg,
            completed_at=datetime.now(timezone.utc),
        )


async def cancel_training(profile_id: str, db: Session) -> bool:
    """Cancel an active training job."""
    global _active_training_process, _active_training_job_id

    # Find active job
    job = (
        db.query(DBFinetuneJob)
        .filter(
            DBFinetuneJob.profile_id == profile_id,
            DBFinetuneJob.status.in_(["pending", "preparing", "training"]),
        )
        .first()
    )

    if not job:
        return False

    # Kill subprocess
    if _active_training_process and _active_training_job_id == job.id:
        try:
            _active_training_process.terminate()
            # Give it a few seconds to clean up
            try:
                await asyncio.wait_for(_active_training_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                _active_training_process.kill()
        except ProcessLookupError:
            pass

    job.status = "cancelled"
    job.completed_at = datetime.now(timezone.utc)
    db.commit()

    task_manager = get_task_manager()
    task_manager.complete_finetune(job.id)

    return True


async def list_jobs(profile_id: str, db: Session) -> List[FinetuneJobResponse]:
    """List all finetune jobs for a profile."""
    jobs = (
        db.query(DBFinetuneJob)
        .filter_by(profile_id=profile_id)
        .order_by(DBFinetuneJob.created_at.desc())
        .all()
    )
    return [FinetuneJobResponse.model_validate(j) for j in jobs]


# ============================================
# ADAPTER MANAGEMENT
# ============================================

async def list_adapters(profile_id: str, db: Session) -> list:
    """List all trained adapters for a profile."""
    from .models import AdapterInfo

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    jobs = (
        db.query(DBFinetuneJob)
        .filter(
            DBFinetuneJob.profile_id == profile_id,
            DBFinetuneJob.status == "completed",
            DBFinetuneJob.adapter_path.isnot(None),
        )
        .order_by(DBFinetuneJob.completed_at.desc())
        .all()
    )

    adapters = []
    for job in jobs:
        # Only include if adapter files still exist on disk
        adapter_path = Path(job.adapter_path)
        if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
            continue

        adapters.append(AdapterInfo(
            job_id=job.id,
            label=job.label,
            epochs=job.epochs,
            lora_rank=job.lora_rank,
            learning_rate=job.learning_rate,
            num_samples=job.num_samples,
            completed_at=job.completed_at,
            is_active=(profile.active_adapter_path == job.adapter_path),
        ))

    return adapters


async def set_active_adapter(
    profile_id: str,
    job_id: Optional[str],
    db: Session,
) -> bool:
    """Set the active adapter for a profile, or deactivate (job_id=None)."""
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    if job_id is None:
        # Deactivate — use base model
        profile.active_adapter_path = None
        # Keep has_finetune=True so user knows adapters exist
        profile.updated_at = datetime.now(timezone.utc)
        db.commit()
        return True

    # Find the job and verify it has a valid adapter
    job = db.query(DBFinetuneJob).filter_by(
        id=job_id, profile_id=profile_id, status="completed"
    ).first()
    if not job or not job.adapter_path:
        raise ValueError(f"No completed adapter found for job {job_id}")

    adapter_path = Path(job.adapter_path)
    if not adapter_path.exists():
        raise ValueError(f"Adapter files not found at {job.adapter_path}")

    profile.active_adapter_path = job.adapter_path
    profile.has_finetune = True
    profile.updated_at = datetime.now(timezone.utc)
    db.commit()
    return True


async def update_adapter_label(
    profile_id: str,
    job_id: str,
    label: str,
    db: Session,
) -> bool:
    """Update the label for an adapter."""
    job = db.query(DBFinetuneJob).filter_by(
        id=job_id, profile_id=profile_id, status="completed"
    ).first()
    if not job:
        return False

    job.label = label
    db.commit()
    return True


async def delete_adapter(
    profile_id: str,
    job_id: str,
    db: Session,
) -> bool:
    """Delete a trained adapter (files + deactivate if active)."""
    job = db.query(DBFinetuneJob).filter_by(
        id=job_id, profile_id=profile_id, status="completed"
    ).first()
    if not job or not job.adapter_path:
        return False

    # Deactivate if this is the active adapter
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if profile and profile.active_adapter_path == job.adapter_path:
        profile.active_adapter_path = None
        profile.updated_at = datetime.now(timezone.utc)

        # Check if any other completed adapters exist
        other_adapters = (
            db.query(DBFinetuneJob)
            .filter(
                DBFinetuneJob.profile_id == profile_id,
                DBFinetuneJob.status == "completed",
                DBFinetuneJob.id != job_id,
                DBFinetuneJob.adapter_path.isnot(None),
            )
            .count()
        )
        if other_adapters == 0:
            profile.has_finetune = False

    # Delete adapter files from disk
    adapter_path = Path(job.adapter_path)
    if adapter_path.exists():
        shutil.rmtree(str(adapter_path), ignore_errors=True)

    # Clear adapter_path from the job record (keep the job for history)
    job.adapter_path = None
    db.commit()
    return True
