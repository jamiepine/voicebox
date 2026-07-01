"""
The cloud sync engine: encrypted backup + multi-device restore.

Walks the local store (SQLite rows + audio files), maps each entity onto the
cloud's object model, and drives the push/pull loop against the blind server.
Everything crosses the wire as VBX1 ciphertext (``cloud_crypto``); the server
only ever learns kinds, ids, sizes, and hashes.

Mapping (cloud repo ``docs/DESIGN.md`` §5):

| local entity                          | kind       | record (encrypted JSON)   | assets            |
| ------------------------------------- | ---------- | ------------------------- | ----------------- |
| ``captures`` row + wav                | capture    | the row                   | the capture audio |
| ``generations`` row + version wavs    | generation | the row + version rows    | each version wav  |
| ``profiles`` row + samples + avatar   | profile    | the row + sample rows     | sample wavs, avatar |
| ``capture_settings`` / ``generation_settings`` | settings | the row          | —                 |

Path columns are stored storage-relative inside the (encrypted) record, so a
restore re-anchors them under the destination machine's data dir.

Change detection: envelopes are randomized, so ``CloudSyncState`` keeps the
plaintext fingerprint (did the content actually change?) alongside the
ciphertext hash the server holds (re-declare unchanged blobs without
re-encrypting). Conflicts are last-writer-wins per object, matching §6 —
push runs before pull, so local edits are declared before remote state lands.

AAD binding: records are bound to ``(clientId, "record", version)`` and
re-encrypted on every version bump. Asset blobs are bound to
``(clientId, "asset:<clientAssetId>", 1)`` — assets are content-addressed and
practically immutable (audio never changes in place), so their slot binding
doesn't chase the object version.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from .. import config
from ..database import (
    Capture,
    CaptureSettings,
    CloudSettings as DBCloudSettings,
    CloudSyncState,
    Generation,
    GenerationSettings,
    GenerationVersion,
    ProfileSample,
    VoiceProfile,
)
from . import cloud_account, cloud_crypto
from .cloud_api import CloudApiClient

logger = logging.getLogger(__name__)

_ASSET_AAD_VERSION = 1


class CloudSyncError(Exception):
    """Sync could not run or an object failed to round-trip."""


# ---------------------------------------------------------------------------
# Local object collection (push side)


@dataclass(frozen=True)
class LocalAsset:
    client_asset_id: str
    role: str  # audio | version | sample | avatar
    path: Path


@dataclass(frozen=True)
class LocalObject:
    kind: str
    client_id: str
    record: dict
    assets: list[LocalAsset] = field(default_factory=list)


_PATH_COLUMNS = {"audio_path", "avatar_path"}


def _row_to_record(row) -> dict:
    """All mapped columns as JSON-safe values; paths storage-relative,
    datetimes ISO-8601."""
    record: dict = {}
    for column in row.__mapper__.columns:
        value = getattr(row, column.key)
        if value is None:
            record[column.key] = None
        elif column.key in _PATH_COLUMNS:
            # Rows normally hold data-dir-relative paths already; only rebase
            # absolute ones. (to_storage_path on a relative value would resolve
            # it against the CWD and corrupt it.)
            record[column.key] = config.to_storage_path(value) if Path(value).is_absolute() else value
        elif isinstance(value, datetime):
            record[column.key] = value.isoformat()
        else:
            record[column.key] = value
    return record


def _is_datetime_column(column) -> bool:
    try:
        return column.type.python_type is datetime
    except NotImplementedError:  # e.g. JSON columns don't declare a python_type
        return False


def _record_to_row(model, record: dict, existing=None):
    """Build or update a model instance from a record dict."""
    row = existing if existing is not None else model()
    for column in row.__mapper__.columns:
        if column.key not in record:
            continue
        value = record[column.key]
        if isinstance(value, str) and _is_datetime_column(column):
            value = datetime.fromisoformat(value)
        setattr(row, column.key, value)
    return row


def _existing_path(value: str | None) -> Path | None:
    resolved = config.resolve_storage_path(value)
    return resolved if resolved is not None and resolved.exists() else None


def _collect_captures(db: Session) -> list[LocalObject]:
    objects = []
    for row in db.query(Capture).all():
        assets = []
        if (path := _existing_path(row.audio_path)) is not None:
            assets.append(LocalAsset(client_asset_id=row.id, role="audio", path=path))
        objects.append(LocalObject(kind="capture", client_id=row.id, record=_row_to_record(row), assets=assets))
    return objects


def _collect_generations(db: Session) -> list[LocalObject]:
    objects = []
    for row in db.query(Generation).filter(Generation.status == "completed").all():
        record = _row_to_record(row)
        assets = []
        if (path := _existing_path(row.audio_path)) is not None:
            assets.append(LocalAsset(client_asset_id=row.id, role="audio", path=path))
        versions = db.query(GenerationVersion).filter(GenerationVersion.generation_id == row.id).all()
        record["versions"] = [_row_to_record(v) for v in versions]
        for version in versions:
            if (path := _existing_path(version.audio_path)) is not None:
                assets.append(LocalAsset(client_asset_id=version.id, role="version", path=path))
        objects.append(LocalObject(kind="generation", client_id=row.id, record=record, assets=assets))
    return objects


def _collect_profiles(db: Session) -> list[LocalObject]:
    objects = []
    for row in db.query(VoiceProfile).all():
        record = _row_to_record(row)
        assets = []
        if (path := _existing_path(row.avatar_path)) is not None:
            assets.append(LocalAsset(client_asset_id=f"{row.id}-avatar", role="avatar", path=path))
        samples = db.query(ProfileSample).filter(ProfileSample.profile_id == row.id).all()
        record["samples"] = [_row_to_record(s) for s in samples]
        for sample in samples:
            if (path := _existing_path(sample.audio_path)) is not None:
                assets.append(LocalAsset(client_asset_id=sample.id, role="sample", path=path))
        objects.append(LocalObject(kind="profile", client_id=row.id, record=record, assets=assets))
    return objects


def _collect_settings(db: Session) -> list[LocalObject]:
    objects = []
    for client_id, model in (("capture_settings", CaptureSettings), ("generation_settings", GenerationSettings)):
        row = db.query(model).first()
        if row is not None:
            objects.append(LocalObject(kind="settings", client_id=client_id, record=_row_to_record(row)))
    return objects


def collect_local_objects(db: Session) -> list[LocalObject]:
    return _collect_captures(db) + _collect_generations(db) + _collect_profiles(db) + _collect_settings(db)


# ---------------------------------------------------------------------------
# Applying pulled records (pull side)


def _write_asset(record_path_value: str | None, data: bytes) -> None:
    resolved = config.resolve_storage_path(record_path_value)
    if resolved is None:
        return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(data)


def _apply_children(db: Session, model, parent_filter, child_records: list[dict], blobs: dict[str, bytes]) -> None:
    """Upsert child rows (versions/samples) by id; drop local children the
    record no longer contains; write any pulled audio next to them."""
    wanted = {child["id"] for child in child_records}
    for stale in db.query(model).filter(parent_filter).all():
        if stale.id not in wanted:
            db.delete(stale)
    for child in child_records:
        existing = db.query(model).filter(model.id == child["id"]).first()
        row = _record_to_row(model, child, existing)
        if existing is None:
            db.add(row)
        if child["id"] in blobs:
            _write_asset(child.get("audio_path"), blobs[child["id"]])


def _apply_capture(db: Session, client_id: str, record: dict, blobs: dict[str, bytes]) -> None:
    existing = db.query(Capture).filter(Capture.id == client_id).first()
    row = _record_to_row(Capture, record, existing)
    if existing is None:
        db.add(row)
    if client_id in blobs:
        _write_asset(record.get("audio_path"), blobs[client_id])


def _apply_generation(db: Session, client_id: str, record: dict, blobs: dict[str, bytes]) -> None:
    record = dict(record)
    versions = record.pop("versions", [])
    existing = db.query(Generation).filter(Generation.id == client_id).first()
    row = _record_to_row(Generation, record, existing)
    if existing is None:
        db.add(row)
    if client_id in blobs:
        _write_asset(record.get("audio_path"), blobs[client_id])
    _apply_children(db, GenerationVersion, GenerationVersion.generation_id == client_id, versions, blobs)


def _apply_profile(db: Session, client_id: str, record: dict, blobs: dict[str, bytes]) -> None:
    record = dict(record)
    samples = record.pop("samples", [])
    existing = db.query(VoiceProfile).filter(VoiceProfile.id == client_id).first()
    row = _record_to_row(VoiceProfile, record, existing)
    if existing is None:
        db.add(row)
    if f"{client_id}-avatar" in blobs:
        _write_asset(record.get("avatar_path"), blobs[f"{client_id}-avatar"])
    _apply_children(db, ProfileSample, ProfileSample.profile_id == client_id, samples, blobs)


def _apply_settings(db: Session, client_id: str, record: dict) -> None:
    model = CaptureSettings if client_id == "capture_settings" else GenerationSettings
    existing = db.query(model).first()
    row = _record_to_row(model, record, existing)
    if existing is None:
        db.add(row)


def _apply_record(db: Session, kind: str, client_id: str, record: dict, blobs: dict[str, bytes]) -> None:
    if kind == "capture":
        _apply_capture(db, client_id, record, blobs)
    elif kind == "generation":
        _apply_generation(db, client_id, record, blobs)
    elif kind == "profile":
        _apply_profile(db, client_id, record, blobs)
    elif kind == "settings":
        _apply_settings(db, client_id, record)
    else:
        raise CloudSyncError(f"unknown object kind {kind!r}")


def _delete_local(db: Session, kind: str, client_id: str) -> None:
    if kind == "capture":
        db.query(Capture).filter(Capture.id == client_id).delete()
    elif kind == "generation":
        db.query(GenerationVersion).filter(GenerationVersion.generation_id == client_id).delete()
        db.query(Generation).filter(Generation.id == client_id).delete()
    elif kind == "profile":
        db.query(ProfileSample).filter(ProfileSample.profile_id == client_id).delete()
        db.query(VoiceProfile).filter(VoiceProfile.id == client_id).delete()
    # settings singletons are never deleted


# ---------------------------------------------------------------------------
# The engine


@dataclass
class SyncReport:
    pushed: int = 0
    pushed_deletes: int = 0
    pulled: int = 0
    pulled_deletes: int = 0
    cursor: int = 0


def _canonical(record: dict) -> bytes:
    """Canonical bytes for change detection. ``updated_at`` is excluded (at the
    top level and in embedded child rows): its ``onupdate`` trigger can bump it
    as a side effect of *applying* a pulled record, and letting that feed back
    into the fingerprint would bounce an already-synced object back and forth.
    The field still syncs — it just doesn't count as a change by itself."""
    stripped = {k: v for k, v in record.items() if k != "updated_at"}
    for key, value in stripped.items():
        if isinstance(value, list):
            stripped[key] = [
                {k: v for k, v in item.items() if k != "updated_at"} if isinstance(item, dict) else item
                for item in value
            ]
    return json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get_state(db: Session, kind: str, client_id: str) -> CloudSyncState | None:
    return db.query(CloudSyncState).filter(CloudSyncState.kind == kind, CloudSyncState.client_id == client_id).first()


def _settings_row(db: Session) -> DBCloudSettings:
    row = db.query(DBCloudSettings).filter(DBCloudSettings.id == 1).first()
    if row is None:
        raise CloudSyncError("not connected to Voicebox Cloud")
    return row


async def _push_object(
    client: CloudApiClient,
    db: Session,
    master_key: bytes,
    obj: LocalObject,
    state: CloudSyncState | None,
) -> bool:
    """Push one object if it changed. Returns True when a push happened."""
    record_payload = json.dumps(obj.record, sort_keys=True, separators=(",", ":")).encode()
    record_fp = _fingerprint(_canonical(obj.record))
    known_assets: dict = json.loads(state.assets_json) if state else {}

    asset_plain: dict[str, bytes] = {}
    asset_fps: dict[str, str] = {}
    for asset in obj.assets:
        data = asset.path.read_bytes()
        asset_plain[asset.client_asset_id] = data
        asset_fps[asset.client_asset_id] = _fingerprint(data)

    unchanged = (
        state is not None
        and state.server_object_id is not None
        and state.record_fingerprint == record_fp
        and {k: v["fingerprint"] for k, v in known_assets.items()} == asset_fps
    )
    if unchanged:
        return False

    version = (state.version + 1) if state is not None else 1
    record_env = cloud_crypto.encrypt_blob(
        record_payload, master_key, object_id=obj.client_id, role="record", version=version
    )

    descriptors = []
    envelopes: dict[str, bytes] = {}
    next_assets: dict[str, dict] = {}
    for asset in obj.assets:
        caid = asset.client_asset_id
        known = known_assets.get(caid)
        if known and known["fingerprint"] == asset_fps[caid]:
            # Content unchanged: re-declare the ciphertext the server holds.
            entry = {"role": asset.role, "fingerprint": asset_fps[caid], "hash": known["hash"], "size": known["size"]}
        else:
            envelope = cloud_crypto.encrypt_blob(
                asset_plain[caid],
                master_key,
                object_id=obj.client_id,
                role=f"asset:{caid}",
                version=_ASSET_AAD_VERSION,
            )
            envelopes[caid] = envelope
            entry = {
                "role": asset.role,
                "fingerprint": asset_fps[caid],
                "hash": _fingerprint(envelope),
                "size": len(envelope),
            }
        next_assets[caid] = entry
        descriptors.append({"role": asset.role, "clientAssetId": caid, "hash": entry["hash"], "size": entry["size"]})

    pushed = await client.push_object(
        kind=obj.kind,
        client_id=obj.client_id,
        version=version,
        record={"hash": _fingerprint(record_env), "size": len(record_env)},
        assets=descriptors,
    )
    uploads = {u["for"]: u["url"] for u in pushed["uploads"]}
    if "record" in uploads:
        await client.upload_blob(uploads["record"], record_env)
    for caid, envelope in envelopes.items():
        url = uploads.get(f"asset:{caid}")
        if url:
            await client.upload_blob(url, envelope)
    await client.commit_object(pushed["objectId"])

    if state is None:
        state = CloudSyncState(kind=obj.kind, client_id=obj.client_id)
        db.add(state)
    state.server_object_id = pushed["objectId"]
    state.version = version
    state.record_fingerprint = record_fp
    state.record_hash = _fingerprint(record_env)
    state.record_size = len(record_env)
    state.assets_json = json.dumps(next_assets)
    state.last_synced_at = datetime.utcnow()
    return True


async def _push_all(client: CloudApiClient, db: Session, master_key: bytes, report: SyncReport) -> None:
    local = collect_local_objects(db)
    local_ids = {(o.kind, o.client_id) for o in local}

    for obj in local:
        if await _push_object(client, db, master_key, obj, _get_state(db, obj.kind, obj.client_id)):
            report.pushed += 1
            db.commit()

    # Local deletions: state rows whose entity no longer exists → tombstone.
    for state in db.query(CloudSyncState).all():
        if (state.kind, state.client_id) not in local_ids and state.kind != "settings":
            if state.server_object_id:
                await client.delete_object(state.server_object_id)
            db.delete(state)
            report.pushed_deletes += 1
            db.commit()


async def _pull_changes(client: CloudApiClient, db: Session, master_key: bytes, report: SyncReport) -> None:
    settings = _settings_row(db)
    cursor = settings.sync_cursor or 0

    while True:
        page = await client.get_changes(since=cursor)
        for change in page["changes"]:
            kind, client_id = change["kind"], change["clientId"]
            state = _get_state(db, kind, client_id)

            if change["deleted"]:
                if state is not None:
                    _delete_local(db, kind, client_id)
                    db.delete(state)
                    report.pulled_deletes += 1
            # Our own pushes echo back through the feed; the stored ciphertext
            # hash identifies them as already applied.
            elif change["record"] and (state is None or state.record_hash != change["record"]["hash"]):
                await _apply_change(client, db, master_key, change, state)
                report.pulled += 1

            cursor = change["seq"]

        settings.sync_cursor = cursor
        db.commit()
        report.cursor = cursor
        if not page["hasMore"]:
            break


async def _apply_change(
    client: CloudApiClient,
    db: Session,
    master_key: bytes,
    change: dict,
    state: CloudSyncState | None,
) -> None:
    kind, client_id = change["kind"], change["clientId"]
    record_cipher = await client.download_blob(change["record"]["url"])
    record = json.loads(
        cloud_crypto.decrypt_blob(
            record_cipher, master_key, object_id=client_id, role="record", version=change["version"]
        )
    )

    known_assets: dict = json.loads(state.assets_json) if state else {}
    blobs: dict[str, bytes] = {}
    next_assets: dict[str, dict] = {}
    for asset in change["assets"]:
        caid = asset["clientAssetId"]
        known = known_assets.get(caid)
        if known and known["hash"] == asset["hash"]:
            next_assets[caid] = known
            continue  # ciphertext we already hold locally
        if not asset["url"]:
            continue  # declared but never uploaded; skip until it lands
        cipher = await client.download_blob(asset["url"])
        plain = cloud_crypto.decrypt_blob(
            cipher, master_key, object_id=client_id, role=f"asset:{caid}", version=_ASSET_AAD_VERSION
        )
        blobs[caid] = plain
        next_assets[caid] = {
            "role": asset["role"],
            "fingerprint": _fingerprint(plain),
            "hash": asset["hash"],
            "size": asset["size"],
        }

    _apply_record(db, kind, client_id, record, blobs)

    if state is None:
        state = CloudSyncState(kind=kind, client_id=client_id)
        db.add(state)
    state.server_object_id = change["id"]
    state.version = change["version"]
    state.record_fingerprint = _fingerprint(_canonical(record))
    state.record_hash = change["record"]["hash"]
    state.record_size = change["record"]["size"]
    state.assets_json = json.dumps(next_assets)
    state.last_synced_at = datetime.utcnow()


async def run_sync(db: Session) -> SyncReport:
    """One full sync: push local changes, then pull and apply remote ones."""
    settings = _settings_row(db)
    master_key = cloud_account.load_master_key(db)
    report = SyncReport()

    async with CloudApiClient(config.get_cloud_api_url(), settings.api_key) as client:
        await _push_all(client, db, master_key, report)
        await _pull_changes(client, db, master_key, report)

    logger.info(
        "cloud sync: pushed %d (+%d deletes), pulled %d (+%d deletes), cursor %d",
        report.pushed,
        report.pushed_deletes,
        report.pulled,
        report.pulled_deletes,
        report.cursor,
    )
    return report
