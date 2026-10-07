"""REST endpoints for organising voices and generated clips into folders.

One ``folders`` table backs both, discriminated by ``kind``:

  - ``voice``      — flat.  A voice folder never has a parent, because the
    voice list reads better as one level (see Folder in database/models.py).
  - ``generation`` — nests to arbitrary depth, since clips accumulate per
    project and need real hierarchy.

Deleting a folder never deletes its contents.  Members are moved to
Uncategorised and child folders are re-parented to the deleted folder's
parent, so a folder is only ever a view over items, never an owner of them.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..database.models import Folder, Generation, ProfileSample, VoiceProfile
from ..services.folders import folder_and_descendants
from ..services.profiles import _profile_to_response

router = APIRouter()

# Maps a folder kind to the table whose rows it groups.
_MEMBER_MODEL = {
    "voice": VoiceProfile,
    "generation": Generation,
}


def _get_folder_or_404(folder_id: str, db: Session) -> Folder:
    folder = db.query(Folder).filter(Folder.id == folder_id).first()
    if folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder


def _member_counts(kind: str, db: Session) -> dict[str, int]:
    """Direct member count per folder id, for one kind.

    One grouped query rather than a count per folder — the folder list is
    rendered on every voice/clip panel render.
    """
    model = _MEMBER_MODEL[kind]
    rows = (
        db.query(model.folder_id, func.count(model.id))
        .filter(model.folder_id.isnot(None))
        .group_by(model.folder_id)
        .all()
    )
    return {folder_id: count for folder_id, count in rows}


def _to_response(folder: Folder, counts: dict[str, int]) -> models.FolderResponse:
    return models.FolderResponse(
        id=folder.id,
        name=folder.name,
        kind=folder.kind,
        parent_id=folder.parent_id,
        position=folder.position,
        item_count=counts.get(folder.id, 0),
        created_at=folder.created_at,
        updated_at=folder.updated_at,
    )


@router.get("/folders", response_model=list[models.FolderResponse])
async def list_folders(kind: str = "voice", db: Session = Depends(get_db)):
    """List folders of one kind, ordered for direct rendering."""
    if kind not in _MEMBER_MODEL:
        raise HTTPException(status_code=400, detail=f"Unknown folder kind: {kind}")

    folders = (
        db.query(Folder)
        .filter(Folder.kind == kind)
        .order_by(Folder.position, Folder.name)
        .all()
    )
    counts = _member_counts(kind, db)
    return [_to_response(f, counts) for f in folders]


@router.post("/folders", response_model=models.FolderResponse)
async def create_folder(data: models.FolderCreate, db: Session = Depends(get_db)):
    """Create a folder.  Voice folders must be top-level."""
    if data.parent_id is not None:
        if data.kind == "voice":
            raise HTTPException(
                status_code=400, detail="Voice folders cannot be nested"
            )
        parent = _get_folder_or_404(data.parent_id, db)
        if parent.kind != data.kind:
            raise HTTPException(
                status_code=400, detail="Parent folder has a different kind"
            )

    folder = Folder(
        name=data.name.strip(),
        kind=data.kind,
        parent_id=data.parent_id,
        position=_next_position(data.kind, data.parent_id, db),
    )
    db.add(folder)
    db.commit()
    db.refresh(folder)
    return _to_response(folder, {})


@router.patch("/folders/{folder_id}", response_model=models.FolderResponse)
async def update_folder(
    folder_id: str,
    data: models.FolderUpdate,
    db: Session = Depends(get_db),
):
    """Rename, reposition, or reparent a folder."""
    folder = _get_folder_or_404(folder_id, db)

    if data.name is not None:
        folder.name = data.name.strip()

    if data.position is not None:
        folder.position = data.position

    if data.parent_id is not None:
        if folder.kind == "voice":
            raise HTTPException(
                status_code=400, detail="Voice folders cannot be nested"
            )
        if data.parent_id == folder_id:
            raise HTTPException(
                status_code=400, detail="A folder cannot be its own parent"
            )
        parent = _get_folder_or_404(data.parent_id, db)
        if parent.kind != folder.kind:
            raise HTTPException(
                status_code=400, detail="Parent folder has a different kind"
            )
        # Reparenting under your own descendant would detach the whole
        # subtree from the root and make it unreachable in the tree UI.
        if data.parent_id in folder_and_descendants(folder_id, db):
            raise HTTPException(
                status_code=400, detail="Cannot move a folder inside itself"
            )
        folder.parent_id = data.parent_id

    db.commit()
    db.refresh(folder)
    return _to_response(folder, _member_counts(folder.kind, db))


@router.post("/folders/{folder_id}/detach", response_model=models.FolderResponse)
async def detach_folder(folder_id: str, db: Session = Depends(get_db)):
    """Move a folder back to the root.

    Separate from PATCH because FolderUpdate.parent_id=None is
    indistinguishable from "field omitted".
    """
    folder = _get_folder_or_404(folder_id, db)
    folder.parent_id = None
    db.commit()
    db.refresh(folder)
    return _to_response(folder, _member_counts(folder.kind, db))


@router.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str, db: Session = Depends(get_db)):
    """Delete a folder, preserving everything inside it.

    Members become uncategorised; child folders rise to this folder's
    parent.  Nothing the user made is removed.
    """
    folder = _get_folder_or_404(folder_id, db)
    model = _MEMBER_MODEL[folder.kind]

    released = (
        db.query(model)
        .filter(model.folder_id == folder_id)
        .update({model.folder_id: None}, synchronize_session=False)
    )
    reparented = (
        db.query(Folder)
        .filter(Folder.parent_id == folder_id)
        .update({Folder.parent_id: folder.parent_id}, synchronize_session=False)
    )

    db.delete(folder)
    db.commit()
    return {
        "deleted": folder_id,
        "items_released": released,
        "folders_reparented": reparented,
    }


def _next_position(kind: str, parent_id: str | None, db: Session) -> int:
    """Append position — one past the highest sibling."""
    highest = (
        db.query(func.max(Folder.position))
        .filter(Folder.kind == kind, Folder.parent_id == parent_id)
        .scalar()
    )
    return 0 if highest is None else highest + 1


# ── Membership ───────────────────────────────────────────────────────


@router.put("/profiles/{profile_id}/folder", response_model=models.VoiceProfileResponse)
async def set_profile_folder(
    profile_id: str,
    data: models.FolderAssign,
    db: Session = Depends(get_db),
):
    """Move a voice into a folder, or out of one when folder_id is null."""
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    if data.folder_id is not None:
        folder = _get_folder_or_404(data.folder_id, db)
        if folder.kind != "voice":
            raise HTTPException(
                status_code=400, detail="Target folder does not hold voices"
            )

    profile.folder_id = data.folder_id
    db.commit()
    db.refresh(profile)

    generation_count = (
        db.query(func.count(Generation.id))
        .filter(Generation.profile_id == profile.id)
        .scalar()
    )
    sample_count = (
        db.query(func.count(ProfileSample.id))
        .filter(ProfileSample.profile_id == profile.id)
        .scalar()
    )
    return _profile_to_response(profile, generation_count, sample_count)


@router.put("/history/{generation_id}/folder")
async def set_generation_folder(
    generation_id: str,
    data: models.FolderAssign,
    db: Session = Depends(get_db),
):
    """Move a clip into a folder, or out of one when folder_id is null."""
    generation = db.query(Generation).filter(Generation.id == generation_id).first()
    if generation is None:
        raise HTTPException(status_code=404, detail="Generation not found")

    if data.folder_id is not None:
        folder = _get_folder_or_404(data.folder_id, db)
        if folder.kind != "generation":
            raise HTTPException(
                status_code=400, detail="Target folder does not hold clips"
            )

    generation.folder_id = data.folder_id
    db.commit()
    return {"id": generation_id, "folder_id": generation.folder_id}
