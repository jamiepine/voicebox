"""Avatar uploads are size-capped and limited to image extensions."""

import io
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.app  # noqa: F401
from backend.routes import profiles as profiles_route


def _upload(name: str, size: int) -> UploadFile:
    return UploadFile(file=io.BytesIO(b"\0" * size), filename=name)


@pytest.fixture
def captured_upload(monkeypatch):
    captured = {}

    async def fake_upload_avatar(profile_id, tmp_path, db):
        captured["size"] = Path(tmp_path).stat().st_size
        captured["suffix"] = Path(tmp_path).suffix
        return {"id": profile_id}

    monkeypatch.setattr(profiles_route.profiles, "upload_avatar", fake_upload_avatar)
    return captured


@pytest.mark.asyncio
async def test_avatar_over_limit_is_rejected_with_413(captured_upload, monkeypatch, tmp_path):
    monkeypatch.setattr(profiles_route.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(HTTPException) as exc:
        await profiles_route.upload_profile_avatar(
            "p1", _upload("big.png", profiles_route.AVATAR_MAX_FILE_SIZE + 1), None
        )
    assert exc.value.status_code == 413
    assert captured_upload == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_avatar_at_limit_is_accepted(captured_upload):
    await profiles_route.upload_profile_avatar(
        "p1", _upload("ok.JPG", profiles_route.AVATAR_MAX_FILE_SIZE), None
    )
    assert captured_upload == {"size": profiles_route.AVATAR_MAX_FILE_SIZE, "suffix": ".jpg"}


@pytest.mark.asyncio
async def test_avatar_with_non_image_extension_is_rejected(captured_upload):
    with pytest.raises(HTTPException) as exc:
        await profiles_route.upload_profile_avatar("p1", _upload("shell.php", 10), None)
    assert exc.value.status_code == 400
    assert captured_upload == {}


@pytest.mark.asyncio
async def test_avatar_without_extension_defaults_to_png(captured_upload):
    await profiles_route.upload_profile_avatar("p1", _upload("avatar", 10), None)
    assert captured_upload["suffix"] == ".png"
