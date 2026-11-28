"""Voice-sample uploads are size-capped and always clean up their temp file."""

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
def captured_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(profiles_route.tempfile, "tempdir", str(tmp_path))
    captured = {}

    async def fake_add_profile_sample(profile_id, path, reference_text, db):
        captured["size"] = Path(path).stat().st_size
        captured["suffix"] = Path(path).suffix
        return {"id": "s1", "profile_id": profile_id}

    monkeypatch.setattr(profiles_route.profiles, "add_profile_sample", fake_add_profile_sample)
    return captured


@pytest.mark.asyncio
async def test_sample_over_limit_is_rejected_with_413_and_cleaned_up(captured_sample, tmp_path):
    with pytest.raises(HTTPException) as exc:
        await profiles_route.add_profile_sample(
            "p1", _upload("big.wav", profiles_route.SAMPLE_MAX_FILE_SIZE + 1), "hello", None
        )
    assert exc.value.status_code == 413
    assert captured_sample == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_sample_over_limit_closes_temp_file_before_unlink(monkeypatch, tmp_path):
    monkeypatch.setattr(profiles_route.tempfile, "tempdir", str(tmp_path))
    real_ntf = profiles_route.tempfile.NamedTemporaryFile
    handles = []

    def tracking_ntf(*args, **kwargs):
        handle = real_ntf(*args, **kwargs)
        handles.append(handle)
        return handle

    real_unlink = Path.unlink

    def strict_unlink(self, missing_ok=False):
        if any(h.name == str(self) and not h.closed for h in handles):
            raise PermissionError(f"{self} is still open")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(profiles_route.tempfile, "NamedTemporaryFile", tracking_ntf)
    monkeypatch.setattr(Path, "unlink", strict_unlink)

    with pytest.raises(HTTPException) as exc:
        await profiles_route.add_profile_sample(
            "p1", _upload("big.wav", profiles_route.SAMPLE_MAX_FILE_SIZE + 1), "hello", None
        )
    assert exc.value.status_code == 413
    assert handles and all(h.closed for h in handles)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_sample_at_limit_is_accepted_and_cleaned_up(captured_sample, tmp_path):
    await profiles_route.add_profile_sample(
        "p1", _upload("ok.FLAC", profiles_route.SAMPLE_MAX_FILE_SIZE), "hello", None
    )
    assert captured_sample == {"size": profiles_route.SAMPLE_MAX_FILE_SIZE, "suffix": ".flac"}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_sample_rejected_by_service_returns_400(monkeypatch, tmp_path):
    monkeypatch.setattr(profiles_route.tempfile, "tempdir", str(tmp_path))

    async def reject(profile_id, path, reference_text, db):
        raise ValueError("Audio too short")

    monkeypatch.setattr(profiles_route.profiles, "add_profile_sample", reject)
    with pytest.raises(HTTPException) as exc:
        await profiles_route.add_profile_sample("p1", _upload("short.wav", 10), "hello", None)
    assert exc.value.status_code == 400
    assert list(tmp_path.iterdir()) == []
