"""
Regression tests for DELETE /profiles/{id} cleaning up everything it owns.

``delete_profile`` used to remove only the profile row, its samples, and the
profile directory. Generations carry a non-null FK to the profile and
``list_generations`` inner-joins profiles, so every generation made with a
deleted profile turned into a row the UI could never show plus a ``.wav`` in
``data/generations`` the user could never reclaim — unbounded disk growth with
no way out. Version rows, story items, and channel mappings leaked the same way.

Usage:
    python -m pytest backend/tests/test_profile_delete_cascade.py -v
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Repo root on sys.path so ``backend`` imports as a package (the services use
# package-relative imports).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend import config
from backend.database import (
    Base,
    CaptureSettings,
    Generation,
    GenerationVersion,
    MCPClientBinding,
    ProfileChannelMapping,
    ProfileSample,
    Story,
    StoryItem,
    VoiceProfile,
)
from backend.models import HistoryQuery
from backend.services import history, profiles


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A session backed by a temp SQLite file, with the data dir redirected."""
    monkeypatch.setattr(config, "_data_dir", tmp_path)

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()

    yield session

    session.close()


def _write_wav(name: str) -> tuple[Path, str]:
    """Create a placeholder audio file and return (abs path, stored path)."""
    path = config.get_generations_dir() / name
    path.write_bytes(b"RIFF placeholder")
    return path, f"generations/{name}"


def _make_profile(db, profile_id: str = "profile-1") -> VoiceProfile:
    profile = VoiceProfile(id=profile_id, name=f"Voice {profile_id}", language="en")
    db.add(profile)
    db.commit()
    return profile


@pytest.mark.asyncio
async def test_delete_profile_removes_generations_and_audio(db):
    """Generations, versions, and their files go with the profile."""
    _make_profile(db)
    gen_file, gen_stored = _write_wav("gen.wav")
    version_file, version_stored = _write_wav("gen-reverb.wav")

    db.add(Generation(id="gen-1", profile_id="profile-1", text="hello", audio_path=gen_stored))
    db.add(
        GenerationVersion(
            id="version-1",
            generation_id="gen-1",
            label="Reverb",
            audio_path=version_stored,
        )
    )
    db.commit()

    assert await profiles.delete_profile("profile-1", db) is True

    assert db.query(Generation).count() == 0
    assert db.query(GenerationVersion).count() == 0
    assert not gen_file.exists()
    assert not version_file.exists()


@pytest.mark.asyncio
async def test_orphaned_generations_are_invisible_in_history(db):
    """The history join hides orphans, so leaving them behind strands them."""
    _make_profile(db)
    _, gen_stored = _write_wav("gen.wav")
    db.add(Generation(id="gen-1", profile_id="profile-1", text="hello", audio_path=gen_stored))
    db.commit()

    assert (await history.list_generations(HistoryQuery(limit=50, offset=0), db)).total == 1

    await profiles.delete_profile("profile-1", db)

    listed = await history.list_generations(HistoryQuery(limit=50, offset=0), db)
    assert listed.total == 0
    assert db.query(Generation).count() == 0, "generation is unreachable but still on disk"


@pytest.mark.asyncio
async def test_delete_profile_removes_story_items(db):
    """Story items reference generations; the story query would hide leftovers."""
    _make_profile(db)
    _, gen_stored = _write_wav("gen.wav")
    db.add(Generation(id="gen-1", profile_id="profile-1", text="hello", audio_path=gen_stored))
    db.add(Story(id="story-1", name="Chapter one"))
    db.add(StoryItem(id="item-1", story_id="story-1", generation_id="gen-1"))
    db.commit()

    await profiles.delete_profile("profile-1", db)

    assert db.query(StoryItem).count() == 0
    assert db.query(Story).count() == 1, "the story itself must survive"


@pytest.mark.asyncio
async def test_delete_profile_clears_references_to_it(db):
    """Samples, channel mappings, and default-voice pointers are cleaned up."""
    _make_profile(db)
    db.add(ProfileSample(id="sample-1", profile_id="profile-1", audio_path="p.wav", reference_text="hi"))
    db.add(ProfileChannelMapping(profile_id="profile-1", channel_id="channel-1"))
    db.add(MCPClientBinding(client_id="claude-code", profile_id="profile-1"))
    db.add(CaptureSettings(id=1, default_playback_voice_id="profile-1"))
    db.commit()

    await profiles.delete_profile("profile-1", db)

    assert db.query(ProfileSample).count() == 0
    assert db.query(ProfileChannelMapping).count() == 0
    assert db.query(MCPClientBinding).filter_by(client_id="claude-code").one().profile_id is None
    assert db.query(CaptureSettings).filter_by(id=1).one().default_playback_voice_id is None


@pytest.mark.asyncio
async def test_delete_profile_survives_unremovable_audio(db, monkeypatch):
    """A file locked by playback must not abort the delete half-way through."""
    _make_profile(db)
    gen_file, gen_stored = _write_wav("gen.wav")
    db.add(Generation(id="gen-1", profile_id="profile-1", text="hello", audio_path=gen_stored))
    db.commit()

    real_unlink = Path.unlink

    def refuse_locked_file(self, *args, **kwargs):
        if self == gen_file:
            raise OSError("file is in use by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_locked_file)

    assert await profiles.delete_profile("profile-1", db) is True
    assert db.query(VoiceProfile).count() == 0
    assert db.query(Generation).count() == 0


@pytest.mark.asyncio
async def test_delete_profile_leaves_other_profiles_alone(db):
    """Cleanup is scoped to the deleted profile."""
    _make_profile(db, "profile-1")
    _make_profile(db, "profile-2")
    _, keep_stored = _write_wav("keep.wav")
    _, drop_stored = _write_wav("drop.wav")
    db.add(Generation(id="gen-keep", profile_id="profile-2", text="keep", audio_path=keep_stored))
    db.add(Generation(id="gen-drop", profile_id="profile-1", text="drop", audio_path=drop_stored))
    db.add(ProfileChannelMapping(profile_id="profile-2", channel_id="channel-1"))
    db.commit()

    await profiles.delete_profile("profile-1", db)

    assert [g.id for g in db.query(Generation).all()] == ["gen-keep"]
    assert db.query(ProfileChannelMapping).count() == 1


@pytest.mark.asyncio
async def test_delete_missing_profile_returns_false(db):
    assert await profiles.delete_profile("nope", db) is False
