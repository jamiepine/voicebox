"""Tests for the cloud sync engine (services/cloud_sync.py).

Simulates two installs ("machines") of the desktop app — each with its own
SQLite database, data directory, and keychain — syncing through the in-process
FakeCloud. Covers the full backup → restore path, incremental pushes, deletes,
and the blindness invariant (nothing plaintext ever lands in server storage).
"""

from datetime import datetime

import keyring
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database.models import (
    Base,
    Capture,
    CaptureSettings,
    CloudSettings,
    Generation,
    GenerationVersion,
    ProfileSample,
    VoiceProfile,
)
from backend.services import cloud_account, cloud_sync
from backend.services.cloud_api import CloudApiClient
from backend.tests.fake_cloud import FakeCloud

AUDIO_A = b"RIFFfake-capture-audio" + b"\x11" * 4000
AUDIO_B = b"RIFFfake-generation-audio" + b"\x22" * 4000
AUDIO_C = b"RIFFfake-version-audio" + b"\x33" * 4000
AUDIO_D = b"RIFFfake-sample-audio" + b"\x44" * 4000
AVATAR = b"\x89PNGfake-avatar" + b"\x55" * 500


class Install:
    """One simulated machine: its own DB, data dir, and keychain store."""

    def __init__(self, root, name: str):
        self.data_dir = root / name
        self.data_dir.mkdir()
        self.keychain: dict[tuple[str, str], str] = {}
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()
        self.db.add(
            CloudSettings(
                id=1,
                api_key=f"voicebox_{name}",
                device_name=name,
                account_user_id="user-1",
                connected_at=datetime(2026, 7, 1),
            )
        )
        self.db.commit()

    def activate(self, monkeypatch):
        """Point global config + keychain at this machine."""
        config.set_data_dir(self.data_dir)
        store = self.keychain
        monkeypatch.setattr(keyring, "get_password", lambda s, u: store.get((s, u)))
        monkeypatch.setattr(keyring, "set_password", lambda s, u, p: store.__setitem__((s, u), p))
        monkeypatch.setattr(keyring, "delete_password", lambda s, u: store.pop((s, u), None))

    def write_file(self, relative: str, data: bytes) -> str:
        path = self.data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)


@pytest.fixture
def cloud(monkeypatch):
    fake = FakeCloud()
    monkeypatch.setattr(
        cloud_sync,
        "CloudApiClient",
        lambda url, key: CloudApiClient(url, key, transport=fake.transport()),
    )
    monkeypatch.setattr(
        cloud_account,
        "_client",
        lambda row: CloudApiClient("http://cloud.test", row.api_key, transport=fake.transport()),
    )
    return fake


def seed_content(install: Install) -> None:
    db = install.db
    profile = VoiceProfile(
        id="prof-1",
        name="Morgan",
        description="test voice",
        language="en",
        avatar_path=install.write_file("profiles/prof-1/avatar.png", AVATAR),
        personality="dry wit",
    )
    db.add(profile)
    db.add(
        ProfileSample(
            id="samp-1",
            profile_id="prof-1",
            audio_path=install.write_file("profiles/prof-1/samples/samp-1.wav", AUDIO_D),
            reference_text="hello there",
        )
    )
    db.add(
        Capture(
            id="cap-1",
            audio_path=install.write_file("captures/cap-1.wav", AUDIO_A),
            source="dictation",
            language="en",
            transcript_raw="the raw transcript",
            transcript_refined="the refined transcript",
        )
    )
    db.add(
        Generation(
            id="gen-1",
            profile_id="prof-1",
            text="hello world",
            audio_path=install.write_file("generations/gen-1.wav", AUDIO_B),
            status="completed",
            source="manual",
        )
    )
    db.add(
        GenerationVersion(
            id="ver-1",
            generation_id="gen-1",
            label="Take 2",
            audio_path=install.write_file("generations/ver-1.wav", AUDIO_C),
        )
    )
    db.add(CaptureSettings(id=1, stt_model="turbo", language="auto"))
    db.commit()


async def connect(install: Install, monkeypatch, phrase: str | None = None) -> str | None:
    install.activate(monkeypatch)
    result = await cloud_account.setup_device(install.db)
    if phrase is not None:
        await cloud_account.restore_with_phrase(install.db, phrase)
    return result


class TestSyncEngine:
    async def test_backup_then_restore_on_second_machine(self, cloud, tmp_path, monkeypatch):
        a = Install(tmp_path, "machine-a")
        phrase = await connect(a, monkeypatch)
        seed_content(a)

        report = await cloud_sync.run_sync(a.db)
        assert report.pushed == 4  # capture, generation, profile, capture_settings
        assert report.pulled == 0  # own echoes are recognized by ciphertext hash

        # Server blindness: every stored blob is a VBX1 envelope, no plaintext.
        assert cloud.storage
        for blob in cloud.storage.values():
            assert blob[:4] == b"VBX1"
            assert b"transcript" not in blob
            assert AUDIO_A not in blob
        for body in cloud.seen_bodies:
            assert b"the raw transcript" not in body
            assert b"Morgan" not in body

        # Fresh machine: restore identity from phrase, then pull everything.
        b = Install(tmp_path, "machine-b")
        await connect(b, monkeypatch, phrase=phrase)
        report_b = await cloud_sync.run_sync(b.db)
        assert report_b.pulled == report.pushed

        cap = b.db.query(Capture).one()
        assert cap.id == "cap-1"
        assert cap.transcript_raw == "the raw transcript"
        assert (b.data_dir / "captures/cap-1.wav").read_bytes() == AUDIO_A

        prof = b.db.query(VoiceProfile).one()
        assert prof.name == "Morgan"
        assert prof.personality == "dry wit"
        assert (b.data_dir / "profiles/prof-1/avatar.png").read_bytes() == AVATAR
        samp = b.db.query(ProfileSample).one()
        assert samp.reference_text == "hello there"
        assert (b.data_dir / "profiles/prof-1/samples/samp-1.wav").read_bytes() == AUDIO_D

        gen = b.db.query(Generation).one()
        assert gen.text == "hello world"
        assert (b.data_dir / "generations/gen-1.wav").read_bytes() == AUDIO_B
        ver = b.db.query(GenerationVersion).one()
        assert ver.label == "Take 2"
        assert (b.data_dir / "generations/ver-1.wav").read_bytes() == AUDIO_C

        settings = b.db.query(CaptureSettings).one()
        assert settings.stt_model == "turbo"

        # Second sync on B is a no-op in both directions.
        report_b2 = await cloud_sync.run_sync(b.db)
        assert (report_b2.pushed, report_b2.pulled) == (0, 0)

    async def test_incremental_edit_propagates(self, cloud, tmp_path, monkeypatch):
        a = Install(tmp_path, "machine-a")
        phrase = await connect(a, monkeypatch)
        seed_content(a)
        await cloud_sync.run_sync(a.db)

        b = Install(tmp_path, "machine-b")
        await connect(b, monkeypatch, phrase=phrase)
        await cloud_sync.run_sync(b.db)

        # Edit on A: only the capture should push, and only its record blob
        # should re-upload (the audio is unchanged).
        a.activate(monkeypatch)
        blobs_before = dict(cloud.storage)
        cap = a.db.query(Capture).one()
        cap.transcript_refined = "edited on machine A"
        a.db.commit()
        report_a = await cloud_sync.run_sync(a.db)
        assert report_a.pushed == 1
        changed_keys = [k for k, v in cloud.storage.items() if blobs_before.get(k) != v]
        assert changed_keys == [k for k in changed_keys if k.endswith("/record")]

        b.activate(monkeypatch)
        report_b = await cloud_sync.run_sync(b.db)
        assert report_b.pulled == 1
        assert b.db.query(Capture).one().transcript_refined == "edited on machine A"

    async def test_delete_propagates(self, cloud, tmp_path, monkeypatch):
        a = Install(tmp_path, "machine-a")
        phrase = await connect(a, monkeypatch)
        seed_content(a)
        await cloud_sync.run_sync(a.db)

        b = Install(tmp_path, "machine-b")
        await connect(b, monkeypatch, phrase=phrase)
        await cloud_sync.run_sync(b.db)

        a.activate(monkeypatch)
        cap = a.db.query(Capture).one()
        a.db.delete(cap)
        a.db.commit()
        report_a = await cloud_sync.run_sync(a.db)
        assert report_a.pushed_deletes == 1

        b.activate(monkeypatch)
        report_b = await cloud_sync.run_sync(b.db)
        assert report_b.pulled_deletes == 1
        assert b.db.query(Capture).count() == 0

    async def test_last_writer_wins_on_conflict(self, cloud, tmp_path, monkeypatch):
        a = Install(tmp_path, "machine-a")
        phrase = await connect(a, monkeypatch)
        seed_content(a)
        await cloud_sync.run_sync(a.db)

        b = Install(tmp_path, "machine-b")
        await connect(b, monkeypatch, phrase=phrase)
        await cloud_sync.run_sync(b.db)

        # Concurrent edits to the same capture on both machines.
        a.activate(monkeypatch)
        a.db.query(Capture).one().transcript_refined = "A's edit"
        a.db.commit()
        await cloud_sync.run_sync(a.db)

        b.activate(monkeypatch)
        b.db.query(Capture).one().transcript_refined = "B's edit"
        b.db.commit()
        await cloud_sync.run_sync(b.db)  # B pushes after A: B is the last writer

        a.activate(monkeypatch)
        await cloud_sync.run_sync(a.db)
        assert a.db.query(Capture).one().transcript_refined == "B's edit"

        b.activate(monkeypatch)
        await cloud_sync.run_sync(b.db)
        assert b.db.query(Capture).one().transcript_refined == "B's edit"
