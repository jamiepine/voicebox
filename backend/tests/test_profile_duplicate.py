"""
Tests for POST /profiles/{id}/duplicate.

The point of the endpoint is that it is *not* an export/import round-trip.
export_import.py writes only name/description/language into its manifest, so
importing an exported profile silently drops personality, effects_chain,
default_engine and the preset/designed fields -- and export refuses profiles
with no samples, which is every preset voice.  These tests pin the fields a
round-trip would have lost.

Usage:
    python -m pytest backend/tests/test_profile_duplicate.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-duplicate-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402

EFFECTS = [{"type": "reverb", "params": {"room_size": 0.4}}]
PERSONALITY = "Speaks in short, dry sentences. Never uses exclamation marks."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _delete(client, profile_id: str) -> None:
    client.delete(f"/profiles/{profile_id}")


@pytest.fixture
def preset_profile(client):
    """A preset (Kokoro) profile: rich metadata, zero samples."""
    r = client.post(
        "/profiles",
        json={
            "name": "Duplicate Source",
            "description": "original description",
            "language": "en",
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": "af_bella",
            "personality": PERSONALITY,
        },
    )
    assert r.status_code == 200, r.text
    created = r.json()
    yield created
    _delete(client, created["id"])


def test_duplicate_preserves_personality(client, preset_profile):
    """The field an export/import round-trip loses most damagingly."""
    copy = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        assert copy["personality"] == PERSONALITY
    finally:
        _delete(client, copy["id"])


def test_duplicate_preserves_preset_fields(client, preset_profile):
    copy = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        assert copy["voice_type"] == "preset"
        assert copy["preset_engine"] == "kokoro"
        assert copy["preset_voice_id"] == "af_bella"
        # default_engine is auto-derived from preset_engine on create.
        assert copy["default_engine"] == "kokoro"
    finally:
        _delete(client, copy["id"])


def test_duplicate_works_for_profiles_without_samples(client, preset_profile):
    """Export raises ValueError for a sample-less profile, so this case is
    unreachable via export/import."""
    r = client.post(f"/profiles/{preset_profile['id']}/duplicate")
    assert r.status_code == 200, r.text
    try:
        assert r.json()["sample_count"] == 0
    finally:
        _delete(client, r.json()["id"])


def test_duplicate_preserves_description_and_language(client, preset_profile):
    copy = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        assert copy["description"] == "original description"
        assert copy["language"] == "en"
    finally:
        _delete(client, copy["id"])


def test_duplicate_gets_a_new_id_and_copy_suffixed_name(client, preset_profile):
    copy = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        assert copy["id"] != preset_profile["id"]
        assert copy["name"] == "Duplicate Source (copy)"
    finally:
        _delete(client, copy["id"])


def test_repeated_duplicates_get_distinct_names(client, preset_profile):
    """profiles.name is UNIQUE, so the second copy must not collide."""
    first = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    second = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        assert first["name"] == "Duplicate Source (copy)"
        assert second["name"] == "Duplicate Source (copy) (1)"
    finally:
        _delete(client, first["id"])
        _delete(client, second["id"])


def test_duplicate_accepts_an_explicit_name(client, preset_profile):
    copy = client.post(
        f"/profiles/{preset_profile['id']}/duplicate", json={"name": "Chosen Name"}
    ).json()
    try:
        assert copy["name"] == "Chosen Name"
    finally:
        _delete(client, copy["id"])


def test_duplicate_starts_with_no_generations(client, preset_profile):
    copy = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        assert copy["generation_count"] == 0
    finally:
        _delete(client, copy["id"])


def test_duplicate_preserves_effects_chain(client, preset_profile):
    set_effects = client.put(
        f"/profiles/{preset_profile['id']}/effects", json={"effects_chain": EFFECTS}
    )
    assert set_effects.status_code == 200, set_effects.text

    copy = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        assert copy["effects_chain"], "effects chain was dropped by duplicate"
        assert copy["effects_chain"][0]["type"] == "reverb"
    finally:
        _delete(client, copy["id"])


def test_duplicate_inherits_folder(client, preset_profile):
    folder = client.post("/folders", json={"name": "Dup Folder", "kind": "voice"}).json()
    client.put(f"/profiles/{preset_profile['id']}/folder", json={"folder_id": folder["id"]})

    copy = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        assert copy["folder_id"] == folder["id"]
    finally:
        _delete(client, copy["id"])
        client.delete(f"/folders/{folder['id']}")


def test_editing_the_copy_leaves_the_original_untouched(client, preset_profile):
    """A duplicate must be independent, not a shared reference."""
    copy = client.post(f"/profiles/{preset_profile['id']}/duplicate").json()
    try:
        client.put(
            f"/profiles/{copy['id']}",
            json={
                "name": "Edited Copy",
                "description": "changed",
                "language": "en",
                "voice_type": "preset",
                "preset_engine": "kokoro",
                "preset_voice_id": "af_bella",
                "personality": "Totally different.",
            },
        )
        original = client.get(f"/profiles/{preset_profile['id']}").json()
        assert original["personality"] == PERSONALITY
        assert original["description"] == "original description"
    finally:
        _delete(client, copy["id"])


def test_duplicating_a_missing_profile_is_404(client):
    assert client.post("/profiles/does-not-exist/duplicate").status_code == 404


# ── Sample copying ───────────────────────────────────────────────────


def _write_reference_wav(path: Path) -> None:
    """A 3s tone: long enough for the 2s minimum, loud enough for the RMS floor."""
    import numpy as np
    import soundfile as sf

    sr = 24000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(str(path), tone, sr)


@pytest.fixture
def cloned_profile_with_sample(client, tmp_path):
    r = client.post(
        "/profiles",
        json={"name": "Cloned Source", "description": "has a sample", "language": "en"},
    )
    assert r.status_code == 200, r.text
    created = r.json()

    # Delete unconditionally: if the upload below fails, the profile would
    # otherwise survive and collide with the next test's UNIQUE name.
    try:
        wav = tmp_path / "reference.wav"
        _write_reference_wav(wav)
        with wav.open("rb") as fh:
            upload = client.post(
                f"/profiles/{created['id']}/samples",
                files={"file": ("reference.wav", fh, "audio/wav")},
                data={"reference_text": "This is the reference transcript."},
            )
        assert upload.status_code == 200, upload.text

        yield created
    finally:
        _delete(client, created["id"])


def test_duplicate_copies_samples(client, cloned_profile_with_sample):
    copy = client.post(f"/profiles/{cloned_profile_with_sample['id']}/duplicate").json()
    try:
        assert copy["sample_count"] == 1

        original_samples = client.get(
            f"/profiles/{cloned_profile_with_sample['id']}/samples"
        ).json()
        copy_samples = client.get(f"/profiles/{copy['id']}/samples").json()

        assert copy_samples[0]["reference_text"] == original_samples[0]["reference_text"]
        # Distinct rows pointing at distinct files — not a shared reference.
        assert copy_samples[0]["id"] != original_samples[0]["id"]
        assert copy_samples[0]["audio_path"] != original_samples[0]["audio_path"]
    finally:
        _delete(client, copy["id"])


def test_duplicated_sample_audio_is_byte_identical(client, cloned_profile_with_sample):
    """Copied rather than re-encoded, so the clone sounds like the original."""
    from backend import config

    copy = client.post(f"/profiles/{cloned_profile_with_sample['id']}/duplicate").json()
    try:
        original_samples = client.get(
            f"/profiles/{cloned_profile_with_sample['id']}/samples"
        ).json()
        copy_samples = client.get(f"/profiles/{copy['id']}/samples").json()

        original_bytes = config.resolve_storage_path(
            original_samples[0]["audio_path"]
        ).read_bytes()
        copy_bytes = config.resolve_storage_path(
            copy_samples[0]["audio_path"]
        ).read_bytes()

        assert copy_bytes == original_bytes
    finally:
        _delete(client, copy["id"])


def test_deleting_the_copy_leaves_the_originals_audio_intact(client, cloned_profile_with_sample):
    """Copied files must be independent — deleting one must not take the
    other's audio with it."""
    from backend import config

    copy = client.post(f"/profiles/{cloned_profile_with_sample['id']}/duplicate").json()
    original_samples = client.get(
        f"/profiles/{cloned_profile_with_sample['id']}/samples"
    ).json()
    original_audio = config.resolve_storage_path(original_samples[0]["audio_path"])

    _delete(client, copy["id"])

    assert original_audio.exists()


# ── Name allocation under contention (review finding on #1007) ───────


def test_duplicate_names_are_settled_by_the_insert(client):
    """`get_unique_profile_name` was a check-then-act: it SELECTed a free name
    and a later INSERT took it, so two concurrent duplicates could both be
    handed the same name and the loser hit the unique constraint as a 500.

    Simulated here by pre-taking the name between allocation attempts, which is
    what a racing request does."""
    import uuid as _uuid

    from backend.database import VoiceProfile as DBVoiceProfile, get_db
    from backend.services.profiles import insert_profile_with_unique_name

    db = next(get_db())
    try:
        base = f"Race Test {_uuid.uuid4().hex[:8]}"

        # Something else already holds the base name.
        db.add(DBVoiceProfile(id=str(_uuid.uuid4()), name=base, language="en"))
        db.commit()

        row = insert_profile_with_unique_name(
            base,
            db,
            lambda candidate: DBVoiceProfile(
                id=str(_uuid.uuid4()), name=candidate, language="en"
            ),
        )
        assert row.name == f"{base} (1)", "should fall through to the next suffix"

        # And again, so the counter keeps advancing rather than sticking.
        row2 = insert_profile_with_unique_name(
            base,
            db,
            lambda candidate: DBVoiceProfile(
                id=str(_uuid.uuid4()), name=candidate, language="en"
            ),
        )
        assert row2.name == f"{base} (2)"

        for name in (base, f"{base} (1)", f"{base} (2)"):
            db.query(DBVoiceProfile).filter_by(name=name).delete()
        db.commit()
    finally:
        db.close()
