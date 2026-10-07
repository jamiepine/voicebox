"""
Tests for folder organisation of voices and generated clips.

Covers the asymmetry between the two folder kinds: voice folders are flat,
clip folders nest.  Also covers the guarantee that deleting a folder never
deletes what is inside it.

VOICEBOX_DATA_DIR is set before importing the app so the whole suite runs
against a throwaway data directory and never touches a real install.

Usage:
    python -m pytest backend/tests/test_folders.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_DATA_DIR = tempfile.mkdtemp(prefix="voicebox-folders-test-")
os.environ["VOICEBOX_DATA_DIR"] = _DATA_DIR

from starlette.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Context-manager form triggers the lifespan, which runs init_db().
    with TestClient(app) as c:
        yield c


@pytest.fixture
def voice_folder(client):
    r = client.post("/folders", json={"name": "Podcast voices", "kind": "voice"})
    assert r.status_code == 200, r.text
    folder = r.json()
    yield folder
    client.delete(f"/folders/{folder['id']}")


@pytest.fixture
def profile(client):
    """A preset profile — deliberately one with no samples, since those are
    exactly the profiles export/import cannot round-trip."""
    r = client.post(
        "/profiles",
        json={
            "name": "Folder Test Voice",
            "description": "fixture",
            "language": "en",
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": "af_bella",
            "personality": "Speaks in short, dry sentences.",
        },
    )
    assert r.status_code == 200, r.text
    created = r.json()
    yield created
    client.delete(f"/profiles/{created['id']}")


# ── Folder CRUD ──────────────────────────────────────────────────────


def test_create_and_list_voice_folder(client, voice_folder):
    assert voice_folder["kind"] == "voice"
    assert voice_folder["parent_id"] is None

    listed = client.get("/folders", params={"kind": "voice"}).json()
    assert voice_folder["id"] in [f["id"] for f in listed]


def test_voice_folders_are_excluded_from_generation_listing(client, voice_folder):
    listed = client.get("/folders", params={"kind": "generation"}).json()
    assert voice_folder["id"] not in [f["id"] for f in listed]


def test_unknown_kind_is_rejected(client):
    assert client.get("/folders", params={"kind": "nonsense"}).status_code == 400


# ── Story folders ────────────────────────────────────────────────────


def test_story_folders_nest(client):
    """Stories nest like clips, not flat like voices."""
    parent = client.post("/folders", json={"name": "Podcast", "kind": "story"}).json()
    child = client.post(
        "/folders", json={"name": "Season 2", "kind": "story", "parent_id": parent["id"]}
    )
    assert child.status_code == 200, child.text
    assert child.json()["parent_id"] == parent["id"]

    client.delete(f"/folders/{child.json()['id']}")
    client.delete(f"/folders/{parent['id']}")


def test_assign_and_unassign_story(client):
    folder = client.post("/folders", json={"name": "Archive", "kind": "story"}).json()
    story = client.post("/stories", json={"name": "Folder Test Story"}).json()

    r = client.put(f"/stories/{story['id']}/folder", json={"folder_id": folder["id"]})
    assert r.status_code == 200, r.text
    assert r.json()["folder_id"] == folder["id"]

    listed = client.get("/stories").json()
    filed = next(s for s in listed if s["id"] == story["id"])
    assert filed["folder_id"] == folder["id"], "folder_id missing from the story list response"

    assert (
        client.put(f"/stories/{story['id']}/folder", json={"folder_id": None}).json()["folder_id"]
        is None
    )

    client.delete(f"/stories/{story['id']}")
    client.delete(f"/folders/{folder['id']}")


def test_story_cannot_go_into_a_voice_folder(client, voice_folder):
    story = client.post("/stories", json={"name": "Wrong Folder Story"}).json()
    r = client.put(f"/stories/{story['id']}/folder", json={"folder_id": voice_folder["id"]})
    assert r.status_code == 400
    client.delete(f"/stories/{story['id']}")


def test_deleting_a_story_folder_keeps_the_stories(client):
    folder = client.post("/folders", json={"name": "Doomed", "kind": "story"}).json()
    story = client.post("/stories", json={"name": "Survivor Story"}).json()
    client.put(f"/stories/{story['id']}/folder", json={"folder_id": folder["id"]})

    r = client.delete(f"/folders/{folder['id']}")
    assert r.json()["items_released"] == 1

    survivor = client.get("/stories").json()
    kept = next(s for s in survivor if s["id"] == story["id"])
    assert kept["folder_id"] is None

    client.delete(f"/stories/{story['id']}")


def test_rename_folder(client, voice_folder):
    r = client.patch(f"/folders/{voice_folder['id']}", json={"name": "Renamed"})
    assert r.status_code == 200
    assert r.json()["name"] == "Renamed"


def test_folder_name_is_trimmed(client):
    r = client.post("/folders", json={"name": "  Padded  ", "kind": "voice"})
    assert r.json()["name"] == "Padded"
    client.delete(f"/folders/{r.json()['id']}")


def test_missing_folder_is_404(client):
    assert client.patch("/folders/nope", json={"name": "x"}).status_code == 404
    assert client.delete("/folders/nope").status_code == 404


# ── Nesting rules ────────────────────────────────────────────────────


def test_voice_folders_cannot_nest(client, voice_folder):
    r = client.post(
        "/folders",
        json={"name": "Child", "kind": "voice", "parent_id": voice_folder["id"]},
    )
    assert r.status_code == 400
    assert "nested" in r.json()["detail"].lower()


def test_generation_folders_nest(client):
    parent = client.post("/folders", json={"name": "Season 1", "kind": "generation"}).json()
    child = client.post(
        "/folders",
        json={"name": "Episode 1", "kind": "generation", "parent_id": parent["id"]},
    ).json()

    assert child["parent_id"] == parent["id"]

    grandchild = client.post(
        "/folders",
        json={"name": "Takes", "kind": "generation", "parent_id": child["id"]},
    )
    assert grandchild.status_code == 200

    client.delete(f"/folders/{grandchild.json()['id']}")
    client.delete(f"/folders/{child['id']}")
    client.delete(f"/folders/{parent['id']}")


def test_parent_must_share_kind(client, voice_folder):
    r = client.post(
        "/folders",
        json={"name": "Mismatched", "kind": "generation", "parent_id": voice_folder["id"]},
    )
    assert r.status_code == 400


def test_folder_cannot_be_its_own_parent(client):
    folder = client.post("/folders", json={"name": "Loop", "kind": "generation"}).json()
    r = client.patch(f"/folders/{folder['id']}", json={"parent_id": folder["id"]})
    assert r.status_code == 400
    client.delete(f"/folders/{folder['id']}")


def test_folder_cannot_move_into_its_own_descendant(client):
    """The cycle that would orphan a whole subtree from the root."""
    parent = client.post("/folders", json={"name": "Outer", "kind": "generation"}).json()
    child = client.post(
        "/folders",
        json={"name": "Inner", "kind": "generation", "parent_id": parent["id"]},
    ).json()

    r = client.patch(f"/folders/{parent['id']}", json={"parent_id": child["id"]})
    assert r.status_code == 400
    assert "inside itself" in r.json()["detail"].lower()

    client.delete(f"/folders/{child['id']}")
    client.delete(f"/folders/{parent['id']}")


def test_detach_moves_folder_to_root(client):
    parent = client.post("/folders", json={"name": "P", "kind": "generation"}).json()
    child = client.post(
        "/folders", json={"name": "C", "kind": "generation", "parent_id": parent["id"]}
    ).json()

    r = client.post(f"/folders/{child['id']}/detach")
    assert r.status_code == 200
    assert r.json()["parent_id"] is None

    client.delete(f"/folders/{child['id']}")
    client.delete(f"/folders/{parent['id']}")


# ── Membership ───────────────────────────────────────────────────────


def test_assign_and_unassign_profile(client, voice_folder, profile):
    r = client.put(
        f"/profiles/{profile['id']}/folder", json={"folder_id": voice_folder["id"]}
    )
    assert r.status_code == 200
    assert r.json()["folder_id"] == voice_folder["id"]

    r = client.put(f"/profiles/{profile['id']}/folder", json={"folder_id": None})
    assert r.json()["folder_id"] is None


def test_profile_cannot_go_into_a_clip_folder(client, profile):
    clip_folder = client.post(
        "/folders", json={"name": "Clips", "kind": "generation"}
    ).json()

    r = client.put(
        f"/profiles/{profile['id']}/folder", json={"folder_id": clip_folder["id"]}
    )
    assert r.status_code == 400

    client.delete(f"/folders/{clip_folder['id']}")


def test_item_count_reflects_members(client, voice_folder, profile):
    client.put(f"/profiles/{profile['id']}/folder", json={"folder_id": voice_folder["id"]})

    listed = client.get("/folders", params={"kind": "voice"}).json()
    entry = next(f for f in listed if f["id"] == voice_folder["id"])
    assert entry["item_count"] == 1


# ── Deletion preserves contents ──────────────────────────────────────


def test_deleting_folder_releases_members_but_keeps_them(client, profile):
    folder = client.post("/folders", json={"name": "Temp", "kind": "voice"}).json()
    client.put(f"/profiles/{profile['id']}/folder", json={"folder_id": folder["id"]})

    r = client.delete(f"/folders/{folder['id']}")
    assert r.status_code == 200
    assert r.json()["items_released"] == 1

    # The voice itself must survive, now uncategorised.
    survivor = client.get(f"/profiles/{profile['id']}")
    assert survivor.status_code == 200
    assert survivor.json()["folder_id"] is None


def test_deleting_parent_reparents_children_rather_than_orphaning(client):
    grandparent = client.post("/folders", json={"name": "GP", "kind": "generation"}).json()
    parent = client.post(
        "/folders", json={"name": "P", "kind": "generation", "parent_id": grandparent["id"]}
    ).json()
    child = client.post(
        "/folders", json={"name": "C", "kind": "generation", "parent_id": parent["id"]}
    ).json()

    r = client.delete(f"/folders/{parent['id']}")
    assert r.json()["folders_reparented"] == 1

    listed = client.get("/folders", params={"kind": "generation"}).json()
    moved = next(f for f in listed if f["id"] == child["id"])
    assert moved["parent_id"] == grandparent["id"]

    client.delete(f"/folders/{child['id']}")
    client.delete(f"/folders/{grandparent['id']}")


# ── History filtering ────────────────────────────────────────────────


def test_history_rejects_out_of_range_limit_with_422(client):
    """Previously surfaced as a 500 — HistoryQuery was built from raw ints
    inside the handler, so its ValidationError escaped as a server error."""
    assert client.get("/history", params={"limit": 500}).status_code == 422


def test_history_accepts_folder_filters(client):
    folder = client.post("/folders", json={"name": "F", "kind": "generation"}).json()

    assert client.get("/history", params={"folder_id": folder["id"]}).status_code == 200
    assert client.get("/history", params={"uncategorised_only": True}).status_code == 200

    client.delete(f"/folders/{folder['id']}")
