"""End-to-end round-trip against a real voicebox-cloud dev server.

Skipped unless VOICEBOX_CLOUD_TEST_API + VOICEBOX_CLOUD_TEST_KEY are set:

    cd voicebox-cloud && pnpm dev:db && pnpm db:migrate && pnpm dev:api
    # create an account + API key (web app or seed script), then:
    VOICEBOX_CLOUD_TEST_API=http://localhost:17593 \\
    VOICEBOX_CLOUD_TEST_KEY=voicebox_… \\
    pytest backend/tests/test_cloud_roundtrip_integration.py -v

Exercises the scaffolded server for real: device registration, recovery
escrow, encrypted push (presigned PUT + commit), sync pull, decrypt — and
verifies the ciphertext at rest is opaque.
"""

import base64
import hashlib
import json
import os
import uuid

import pytest

from backend.services import cloud_crypto
from backend.services.cloud_api import CloudApiClient

API_URL = os.environ.get("VOICEBOX_CLOUD_TEST_API")
API_KEY = os.environ.get("VOICEBOX_CLOUD_TEST_KEY")

pytestmark = pytest.mark.skipif(
    not (API_URL and API_KEY),
    reason="set VOICEBOX_CLOUD_TEST_API and VOICEBOX_CLOUD_TEST_KEY to run against a dev server",
)


async def test_full_roundtrip():
    master_key = cloud_crypto.generate_master_key()
    client_id = str(uuid.uuid4())
    record_plain = json.dumps({"transcript_raw": "hello from the integration test", "language": "en"}).encode()
    audio_plain = os.urandom(64_000)  # stands in for capture audio

    async with CloudApiClient(API_URL, API_KEY) as client:
        # Device + escrow round-trip.
        private_key, public_key = cloud_crypto.generate_device_keypair()
        registered = await client.register_device("integration-test", base64.b64encode(public_key).decode())
        device_id = registered["deviceId"]
        wrapped = cloud_crypto.wrap_master_key_for_device(master_key, public_key)
        await client.put_wrapped_key(device_id, base64.b64encode(wrapped).decode())
        fetched = await client.get_wrapped_key(device_id)
        assert cloud_crypto.unwrap_master_key_for_device(base64.b64decode(fetched), private_key) == master_key

        # Push: encrypt locally, upsert metadata, PUT ciphertext, commit.
        object_id_placeholder = client_id  # AAD object binding uses the client id pre-push
        record_env = cloud_crypto.encrypt_blob(
            record_plain, master_key, object_id=object_id_placeholder, role="record", version=1
        )
        audio_env = cloud_crypto.encrypt_blob(
            audio_plain, master_key, object_id=object_id_placeholder, role="audio", version=1
        )
        pushed = await client.push_object(
            kind="capture",
            client_id=client_id,
            version=1,
            record={"hash": hashlib.sha256(record_env).hexdigest(), "size": len(record_env)},
            assets=[
                {
                    "role": "audio",
                    "clientAssetId": f"{client_id}-audio",
                    "hash": hashlib.sha256(audio_env).hexdigest(),
                    "size": len(audio_env),
                }
            ],
        )
        uploads = {u["for"]: u["url"] for u in pushed["uploads"]}
        await client.upload_blob(uploads["record"], record_env)
        await client.upload_blob(uploads[f"asset:{client_id}-audio"], audio_env)
        await client.commit_object(pushed["objectId"])

        # Pull: cursor 0 must include our object; ciphertext decrypts to the original.
        changes = await client.get_changes(since=pushed["seq"] - 1, limit=10)
        change = next(c for c in changes["changes"] if c["clientId"] == client_id)
        assert change["kind"] == "capture"
        assert not change["deleted"]

        record_cipher = await client.download_blob(change["record"]["url"])
        assert record_cipher == record_env  # opaque, byte-identical ciphertext at rest
        assert record_plain not in record_cipher
        assert (
            cloud_crypto.decrypt_blob(record_cipher, master_key, object_id=client_id, role="record", version=1)
            == record_plain
        )

        (asset,) = change["assets"]
        audio_cipher = await client.download_blob(asset["url"])
        assert (
            cloud_crypto.decrypt_blob(audio_cipher, master_key, object_id=client_id, role="audio", version=1)
            == audio_plain
        )

        # Tombstone propagates.
        await client.delete_object(pushed["objectId"])
        changes = await client.get_changes(since=changes["cursor"], limit=10)
        tombstone = next(c for c in changes["changes"] if c["clientId"] == client_id)
        assert tombstone["deleted"] is True
