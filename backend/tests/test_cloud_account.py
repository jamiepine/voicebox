"""Tests for the cloud sync identity flows (services/cloud_account.py).

Runs the real flows against a fake in-process cloud (httpx.MockTransport) and
a fake in-memory keyring — no network, no OS keychain. The central assertion:
the master key and recovery phrase never appear in anything sent to the server.
"""

import base64
from datetime import datetime

import keyring
import keyring.backend
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, CloudSettings
from backend.services import cloud_account, cloud_crypto, cloud_keys
from backend.services.cloud_account import CloudAccountError
from backend.services.cloud_api import CloudApiClient
from backend.tests.fake_cloud import FakeCloud

USER_A = "user-a"


class InMemoryKeyring(keyring.backend.KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        self.store.pop((service, username), None)


@pytest.fixture
def fake_keyring(monkeypatch):
    backend = InMemoryKeyring()
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    return backend


@pytest.fixture
def cloud():
    return FakeCloud()


def make_db(account_user_id=USER_A):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(
        CloudSettings(
            id=1,
            api_key="voicebox_test",
            device_name="Test Mac",
            account_user_id=account_user_id,
            connected_at=datetime(2026, 7, 1),
        )
    )
    db.commit()
    return db


@pytest.fixture
def patched_client(monkeypatch, cloud):
    def _client(row):
        return CloudApiClient("http://cloud.test", row.api_key, transport=cloud.transport())

    monkeypatch.setattr(cloud_account, "_client", _client)


@pytest.mark.usefixtures("patched_client", "fake_keyring")
class TestIdentityFlows:
    async def test_first_device_setup(self, cloud):
        db = make_db()
        phrase = await cloud_account.setup_device(db)

        assert phrase is not None
        assert cloud_crypto.validate_recovery_phrase(phrase)
        assert cloud_account.identity_status(db).status == "ready"
        assert cloud.account_key is not None
        # Registered + provisioned to itself.
        (device,) = cloud.devices.values()
        assert device["wrappedMasterKey"]

        # The invariant: neither MK nor the phrase ever crossed the wire.
        mk = cloud_account.load_master_key(db)
        for body in cloud.seen_bodies:
            assert mk not in body
            assert base64.b64encode(mk) not in body
            assert phrase.encode() not in body

    async def test_second_device_via_provisioning(self, cloud):
        db_a = make_db()
        await cloud_account.setup_device(db_a)
        mk_a = cloud_account.load_master_key(db_a)

        # Second install: same account, its own DB + keychain namespace. Reuse
        # the same fake keyring but a distinct account row would collide, so
        # simulate the second device with a separate account_user_id-scoped
        # keychain by clearing MK after capturing device state.
        db_b = make_db(account_user_id="user-a-second-install")
        assert await cloud_account.setup_device(db_b) is None  # account already has key material
        assert cloud_account.identity_status(db_b).status == "awaiting_provision"
        assert await cloud_account.adopt_wrapped_key(db_b) is False  # nothing provisioned yet

        target_id = db_b.query(CloudSettings).one().sync_device_id
        await cloud_account.provision_device(db_a, target_id)
        assert await cloud_account.adopt_wrapped_key(db_b) is True
        assert cloud_account.load_master_key(db_b) == mk_a

    async def test_restore_with_phrase(self, cloud):
        db_a = make_db()
        phrase = await cloud_account.setup_device(db_a)
        mk_a = cloud_account.load_master_key(db_a)

        db_b = make_db(account_user_id="user-a-fresh-machine")
        assert await cloud_account.setup_device(db_b) is None
        await cloud_account.restore_with_phrase(db_b, phrase)
        assert cloud_account.load_master_key(db_b) == mk_a
        assert cloud_account.identity_status(db_b).status == "ready"

    async def test_restore_rejects_wrong_phrase(self, cloud):
        db_a = make_db()
        await cloud_account.setup_device(db_a)

        db_b = make_db(account_user_id="user-a-fresh-machine")
        await cloud_account.setup_device(db_b)
        with pytest.raises(cloud_crypto.CloudCryptoError):
            await cloud_account.restore_with_phrase(db_b, cloud_crypto.generate_recovery_phrase())

    async def test_restore_rejects_invalid_phrase_early(self, cloud):
        db = make_db()
        await cloud_account.setup_device(db)
        with pytest.raises(CloudAccountError, match="valid recovery phrase"):
            await cloud_account.restore_with_phrase(
                db, "not a real phrase at all twelve words missing checksum here ok"
            )

    async def test_double_registration_rejected(self, cloud):
        db = make_db()
        await cloud_account.setup_device(db)
        with pytest.raises(CloudAccountError, match="already registered"):
            await cloud_account.setup_device(db)

    async def test_requires_login(self, cloud):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine)()
        with pytest.raises(CloudAccountError, match="log in"):
            await cloud_account.setup_device(db)


@pytest.mark.usefixtures("fake_keyring")
class TestKeyStore:
    def test_round_trip_and_clear(self):
        cloud_keys.store_secret(USER_A, cloud_keys.MASTER_KEY, b"\x01" * 32)
        assert cloud_keys.load_secret(USER_A, cloud_keys.MASTER_KEY) == b"\x01" * 32
        assert cloud_keys.load_secret("other-user", cloud_keys.MASTER_KEY) is None
        cloud_keys.clear(USER_A)
        assert cloud_keys.load_secret(USER_A, cloud_keys.MASTER_KEY) is None

    def test_delete_absent_is_noop(self):
        cloud_keys.delete_secret(USER_A, cloud_keys.DEVICE_PRIVATE_KEY)
