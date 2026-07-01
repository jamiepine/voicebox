"""
Cloud sync identity: the Master Key lifecycle across devices.

Ties together the pieces below into the flows from the cloud design doc §9
(first device, add-a-device, recovery). The server participates only as a
mailbox for ciphertext — every wrap/unwrap here happens locally.

- ``cloud_crypto``  — the primitives (MK, recovery phrase, sealed boxes)
- ``cloud_keys``    — OS-keychain persistence of the private key + MK
- ``cloud_api``     — the bearer-key HTTP client
- ``CloudSettings`` — the local row holding the API key + sync device id

Flows:
- **First device** (``setup_device`` on a keyless account): generate MK, wrap
  it under a fresh recovery phrase → escrow to the server, wrap it to our own
  device key, keep MK in the keychain. Returns the phrase for one-time display.
- **New device on an existing account** (``setup_device`` when the account has
  key material): register and wait — an existing device provisions us
  (``provision_device`` there, ``adopt_wrapped_key`` here), or the user types
  the recovery phrase (``restore_with_phrase``).
"""

import base64
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from .. import config
from ..database import CloudSettings as DBCloudSettings
from . import cloud_crypto, cloud_keys
from .cloud_api import CloudApiClient
from .cloud_crypto import RecoveryWrap

logger = logging.getLogger(__name__)


class CloudAccountError(Exception):
    """The identity flow cannot proceed (not connected, no escrow, bad phrase…)."""


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(encoded: str) -> bytes:
    return base64.b64decode(encoded)


def _settings(db: Session) -> DBCloudSettings:
    row = db.query(DBCloudSettings).filter(DBCloudSettings.id == 1).first()
    if row is None or not row.api_key or not row.account_user_id:
        raise CloudAccountError("not connected to Voicebox Cloud — log in first")
    return row


def _client(row: DBCloudSettings) -> CloudApiClient:
    return CloudApiClient(config.get_cloud_api_url(), row.api_key)


@dataclass(frozen=True)
class SyncIdentity:
    # "unregistered"       — logged in, but not yet a sync device
    # "awaiting_provision" — registered, waiting for a wrapped MK or the phrase
    # "ready"              — MK in the keychain, sync can run
    status: str
    device_id: str | None


def identity_status(db: Session) -> SyncIdentity:
    row = _settings(db)
    if not row.sync_device_id:
        return SyncIdentity(status="unregistered", device_id=None)
    has_mk = cloud_keys.load_secret(row.account_user_id, cloud_keys.MASTER_KEY) is not None
    return SyncIdentity(status="ready" if has_mk else "awaiting_provision", device_id=row.sync_device_id)


async def setup_device(db: Session) -> str | None:
    """Register this install as a sync device.

    On a keyless account this is first-device setup: mints MK + the recovery
    escrow and returns the phrase — the caller must display it exactly once and
    never persist it. On an account with existing key material it returns None
    and the device waits in ``awaiting_provision``.
    """
    row = _settings(db)
    if row.sync_device_id:
        raise CloudAccountError("this install is already registered as a sync device")

    private_key, public_key = cloud_crypto.generate_device_keypair()
    async with _client(row) as client:
        registered = await client.register_device(row.device_name or "Voicebox Desktop", _b64e(public_key))
        device_id = registered["deviceId"]

        # Persist the private key before anything can depend on it; a crash
        # after registration leaves a provisionable device, never a locked one.
        cloud_keys.store_secret(row.account_user_id, cloud_keys.DEVICE_PRIVATE_KEY, private_key)
        row.sync_device_id = device_id
        db.commit()

        if registered["accountHasKey"]:
            logger.info("registered sync device %s; awaiting MK provisioning", device_id)
            return None

        master_key = cloud_crypto.generate_master_key()
        phrase = cloud_crypto.generate_recovery_phrase()
        escrow = cloud_crypto.wrap_master_key_with_phrase(master_key, phrase)
        await client.put_account_key(_b64e(escrow.wrapped_key), _b64e(escrow.kdf_salt), escrow.kdf_params)
        await client.put_wrapped_key(device_id, _b64e(cloud_crypto.wrap_master_key_for_device(master_key, public_key)))
        cloud_keys.store_secret(row.account_user_id, cloud_keys.MASTER_KEY, master_key)
        logger.info("initialized account key material as first device %s", device_id)
        return phrase


async def restore_with_phrase(db: Session, phrase: str) -> None:
    """Recover MK from the server-side escrow using the recovery phrase.
    The device must be registered (``setup_device``) first."""
    row = _settings(db)
    if not row.sync_device_id:
        raise CloudAccountError("register this device before restoring")
    if not cloud_crypto.validate_recovery_phrase(phrase):
        raise CloudAccountError("that doesn't look like a valid recovery phrase — check for typos")

    async with _client(row) as client:
        escrow = await client.get_account_key()
        if not escrow:
            raise CloudAccountError("this account has no recovery escrow yet")
        wrap = RecoveryWrap(
            wrapped_key=_b64d(escrow["recoveryWrappedKey"]),
            kdf_salt=_b64d(escrow["kdfSalt"]),
            kdf_params=escrow["kdfParams"],
        )
        master_key = cloud_crypto.unwrap_master_key_with_phrase(wrap, phrase)

        # Also wrap MK to our own device key so future restores on this device
        # don't need the phrase.
        private_key = cloud_keys.load_secret(row.account_user_id, cloud_keys.DEVICE_PRIVATE_KEY)
        if private_key is None:
            raise CloudAccountError("device key missing from the OS keychain — register this device again")
        public_key = cloud_crypto.device_public_key(private_key)
        await client.put_wrapped_key(
            row.sync_device_id, _b64e(cloud_crypto.wrap_master_key_for_device(master_key, public_key))
        )
        cloud_keys.store_secret(row.account_user_id, cloud_keys.MASTER_KEY, master_key)
        logger.info("restored master key from recovery phrase on device %s", row.sync_device_id)


async def adopt_wrapped_key(db: Session) -> bool:
    """For a device in ``awaiting_provision``: fetch our wrapped MK if an
    existing device has provisioned it. Returns True once MK is in the keychain."""
    row = _settings(db)
    if not row.sync_device_id:
        raise CloudAccountError("register this device before adopting a key")

    private_key = cloud_keys.load_secret(row.account_user_id, cloud_keys.DEVICE_PRIVATE_KEY)
    if private_key is None:
        raise CloudAccountError("device key missing from the OS keychain — register this device again")

    async with _client(row) as client:
        wrapped = await client.get_wrapped_key(row.sync_device_id)
    if not wrapped:
        return False
    master_key = cloud_crypto.unwrap_master_key_for_device(_b64d(wrapped), private_key)
    cloud_keys.store_secret(row.account_user_id, cloud_keys.MASTER_KEY, master_key)
    logger.info("adopted provisioned master key on device %s", row.sync_device_id)
    return True


async def provision_device(db: Session, target_device_id: str) -> None:
    """Run on a device that already holds MK: wrap it to another registered
    device's public key so that device can start syncing."""
    row = _settings(db)
    master_key = cloud_keys.load_secret(row.account_user_id, cloud_keys.MASTER_KEY)
    if master_key is None:
        raise CloudAccountError("this device holds no master key to provision with")

    async with _client(row) as client:
        devices = await client.list_devices()
        target = next((d for d in devices if d["id"] == target_device_id and not d.get("revokedAt")), None)
        if target is None:
            raise CloudAccountError("target device not found (or revoked)")
        wrapped = cloud_crypto.wrap_master_key_for_device(master_key, _b64d(target["publicKey"]))
        await client.put_wrapped_key(target_device_id, _b64e(wrapped))
    logger.info("provisioned master key to device %s", target_device_id)


def load_master_key(db: Session) -> bytes:
    """The MK for the sync engine. Raises if this device isn't ready."""
    row = _settings(db)
    master_key = cloud_keys.load_secret(row.account_user_id, cloud_keys.MASTER_KEY)
    if master_key is None:
        raise CloudAccountError("no master key on this device — finish setup or restore first")
    return master_key
