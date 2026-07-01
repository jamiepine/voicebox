"""
OS-keychain persistence for cloud E2E key material.

The bearer API key lives in the local database (``CloudSettings``) because it
is auth, not secrecy. The *encryption* keys never touch the database: this
module stores the device's X25519 private key and the unwrapped Master Key in
the OS keychain (macOS Keychain, Windows Credential Locker, Secret Service on
Linux) via ``keyring``. Entries are namespaced by cloud account user id so
relinking to a different account can never read the previous account's keys.

Headless installs (Docker/web) may have no keychain backend; operations then
raise ``CloudKeyStoreError`` and cloud sync stays unavailable rather than
silently degrading to plaintext-on-disk key storage.
"""

import base64

import keyring
import keyring.errors

_SERVICE = "sh.voicebox.cloud"

DEVICE_PRIVATE_KEY = "device_private_key"
MASTER_KEY = "master_key"
_ALL_ENTRIES = (DEVICE_PRIVATE_KEY, MASTER_KEY)


class CloudKeyStoreError(Exception):
    """The OS keychain is unavailable or rejected the operation."""


def _entry(account_user_id: str, name: str) -> str:
    return f"{account_user_id}:{name}"


def store_secret(account_user_id: str, name: str, secret: bytes) -> None:
    try:
        keyring.set_password(_SERVICE, _entry(account_user_id, name), base64.b64encode(secret).decode("ascii"))
    except keyring.errors.KeyringError as err:
        raise CloudKeyStoreError(f"could not store {name} in the OS keychain") from err


def load_secret(account_user_id: str, name: str) -> bytes | None:
    try:
        stored = keyring.get_password(_SERVICE, _entry(account_user_id, name))
    except keyring.errors.KeyringError as err:
        raise CloudKeyStoreError(f"could not read {name} from the OS keychain") from err
    return base64.b64decode(stored) if stored else None


def delete_secret(account_user_id: str, name: str) -> None:
    try:
        keyring.delete_password(_SERVICE, _entry(account_user_id, name))
    except keyring.errors.PasswordDeleteError:
        pass  # already absent
    except keyring.errors.KeyringError as err:
        raise CloudKeyStoreError(f"could not delete {name} from the OS keychain") from err


def clear(account_user_id: str) -> None:
    """Forget all key material for an account (disconnect / account switch)."""
    for name in _ALL_ENTRIES:
        delete_secret(account_user_id, name)
