"""
Client-side cryptography for Voicebox Cloud backup & sync.

This is the auditable half of the cloud's privacy promise: every byte that
leaves this machine is encrypted here first, and the server only ever stores
ciphertext plus routing metadata. The key hierarchy (cloud repo,
``docs/DESIGN.md``):

    Recovery phrase (BIP39)                Device X25519 keypairs
            | Argon2id(salt, params)               | sealed box
            v                                      v
       Recovery KEK --wrap--> Master Key (MK) <--wrapped to each device
                                    |
                                    | per-blob random Content Key (CK),
                                    | wrapped by MK inside the envelope
                                    v
                XChaCha20-Poly1305(CK) over each record/asset blob

Everything in this module is a pure function over bytes — no I/O, no storage,
no network. Key persistence (OS keychain) and the sync engine live elsewhere.

Invariants this module enforces:
- The Master Key is random, never derived from anything the server holds
  (API keys and wallet keys are auth/entitlement, never encryption roots).
- Each blob's AAD binds it to its logical slot (object, role, version), so a
  server cannot substitute one blob for another without decryption failing.
- Content Keys travel only inside the envelope, wrapped by MK — the database
  (local and cloud) stays free of key material.
"""

import json
import secrets
from dataclasses import dataclass

import nacl.bindings
import nacl.exceptions
import nacl.pwhash
import nacl.utils
from mnemonic import Mnemonic
from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.secret import SecretBox

KEY_BYTES = 32

# Envelope framing: magic | alg | len(wrapped_ck) | wrapped_ck | nonce | ciphertext
ENVELOPE_MAGIC = b"VBX1"
ALG_XCHACHA20_POLY1305 = 1
_WRAPPED_CK_LEN_BYTES = 2
_NONCE_BYTES = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES  # 24

# Argon2id cost for the recovery-phrase KEK. MODERATE (~256 MiB, interactive
# latency) fits a desktop restore flow; the phrase itself already carries
# 128 bits of entropy, so the KDF is hardening, not the main defense.
_KDF_OPSLIMIT = nacl.pwhash.argon2id.OPSLIMIT_MODERATE
_KDF_MEMLIMIT = nacl.pwhash.argon2id.MEMLIMIT_MODERATE

_mnemonic = Mnemonic("english")


class CloudCryptoError(Exception):
    """Envelope malformed, key wrong, or ciphertext tampered with."""


# ---------------------------------------------------------------------------
# Master Key + device keys


def generate_master_key() -> bytes:
    """The 32-byte root of content secrecy. Generated once per account, client-side."""
    return secrets.token_bytes(KEY_BYTES)


def generate_device_keypair() -> tuple[bytes, bytes]:
    """X25519 (private, public) for this install. The private key never leaves
    the device; the public key is registered with the cloud so existing devices
    can wrap MK to it."""
    private = PrivateKey.generate()
    return bytes(private), bytes(private.public_key)


def wrap_master_key_for_device(master_key: bytes, device_public_key: bytes) -> bytes:
    """Seal MK to another device's public key (run on an *existing* device when
    provisioning a new one). Only the target device's private key can open it."""
    return SealedBox(PublicKey(device_public_key)).encrypt(master_key)


def unwrap_master_key_for_device(wrapped: bytes, device_private_key: bytes) -> bytes:
    """Open a sealed MK with this device's private key."""
    try:
        return SealedBox(PrivateKey(device_private_key)).decrypt(wrapped)
    except nacl.exceptions.CryptoError as err:
        raise CloudCryptoError("wrapped master key does not match this device key") from err


# ---------------------------------------------------------------------------
# Recovery phrase escrow


@dataclass(frozen=True)
class RecoveryWrap:
    """What the cloud stores in ``account_key``: ciphertext + the KDF context
    any client needs to re-derive the KEK from the phrase. Holds no secrets."""

    wrapped_key: bytes
    kdf_salt: bytes
    kdf_params: str  # JSON, e.g. {"m": ..., "t": ..., "p": 1}


def generate_recovery_phrase() -> str:
    """12-word BIP39 mnemonic (128-bit entropy) — shown to the user exactly once."""
    return _mnemonic.generate(strength=128)


def validate_recovery_phrase(phrase: str) -> bool:
    """Word-level checksum validation, for catching typos before an unwrap attempt."""
    return _mnemonic.check(_normalize_phrase(phrase))


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.lower().split())


def _derive_kek(phrase: str, salt: bytes, opslimit: int, memlimit: int) -> bytes:
    return nacl.pwhash.argon2id.kdf(
        KEY_BYTES,
        _normalize_phrase(phrase).encode("utf-8"),
        salt,
        opslimit=opslimit,
        memlimit=memlimit,
    )


def wrap_master_key_with_phrase(master_key: bytes, phrase: str) -> RecoveryWrap:
    """Argon2id-stretch the phrase into a KEK and wrap MK under it."""
    salt = nacl.utils.random(nacl.pwhash.argon2id.SALTBYTES)
    kek = _derive_kek(phrase, salt, _KDF_OPSLIMIT, _KDF_MEMLIMIT)
    return RecoveryWrap(
        wrapped_key=SecretBox(kek).encrypt(master_key),
        kdf_salt=salt,
        kdf_params=json.dumps({"m": _KDF_MEMLIMIT, "t": _KDF_OPSLIMIT, "p": 1}),
    )


def unwrap_master_key_with_phrase(wrap: RecoveryWrap, phrase: str) -> bytes:
    """Recover MK on a fresh device from the phrase + the stored KDF context."""
    try:
        params = json.loads(wrap.kdf_params)
        opslimit, memlimit = int(params["t"]), int(params["m"])
    except (ValueError, KeyError, TypeError) as err:
        raise CloudCryptoError("malformed KDF parameters") from err
    kek = _derive_kek(phrase, wrap.kdf_salt, opslimit, memlimit)
    try:
        return SecretBox(kek).decrypt(wrap.wrapped_key)
    except nacl.exceptions.CryptoError as err:
        raise CloudCryptoError("recovery phrase does not unlock this account key") from err


# ---------------------------------------------------------------------------
# Blob envelope


def _build_aad(object_id: str, role: str, version: int) -> bytes:
    # Binds a blob to its logical slot. Unambiguous because object_id is a UUID
    # and role is a fixed token — neither contains ":".
    return f"{object_id}:{role}:{version}".encode()


def encrypt_blob(plaintext: bytes, master_key: bytes, *, object_id: str, role: str, version: int) -> bytes:
    """Produce a self-describing VBX1 envelope: a fresh Content Key wrapped by
    MK rides in the header, and the AAD pins the blob to (object, role, version)."""
    content_key = secrets.token_bytes(KEY_BYTES)
    wrapped_ck = SecretBox(master_key).encrypt(content_key)
    nonce = nacl.utils.random(_NONCE_BYTES)
    ciphertext = nacl.bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
        plaintext,
        _build_aad(object_id, role, version),
        nonce,
        content_key,
    )
    return b"".join(
        [
            ENVELOPE_MAGIC,
            bytes([ALG_XCHACHA20_POLY1305]),
            len(wrapped_ck).to_bytes(_WRAPPED_CK_LEN_BYTES, "big"),
            wrapped_ck,
            nonce,
            ciphertext,
        ]
    )


def decrypt_blob(envelope: bytes, master_key: bytes, *, object_id: str, role: str, version: int) -> bytes:
    """Open a VBX1 envelope. Raises CloudCryptoError if the envelope is
    malformed, the key is wrong, the ciphertext was modified, or the blob was
    served for a different (object, role, version) slot."""
    offset = len(ENVELOPE_MAGIC)
    if envelope[:offset] != ENVELOPE_MAGIC:
        raise CloudCryptoError("not a VBX1 envelope")
    if len(envelope) < offset + 1 + _WRAPPED_CK_LEN_BYTES:
        raise CloudCryptoError("envelope truncated")
    alg = envelope[offset]
    if alg != ALG_XCHACHA20_POLY1305:
        raise CloudCryptoError(f"unsupported envelope algorithm {alg}")
    offset += 1

    wrapped_len = int.from_bytes(envelope[offset : offset + _WRAPPED_CK_LEN_BYTES], "big")
    offset += _WRAPPED_CK_LEN_BYTES
    wrapped_ck = envelope[offset : offset + wrapped_len]
    offset += wrapped_len
    nonce = envelope[offset : offset + _NONCE_BYTES]
    offset += _NONCE_BYTES
    ciphertext = envelope[offset:]
    if len(wrapped_ck) != wrapped_len or len(nonce) != _NONCE_BYTES or not ciphertext:
        raise CloudCryptoError("envelope truncated")

    try:
        content_key = SecretBox(master_key).decrypt(wrapped_ck)
    except nacl.exceptions.CryptoError as err:
        raise CloudCryptoError("master key does not unwrap this blob's content key") from err
    try:
        return nacl.bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
            ciphertext,
            _build_aad(object_id, role, version),
            nonce,
            content_key,
        )
    except nacl.exceptions.CryptoError as err:
        raise CloudCryptoError("blob failed authentication (tampered, or served for the wrong slot)") from err
