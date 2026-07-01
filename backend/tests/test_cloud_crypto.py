"""Tests for the cloud E2E crypto primitives (services/cloud_crypto.py)."""

import pytest

from backend.services.cloud_crypto import (
    ALG_XCHACHA20_POLY1305,
    ENVELOPE_MAGIC,
    CloudCryptoError,
    RecoveryWrap,
    decrypt_blob,
    encrypt_blob,
    generate_device_keypair,
    generate_master_key,
    generate_recovery_phrase,
    unwrap_master_key_for_device,
    unwrap_master_key_with_phrase,
    validate_recovery_phrase,
    wrap_master_key_for_device,
    wrap_master_key_with_phrase,
)

SLOT = {"object_id": "0c8f6f4e-9f5a-4a2f-8f6a-1d2e3f4a5b6c", "role": "audio", "version": 3}


class TestMasterKey:
    def test_master_keys_are_random_32_bytes(self):
        a, b = generate_master_key(), generate_master_key()
        assert len(a) == 32
        assert a != b


class TestDeviceWrap:
    def test_round_trip(self):
        mk = generate_master_key()
        private, public = generate_device_keypair()
        assert unwrap_master_key_for_device(wrap_master_key_for_device(mk, public), private) == mk

    def test_wrong_device_key_fails(self):
        mk = generate_master_key()
        _, public = generate_device_keypair()
        other_private, _ = generate_device_keypair()
        with pytest.raises(CloudCryptoError):
            unwrap_master_key_for_device(wrap_master_key_for_device(mk, public), other_private)


class TestRecoveryPhrase:
    def test_phrase_is_valid_bip39(self):
        phrase = generate_recovery_phrase()
        assert len(phrase.split()) == 12
        assert validate_recovery_phrase(phrase)

    def test_typo_fails_checksum(self):
        words = generate_recovery_phrase().split()
        words[0] = "abandon" if words[0] != "abandon" else "ability"
        assert not validate_recovery_phrase(" ".join(words))

    def test_round_trip(self):
        mk = generate_master_key()
        phrase = generate_recovery_phrase()
        assert unwrap_master_key_with_phrase(wrap_master_key_with_phrase(mk, phrase), phrase) == mk

    def test_normalization_tolerates_case_and_whitespace(self):
        mk = generate_master_key()
        phrase = generate_recovery_phrase()
        wrap = wrap_master_key_with_phrase(mk, phrase)
        sloppy = f"  {phrase.upper().replace(' ', '   ')} \n"
        assert unwrap_master_key_with_phrase(wrap, sloppy) == mk

    def test_wrong_phrase_fails(self):
        wrap = wrap_master_key_with_phrase(generate_master_key(), generate_recovery_phrase())
        with pytest.raises(CloudCryptoError):
            unwrap_master_key_with_phrase(wrap, generate_recovery_phrase())

    def test_malformed_kdf_params_fail(self):
        wrap = wrap_master_key_with_phrase(generate_master_key(), generate_recovery_phrase())
        broken = RecoveryWrap(wrapped_key=wrap.wrapped_key, kdf_salt=wrap.kdf_salt, kdf_params="not json")
        with pytest.raises(CloudCryptoError):
            unwrap_master_key_with_phrase(broken, generate_recovery_phrase())


class TestEnvelope:
    def test_round_trip(self):
        mk = generate_master_key()
        plaintext = b"capture transcript \xf0\x9f\x8e\x99 and some audio bytes" * 100
        envelope = encrypt_blob(plaintext, mk, **SLOT)
        assert envelope[:4] == ENVELOPE_MAGIC
        assert envelope[4] == ALG_XCHACHA20_POLY1305
        assert decrypt_blob(envelope, mk, **SLOT) == plaintext

    def test_fresh_content_key_per_blob(self):
        mk = generate_master_key()
        assert encrypt_blob(b"same", mk, **SLOT) != encrypt_blob(b"same", mk, **SLOT)

    def test_wrong_master_key_fails(self):
        envelope = encrypt_blob(b"secret", generate_master_key(), **SLOT)
        with pytest.raises(CloudCryptoError):
            decrypt_blob(envelope, generate_master_key(), **SLOT)

    @pytest.mark.parametrize(
        "slot",
        [
            {**SLOT, "object_id": "11111111-2222-3333-4444-555555555555"},
            {**SLOT, "role": "avatar"},
            {**SLOT, "version": 4},
        ],
    )
    def test_wrong_slot_fails_aad(self, slot):
        mk = generate_master_key()
        envelope = encrypt_blob(b"secret", mk, **SLOT)
        with pytest.raises(CloudCryptoError):
            decrypt_blob(envelope, mk, **slot)

    def test_tampered_ciphertext_fails(self):
        mk = generate_master_key()
        envelope = bytearray(encrypt_blob(b"secret", mk, **SLOT))
        envelope[-1] ^= 0x01
        with pytest.raises(CloudCryptoError):
            decrypt_blob(bytes(envelope), mk, **SLOT)

    def test_bad_magic_and_truncation_fail(self):
        mk = generate_master_key()
        envelope = encrypt_blob(b"secret", mk, **SLOT)
        with pytest.raises(CloudCryptoError):
            decrypt_blob(b"NOPE" + envelope[4:], mk, **SLOT)
        with pytest.raises(CloudCryptoError):
            decrypt_blob(envelope[:20], mk, **SLOT)

    def test_unknown_algorithm_fails(self):
        mk = generate_master_key()
        envelope = bytearray(encrypt_blob(b"secret", mk, **SLOT))
        envelope[4] = 99
        with pytest.raises(CloudCryptoError):
            decrypt_blob(bytes(envelope), mk, **SLOT)

    def test_empty_plaintext_round_trips(self):
        mk = generate_master_key()
        assert decrypt_blob(encrypt_blob(b"", mk, **SLOT), mk, **SLOT) == b""
