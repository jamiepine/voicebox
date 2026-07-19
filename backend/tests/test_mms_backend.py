"""
Tests for the MMS-TTS backend (Romanian).

Unit tests cover the Romanian diacritics normalization (the mms-tts-ron
vocab mixes comma-below ș with cedilla ţ, and the character-level
VitsTokenizer silently drops out-of-vocab characters) and the engine
registration surfaces (model config registry, backend factory, request
model regexes, preset voice endpoints).

The end-to-end generation test downloads the ~150MB model on first run
and is opt-in:

    VOICEBOX_MMS_E2E=1 python -m pytest backend/tests/test_mms_backend.py -v
"""

import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.backends import (
    TTS_ENGINES,
    get_model_config,
    get_tts_backend_for_engine,
    reset_backends,
)
from backend.backends.mms_backend import (
    MMS_DEFAULT_VOICE,
    MMS_HF_REPOS,
    MMS_SAMPLE_RATE,
    MMS_VOICES,
    MMSTTSBackend,
    normalize_romanian_text,
)

S_CEDILLA = "ş"  # ş — not in the mms-tts-ron vocab
S_CEDILLA_UPPER = "Ş"  # Ş
S_COMMA = "ș"  # ș — in vocab
S_COMMA_UPPER = "Ș"  # Ș
T_CEDILLA = "ţ"  # ţ — in vocab
T_CEDILLA_UPPER = "Ţ"  # Ţ
T_COMMA = "ț"  # ț — not in vocab
T_COMMA_UPPER = "Ț"  # Ț
COMBINING_COMMA_BELOW = "̦"
COMBINING_BREVE = "̆"

MMS_PRESET_PROMPT = {
    "voice_type": "preset",
    "preset_engine": "mms",
    "preset_voice_id": MMS_DEFAULT_VOICE,
}


class TestRomanianDiacriticsNormalization:
    """The tokenizer drops unknown chars silently — every real-world
    diacritic variant must be mapped onto the form in the vocab."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (S_CEDILLA, S_COMMA),
            (S_CEDILLA_UPPER, S_COMMA_UPPER),
            (T_COMMA, T_CEDILLA),
            (T_COMMA_UPPER, T_CEDILLA_UPPER),
        ],
    )
    def test_wrong_variant_mapped_to_vocab_form(self, raw, expected):
        assert normalize_romanian_text(raw) == expected

    @pytest.mark.parametrize("char", [S_COMMA, T_CEDILLA, "ă", "â", "î", "a", "b", " "])
    def test_vocab_forms_pass_through_unchanged(self, char):
        assert normalize_romanian_text(char) == char

    def test_nfd_sequences_are_composed(self):
        # s/t + combining comma below compose (via NFC) to the comma-below
        # letters, which then go through the same variant mapping.
        assert normalize_romanian_text("s" + COMBINING_COMMA_BELOW) == S_COMMA
        assert normalize_romanian_text("t" + COMBINING_COMMA_BELOW) == T_CEDILLA
        assert normalize_romanian_text("a" + COMBINING_BREVE) == "ă"

    def test_mixed_sentence_contains_only_vocab_diacritics(self):
        # Both variant families in one string, as found in the wild.
        text = f"{S_CEDILLA_UPPER}tii c{S_CEDILLA} {T_COMMA}ara {T_CEDILLA}ine pa{S_COMMA}ii"
        normalized = normalize_romanian_text(text)
        lowered = normalized.lower()
        assert S_CEDILLA not in lowered
        assert T_COMMA not in lowered
        assert lowered.count(S_COMMA) == 3
        assert lowered.count(T_CEDILLA) == 2

    def test_plain_text_untouched(self):
        text = "Salut, ce mai faci? 1-2!"
        assert normalize_romanian_text(text) == text

    def test_unknown_chars_are_preserved(self):
        # Normalization is not lossy — dropping out-of-vocab characters is
        # the tokenizer's job, not ours.
        text = "kiwi & yoga 42"
        assert normalize_romanian_text(text) == text

    def test_output_is_nfc(self):
        decomposed = unicodedata.normalize("NFD", "Bună ziua, țară")
        assert unicodedata.is_normalized("NFC", normalize_romanian_text(decomposed))


class TestMMSRegistration:
    def test_engine_listed(self):
        assert "mms" in TTS_ENGINES

    def test_model_config_resolves(self):
        cfg = get_model_config("mms-tts-ron")
        assert cfg is not None
        assert cfg.engine == "mms"
        assert cfg.hf_repo_id == "facebook/mms-tts-ron"
        assert cfg.languages == ["ro"]
        assert cfg.size_mb == 150
        assert cfg.supports_instruct is False
        assert cfg.needs_trim is False

    def test_backend_factory_returns_mms_backend(self):
        try:
            backend = get_tts_backend_for_engine("mms")
            assert isinstance(backend, MMSTTSBackend)
            # Factory caches instances per engine
            assert get_tts_backend_for_engine("mms") is backend
        finally:
            reset_backends()

    def test_voice_catalog_shape(self):
        # Same tuple shape as KOKORO_VOICES: (voice_id, name, gender, lang)
        for voice_id, name, gender, lang in MMS_VOICES:
            assert voice_id
            assert name
            assert gender in ("male", "female")
            assert lang in MMS_HF_REPOS
        assert MMS_DEFAULT_VOICE in {v[0] for v in MMS_VOICES}

    def test_preset_voice_ids_service(self):
        from backend.services.profiles import _get_preset_voice_ids

        assert _get_preset_voice_ids("mms") == {MMS_DEFAULT_VOICE}

    async def test_preset_voices_route(self):
        # routes/profiles imports ..app at module scope; importing the app
        # first (the normal boot order) avoids a circular import.
        import backend.app  # noqa: F401 -- side-effect import initializes routers
        from backend.routes.profiles import list_preset_voices

        result = await list_preset_voices("mms")
        assert result["engine"] == "mms"
        assert result["voices"] == [
            {
                "voice_id": MMS_DEFAULT_VOICE,
                "name": "Romanian (MMS)",
                "gender": "male",
                "language": "ro",
            }
        ]


class TestRequestModelRegexes:
    def test_generation_request_accepts_ro_and_mms(self):
        from backend.models import GenerationRequest

        req = GenerationRequest(profile_id="p1", text="Bună ziua", language="ro", engine="mms")
        assert req.language == "ro"
        assert req.engine == "mms"

    def test_generation_request_rejects_unknown_engine(self):
        from pydantic import ValidationError

        from backend.models import GenerationRequest

        with pytest.raises(ValidationError):
            GenerationRequest(profile_id="p1", text="hi", engine="mms2")

    def test_profile_create_accepts_ro(self):
        from backend.models import VoiceProfileCreate

        profile = VoiceProfileCreate(
            name="Vocea",
            language="ro",
            voice_type="preset",
            preset_engine="mms",
            preset_voice_id=MMS_DEFAULT_VOICE,
            default_engine="mms",
        )
        assert profile.language == "ro"
        assert profile.default_engine == "mms"

    def test_speak_request_accepts_ro_and_mms(self):
        from backend.models import SpeakRequest

        req = SpeakRequest(text="Bună", engine="mms", language="ro")
        assert req.engine == "mms"
        assert req.language == "ro"

    def test_mcp_binding_accepts_mms(self):
        from backend.models import MCPClientBindingUpsert

        binding = MCPClientBindingUpsert(client_id="client-1", default_engine="mms")
        assert binding.default_engine == "mms"


class TestMMSBackendUnit:
    def test_initial_state(self):
        backend = MMSTTSBackend()
        assert not backend.is_loaded()
        assert backend._get_model_path("default") == "facebook/mms-tts-ron"

    async def test_create_voice_prompt_returns_preset_fallback(self):
        backend = MMSTTSBackend()
        prompt, was_cached = await backend.create_voice_prompt("/tmp/none.wav", "text")
        assert prompt == MMS_PRESET_PROMPT
        assert was_cached is False

    def test_unload_without_load_is_noop(self):
        backend = MMSTTSBackend()
        backend.unload_model()
        assert not backend.is_loaded()


RUN_MMS_E2E = os.environ.get("VOICEBOX_MMS_E2E") == "1"


@pytest.mark.skipif(not RUN_MMS_E2E, reason="set VOICEBOX_MMS_E2E=1 to run (downloads ~150MB model)")
class TestMMSGenerationE2E:
    """Full generation through the real model — both diacritic conventions
    must produce identical tokens and valid audio at 16kHz."""

    async def test_generate_romanian_with_both_diacritic_variants(self):
        backend = MMSTTSBackend()
        try:
            text = (
                "Bună ziua! Ce mai faceți? Știți că țara noastră e frumoasă? "
                f"{S_CEDILLA_UPPER}i {T_CEDILLA}ine{T_COMMA}i minte pa{S_CEDILLA}ii."
            )
            audio, sample_rate = await backend.generate(text, MMS_PRESET_PROMPT, "ro")

            assert sample_rate == MMS_SAMPLE_RATE == 16000
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert audio.ndim == 1
            assert len(audio) > sample_rate, "expected more than 1s of audio"
            assert not np.isnan(audio).any()
            assert float(np.abs(audio).max()) > 0.01, "audio should not be silence"
        finally:
            backend.unload_model()

    async def test_diacritic_variants_tokenize_identically(self):
        backend = MMSTTSBackend()
        try:
            await backend.load_model()
            tokenizer = backend._tokenizer

            cedilla_ids = tokenizer(normalize_romanian_text("ştiţi paşii ţară"))["input_ids"]
            comma_ids = tokenizer(normalize_romanian_text("știți pașii țară"))["input_ids"]
            assert cedilla_ids == comma_ids
            # Nothing was dropped: with add_blank the tokenizer interleaves a
            # blank between characters -> 2 * len(text) + 1 tokens.
            assert len(comma_ids) == 2 * len("știți pașii țară") + 1
        finally:
            backend.unload_model()

    async def test_seeded_generation_is_deterministic(self):
        backend = MMSTTSBackend()
        try:
            first, _ = await backend.generate("Bună ziua!", MMS_PRESET_PROMPT, "ro", seed=42)
            second, _ = await backend.generate("Bună ziua!", MMS_PRESET_PROMPT, "ro", seed=42)
            np.testing.assert_allclose(first, second)
        finally:
            backend.unload_model()

    async def test_fully_out_of_vocab_text_returns_silence(self):
        backend = MMSTTSBackend()
        try:
            audio, sample_rate = await backend.generate("!!!", MMS_PRESET_PROMPT, "ro")
            assert sample_rate == MMS_SAMPLE_RATE
            assert len(audio) == sample_rate
            assert not audio.any()
        finally:
            backend.unload_model()
