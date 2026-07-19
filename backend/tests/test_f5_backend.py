"""
Tests for the F5-TTS Romanian backend.

Unit tests cover the Romanian diacritics normalization (the fine-tune's
vocab lacks cedilla ţ and f5-tts maps out-of-vocab characters to space),
the reference-audio trimming (F5 degrades with references over 12s), and
the engine registration surfaces (model config registry, backend factory,
request model regexes, cloning-profile validation).

The end-to-end cloning test downloads the ~1.2GB checkpoint on first run
and is opt-in:

    VOICEBOX_F5_E2E=1 python -m pytest backend/tests/test_f5_backend.py -v
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
from backend.backends.f5_backend import (
    F5_HF_REPO,
    F5_MAX_REF_SECONDS,
    F5_SAMPLE_RATE,
    F5_VOCAB_FILE,
    F5TTSBackend,
    normalize_romanian_text,
    trim_reference_audio,
    trim_reference_text,
)

S_CEDILLA = "ş"  # ş — in the F5 vocab, but not the canonical form
S_CEDILLA_UPPER = "Ş"  # Ş
S_COMMA = "ș"  # ș — in vocab (canonical)
S_COMMA_UPPER = "Ș"  # Ș
T_CEDILLA = "ţ"  # ţ — NOT in the F5 vocab
T_CEDILLA_UPPER = "Ţ"  # Ţ
T_COMMA = "ț"  # ț — in vocab (canonical)
T_COMMA_UPPER = "Ț"  # Ț
COMBINING_COMMA_BELOW = "̦"
COMBINING_BREVE = "̆"


class TestRomanianDiacriticsNormalization:
    """f5-tts maps out-of-vocab chars to index 0 (space) silently — both
    real-world diacritic families must land on the comma-below vocab forms.
    Note the mapping direction is the opposite of the MMS backend's: this
    vocab keeps comma-below ț and lacks cedilla ţ."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (S_CEDILLA, S_COMMA),
            (S_CEDILLA_UPPER, S_COMMA_UPPER),
            (T_CEDILLA, T_COMMA),
            (T_CEDILLA_UPPER, T_COMMA_UPPER),
        ],
    )
    def test_cedilla_variants_mapped_to_comma_below(self, raw, expected):
        assert normalize_romanian_text(raw) == expected

    @pytest.mark.parametrize("char", [S_COMMA, T_COMMA, "ă", "â", "î", "Ă", "Â", "Î", "a", " "])
    def test_vocab_forms_pass_through_unchanged(self, char):
        assert normalize_romanian_text(char) == char

    def test_nfd_sequences_are_composed(self):
        # s/t + combining comma below compose (via NFC) to the comma-below
        # letters, which are already the vocab forms.
        assert normalize_romanian_text("s" + COMBINING_COMMA_BELOW) == S_COMMA
        assert normalize_romanian_text("t" + COMBINING_COMMA_BELOW) == T_COMMA
        assert normalize_romanian_text("a" + COMBINING_BREVE) == "ă"

    def test_mixed_sentence_contains_only_vocab_diacritics(self):
        text = f"{S_CEDILLA_UPPER}tii c{S_CEDILLA} {T_CEDILLA}ara {T_COMMA}ine pa{S_COMMA}ii"
        normalized = normalize_romanian_text(text)
        lowered = normalized.lower()
        assert S_CEDILLA not in lowered
        assert T_CEDILLA not in lowered
        assert lowered.count(S_COMMA) == 3
        assert lowered.count(T_COMMA) == 2

    def test_plain_text_untouched(self):
        text = "Salut, ce mai faci? 1-2!"
        assert normalize_romanian_text(text) == text

    def test_output_is_nfc(self):
        decomposed = unicodedata.normalize("NFD", "Bună ziua, țară")
        assert unicodedata.is_normalized("NFC", normalize_romanian_text(decomposed))


class TestVocabDiacriticCoverage:
    """Pin the vocab facts the normalization is built on — if the upstream
    vocab.txt ever changes, this fails loudly instead of silently dropping
    diacritics at generation time."""

    @pytest.fixture(scope="class")
    def vocab(self) -> set[str]:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            path = hf_hub_download(F5_HF_REPO, F5_VOCAB_FILE, local_files_only=True)
        except LocalEntryNotFoundError:
            pytest.skip(f"{F5_HF_REPO} vocab not cached locally")
        return set(Path(path).read_text(encoding="utf-8").split("\n"))

    def test_comma_below_family_in_vocab(self, vocab):
        for char in (S_COMMA, S_COMMA_UPPER, T_COMMA, T_COMMA_UPPER):
            assert char in vocab, f"{char!r} missing from vocab — normalization target invalid"

    def test_cedilla_t_not_in_vocab(self, vocab):
        # The reason the normalization exists: cedilla ţ is absent.
        assert T_CEDILLA not in vocab
        assert T_CEDILLA_UPPER not in vocab

    def test_other_romanian_diacritics_in_vocab(self, vocab):
        for char in ("ă", "Ă", "â", "Â", "î", "Î"):
            assert char in vocab

    def test_normalized_text_fully_covered(self, vocab):
        text = normalize_romanian_text("Ştiţi că ţara Ţării are paşi şi înţelegere, Bună ziua!")
        missing = {c for c in text if c not in vocab and not c.isspace()}
        assert not missing, f"normalized text still has out-of-vocab chars: {missing}"


class TestReferenceTrimming:
    SR = 24000

    def _speech_like(self, seconds: float, level: float = 0.3) -> np.ndarray:
        rng = np.random.default_rng(0)
        return (rng.standard_normal(int(seconds * self.SR)) * level).astype(np.float32)

    def test_short_audio_untouched(self):
        audio = self._speech_like(5.0)
        trimmed, kept = trim_reference_audio(audio, self.SR)
        assert kept == 1.0
        assert trimmed is audio

    def test_audio_at_limit_untouched(self):
        audio = self._speech_like(F5_MAX_REF_SECONDS)
        trimmed, kept = trim_reference_audio(audio, self.SR)
        assert kept == 1.0
        assert len(trimmed) == len(audio)

    def test_cut_lands_in_silence_gap(self):
        # 20s of "speech" with a known 500ms silence gap at 9.5s-10.0s —
        # the quietest-window search must cut inside the gap.
        audio = self._speech_like(20.0)
        gap_start, gap_end = int(9.5 * self.SR), int(10.0 * self.SR)
        audio[gap_start:gap_end] = 0.0

        trimmed, kept = trim_reference_audio(audio, self.SR)

        assert gap_start <= len(trimmed) <= gap_end
        assert 0.0 < kept < 1.0
        assert kept == pytest.approx(len(trimmed) / len(audio))

    def test_result_never_exceeds_max(self):
        # No gaps at all — uniform noise. The cut must still be <= 12s.
        audio = self._speech_like(30.0)
        trimmed, _ = trim_reference_audio(audio, self.SR)
        assert len(trimmed) <= F5_MAX_REF_SECONDS * self.SR

    def test_cut_not_before_min_seconds(self):
        # Silence at the very start must not produce a uselessly short ref.
        audio = self._speech_like(20.0)
        audio[: int(2 * self.SR)] = 0.0
        trimmed, _ = trim_reference_audio(audio, self.SR)
        assert len(trimmed) >= 8.0 * self.SR

    def test_text_untrimmed_when_audio_kept(self):
        assert trim_reference_text("O propoziție întreagă.", 1.0) == "O propoziție întreagă."

    def test_text_proportionally_truncated_at_word_boundary(self):
        text = "unu doi trei patru cinci șase șapte opt nouă zece"
        result = trim_reference_text(text, 0.5)
        assert result == "unu doi trei patru cinci"

    def test_text_trailing_punctuation_stripped(self):
        text = "unu doi trei patru, cinci șase opt nouă"
        result = trim_reference_text(text, 0.5)
        assert result == "unu doi trei patru"

    def test_text_never_empty(self):
        assert trim_reference_text("cuvânt lung aici", 0.01) == "cuvânt"


class TestF5Registration:
    def test_engine_listed(self):
        assert "f5" in TTS_ENGINES

    def test_model_config_resolves(self):
        cfg = get_model_config("f5-tts-romanian")
        assert cfg is not None
        assert cfg.engine == "f5"
        assert cfg.hf_repo_id == F5_HF_REPO
        assert cfg.languages == ["ro", "en"]
        assert cfg.size_mb == 1200
        assert cfg.supports_instruct is False
        assert cfg.needs_trim is False

    def test_backend_factory_returns_f5_backend(self):
        try:
            backend = get_tts_backend_for_engine("f5")
            assert isinstance(backend, F5TTSBackend)
            assert get_tts_backend_for_engine("f5") is backend
        finally:
            reset_backends()

    def test_f5_is_a_cloning_engine(self):
        from backend.services.profiles import CLONING_ENGINES, _get_preset_voice_ids

        assert "f5" in CLONING_ENGINES
        # No preset voices — F5 clones from reference audio only.
        assert _get_preset_voice_ids("f5") == set()

    def test_cloned_profile_validation_accepts_f5(self):
        from types import SimpleNamespace

        from backend.services.profiles import validate_profile_engine

        profile = SimpleNamespace(id="p1", voice_type="cloned")
        validate_profile_engine(profile, "f5")  # must not raise


class TestRequestModelRegexes:
    def test_generation_request_accepts_ro_and_f5(self):
        from backend.models import GenerationRequest

        req = GenerationRequest(profile_id="p1", text="Bună ziua", language="ro", engine="f5")
        assert req.language == "ro"
        assert req.engine == "f5"

    def test_generation_request_rejects_unknown_engine(self):
        from pydantic import ValidationError

        from backend.models import GenerationRequest

        with pytest.raises(ValidationError):
            GenerationRequest(profile_id="p1", text="hi", engine="f5x")

    def test_profile_create_accepts_f5_default_engine(self):
        from backend.models import VoiceProfileCreate

        profile = VoiceProfileCreate(name="Vocea", language="ro", default_engine="f5")
        assert profile.default_engine == "f5"

    def test_speak_request_accepts_ro_and_f5(self):
        from backend.models import SpeakRequest

        req = SpeakRequest(text="Bună", engine="f5", language="ro")
        assert req.engine == "f5"
        assert req.language == "ro"

    def test_mcp_binding_accepts_f5(self):
        from backend.models import MCPClientBindingUpsert

        binding = MCPClientBindingUpsert(client_id="client-1", default_engine="f5")
        assert binding.default_engine == "f5"


class TestF5BackendUnit:
    def test_initial_state(self):
        backend = F5TTSBackend()
        assert not backend.is_loaded()
        assert backend._get_model_path("default") == F5_HF_REPO

    async def test_create_voice_prompt_stores_reference(self):
        backend = F5TTSBackend()
        prompt, was_cached = await backend.create_voice_prompt("/tmp/sample.wav", "Bună ziua")
        assert prompt == {"ref_audio": "/tmp/sample.wav", "ref_text": "Bună ziua"}
        assert was_cached is False

    def test_unload_without_load_is_noop(self):
        backend = F5TTSBackend()
        backend.unload_model()
        assert not backend.is_loaded()

    async def test_generate_rejects_missing_reference_audio(self):
        backend = F5TTSBackend()
        with pytest.raises(ValueError, match="reference audio"):
            await backend.generate("text", {"ref_audio": "/nonexistent.wav", "ref_text": "x"}, "ro")

    async def test_generate_rejects_empty_reference_text(self, tmp_path):
        # Empty ref_text would trigger f5-tts's Whisper auto-transcription
        # (a ~1.6GB surprise download) — the backend must refuse instead.
        import soundfile as sf

        wav = tmp_path / "ref.wav"
        sf.write(str(wav), np.zeros(24000, dtype=np.float32), 24000)
        backend = F5TTSBackend()
        with pytest.raises(ValueError, match="transcript"):
            await backend.generate("text", {"ref_audio": str(wav), "ref_text": "  "}, "ro")


RUN_F5_E2E = os.environ.get("VOICEBOX_F5_E2E") == "1"

ROMANIAN_REF_SENTENCE = "Bună ziua, mă numesc Adrian și locuiesc în București de mulți ani."


@pytest.mark.skipif(not RUN_F5_E2E, reason="set VOICEBOX_F5_E2E=1 to run (downloads ~1.2GB model)")
class TestF5CloningE2E:
    """Full cloning through the real model. The reference audio is produced
    in-test by the MMS engine (cached, CPU-realtime) so the test needs no
    fixture files: one Romanian sentence serves as both ref audio and
    ref text."""

    @pytest.fixture(scope="class")
    def reference(self, tmp_path_factory) -> tuple[str, str]:
        import asyncio

        import soundfile as sf

        from backend.backends.mms_backend import MMSTTSBackend

        mms = MMSTTSBackend()
        try:
            audio, sample_rate = asyncio.run(
                mms.generate(
                    ROMANIAN_REF_SENTENCE,
                    {"voice_type": "preset", "preset_engine": "mms", "preset_voice_id": "mms_ro_default"},
                    "ro",
                    seed=7,
                )
            )
        finally:
            mms.unload_model()
        path = tmp_path_factory.mktemp("f5_ref") / "ref.wav"
        sf.write(str(path), audio, sample_rate)
        return str(path), ROMANIAN_REF_SENTENCE

    @pytest.fixture(scope="class")
    def backend(self):
        backend = F5TTSBackend()
        yield backend
        backend.unload_model()

    async def test_clone_generates_romanian_audio(self, backend, reference):
        ref_audio, ref_text = reference
        prompt, _ = await backend.create_voice_prompt(ref_audio, ref_text)

        audio, sample_rate = await backend.generate(
            "Ștefan cel Mare a domnit în Moldova aproape cincizeci de ani.",
            prompt,
            "ro",
            seed=42,
        )

        assert sample_rate == F5_SAMPLE_RATE == 24000
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) > sample_rate, "expected more than 1s of audio"
        assert not np.isnan(audio).any()
        assert not np.isinf(audio).any()
        assert float(np.abs(audio).max()) > 0.01, "audio should not be silence"

    async def test_seeded_generation_is_deterministic(self, backend, reference):
        ref_audio, ref_text = reference
        prompt, _ = await backend.create_voice_prompt(ref_audio, ref_text)

        first, _ = await backend.generate("Bună ziua!", prompt, "ro", seed=42)
        second, _ = await backend.generate("Bună ziua!", prompt, "ro", seed=42)
        np.testing.assert_allclose(first, second)
