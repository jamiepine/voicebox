"""
F5-TTS Romanian backend implementation.

Wraps the community Romanian fine-tune of F5-TTS
(``MihaiPopa-1/F5-TTS-Romanian``, Apache-2.0, base ``SWivid/F5-TTS``) for
zero-shot voice cloning — the first engine offering true Romanian cloning
from a reference sample. The fine-tune retains English.

Like Chatterbox, this is a cloning engine: the voice prompt stores the
reference audio path and transcript, and the audio is processed at
generation time. References longer than 12 seconds are trimmed at the
quietest moment (F5 conditioning degrades with long references) and the
transcript is proportionally truncated to keep text/audio correspondence.

Output is 24 kHz mono float32 via the Vocos vocoder, which f5-tts
downloads from ``charactr/vocos-mel-24khz`` on first load.
"""

import asyncio
import hashlib
import logging
import unicodedata
from pathlib import Path

import numpy as np

from .. import config
from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    model_load_progress,
)

logger = logging.getLogger(__name__)

F5_HF_REPO = "MihaiPopa-1/F5-TTS-Romanian"
F5_CKPT_FILE = "model_750_pruned.safetensors"
F5_VOCAB_FILE = "vocab.txt"
# f5_tts.api.F5TTS downloads the Vocos vocoder from this repo on init;
# _is_model_cached must account for it so the UI "downloaded" state is truthful.
F5_VOCODER_HF_REPO = "charactr/vocos-mel-24khz"
F5_SAMPLE_RATE = 24000
F5_NFE_STEPS = 32

# F5 conditioning degrades with references over ~12s (upstream also hard-clips
# at 12s). Trim at the quietest 300ms window found between 8s and 12s so the
# cut lands in a natural pause and upstream's cruder clipper never fires.
F5_MAX_REF_SECONDS = 12.0
F5_MIN_REF_SECONDS = 8.0
F5_TRIM_WINDOW_SECONDS = 0.3
_TRIM_HOP_SECONDS = 0.01

# The fine-tune's vocab.txt contains comma-below ș (U+0219) and ț (U+021B)
# plus cedilla ş (U+015F), but NOT cedilla ţ (U+0163). f5-tts maps
# out-of-vocab characters to index 0 (space), silently dropping them, so
# both real-world diacritic families must be mapped onto the comma-below
# forms the model was trained on. (ş is technically in the vocab but the
# comma-below family is the canonical Romanian form; unifying onto it
# keeps ref/gen text consistent.)
_RO_DIACRITICS_TRANSLATION = str.maketrans(
    {
        "ş": "ș",  # ş (s-cedilla) -> ș (s-comma-below)
        "Ş": "Ș",  # Ş (S-cedilla) -> Ș (S-comma-below)
        "ţ": "ț",  # ţ (t-cedilla, NOT in vocab) -> ț (t-comma-below)
        "Ţ": "Ț",  # Ţ (T-cedilla, NOT in vocab) -> Ț (T-comma-below)
    }
)


def normalize_romanian_text(text: str) -> str:
    """Normalize Romanian text to the diacritic forms in the F5 vocab.

    Applies NFC normalization first (composing any decomposed
    letter + combining-mark sequences), then maps cedilla s and t
    variants onto the comma-below forms the fine-tune was trained on,
    so no diacritic is silently dropped by the character tokenizer.
    """
    return unicodedata.normalize("NFC", text).translate(_RO_DIACRITICS_TRANSLATION)


def trim_reference_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_seconds: float = F5_MAX_REF_SECONDS,
    min_seconds: float = F5_MIN_REF_SECONDS,
    window_seconds: float = F5_TRIM_WINDOW_SECONDS,
) -> tuple[np.ndarray, float]:
    """Trim reference audio to at most ``max_seconds`` at the quietest moment.

    Slides a ``window_seconds`` RMS window (10ms hop) over the span between
    ``min_seconds`` and ``max_seconds`` and cuts at the centre of the
    quietest window, so the cut lands in a pause rather than mid-word.

    Args:
        audio: Mono audio array.
        sample_rate: Sample rate of ``audio``.
        max_seconds: Hard upper bound for the trimmed length.
        min_seconds: Earliest allowed cut point.
        window_seconds: RMS window length used to find the quietest moment.

    Returns:
        Tuple of (trimmed_audio, kept_fraction). ``kept_fraction`` is 1.0
        when the audio was already short enough and untouched.
    """
    if len(audio) <= max_seconds * sample_rate:
        return audio, 1.0

    start = int(min_seconds * sample_rate)
    end = min(int(max_seconds * sample_rate), len(audio))
    window = max(1, int(window_seconds * sample_rate))
    hop = max(1, int(_TRIM_HOP_SECONDS * sample_rate))

    segment = audio[start:end].astype(np.float64)
    window_starts = np.arange(0, len(segment) - window, hop)
    if len(window_starts) == 0:
        cut = end
    else:
        # Windowed energy via cumulative sum — O(n) instead of a python loop.
        cumulative = np.concatenate(([0.0], np.cumsum(segment * segment)))
        energies = cumulative[window_starts + window] - cumulative[window_starts]
        quietest = int(window_starts[np.argmin(energies)])
        cut = start + quietest + window // 2

    return audio[:cut], cut / len(audio)


def trim_reference_text(text: str, kept_fraction: float) -> str:
    """Truncate a transcript to match trimmed reference audio.

    Keeps the leading ``kept_fraction`` of words (assuming roughly constant
    speech rate) and cuts at a word boundary. This is an approximation —
    the exact spoken-word boundary isn't known without ASR, and f5-tts's
    auto-transcription path would download a ~1.6GB Whisper model.
    """
    if kept_fraction >= 1.0:
        return text
    words = text.split()
    keep = max(1, round(len(words) * kept_fraction))
    return " ".join(words[:keep]).rstrip(",;:- ")


class F5TTSBackend:
    """F5-TTS Romanian backend for zero-shot voice cloning."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device: str | None = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        # MPS verified stable on this checkpoint with memory free and ~2x
        # faster than CPU (CPU is ~20x slower than realtime for F5's 32-step
        # flow matching, so every bit helps).
        return get_torch_device(allow_mps=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return F5_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        """Check both the fine-tune checkpoint and the Vocos vocoder cache."""
        return is_model_cached(F5_HF_REPO, required_files=[F5_CKPT_FILE, F5_VOCAB_FILE]) and is_model_cached(
            F5_VOCODER_HF_REPO
        )

    async def load_model(self, model_size: str = "default") -> None:
        """Load the F5-TTS Romanian model."""
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        model_name = "f5-tts-romanian"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            from huggingface_hub import hf_hub_download  # lazy: heavy import

            ckpt_file = hf_hub_download(F5_HF_REPO, F5_CKPT_FILE)
            vocab_file = hf_hub_download(F5_HF_REPO, F5_VOCAB_FILE)

            device = self._get_device()
            self._device = device
            logger.info("Loading F5-TTS Romanian on %s...", device)

            from f5_tts.api import F5TTS  # lazy: heavy import

            # The pruned checkpoint stores plain (non-EMA-prefixed) weights,
            # so use_ema=False is the semantically correct load path.
            self.model = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                use_ema=False,
                device=device,
            )

        logger.info("F5-TTS Romanian loaded successfully")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            empty_device_cache(device)
            logger.info("F5-TTS Romanian unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Like Chatterbox, F5 processes reference audio at generation time,
        so the prompt just stores the file path and transcript. Trimming
        to the 12s reference limit also happens at generation time.
        """
        voice_prompt = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=F5_SAMPLE_RATE)

    def _prepare_reference(self, ref_audio: str, ref_text: str) -> tuple[str, str]:
        """Trim the reference to the F5 limit, returning (audio_path, text).

        References at or under the limit pass through untouched. Longer ones
        are cut at the quietest moment between 8s and 12s, written to the
        cache directory (content-keyed, reused across generations), and the
        transcript is proportionally truncated to keep correspondence.
        """
        import librosa  # lazy: heavy import
        import soundfile as sf  # lazy: heavy import

        audio, sample_rate = librosa.load(ref_audio, sr=None, mono=True)
        trimmed, kept_fraction = trim_reference_audio(audio, sample_rate)
        if kept_fraction >= 1.0:
            return ref_audio, ref_text

        stat = Path(ref_audio).stat()
        cache_key = hashlib.md5(f"{ref_audio}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:16]
        trimmed_path = config.get_cache_dir() / f"f5_ref_{cache_key}.wav"
        if not trimmed_path.exists():
            sf.write(str(trimmed_path), trimmed, sample_rate)

        trimmed_text = trim_reference_text(ref_text, kept_fraction)
        logger.info(
            "[F5] Trimmed reference %.1fs -> %.1fs (kept %.0f%%)",
            len(audio) / sample_rate,
            len(trimmed) / sample_rate,
            kept_fraction * 100,
        )
        return str(trimmed_path), trimmed_text

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "ro",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using F5-TTS Romanian.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with ref_audio path and ref_text transcript
            language: Language code ("ro" or "en")
            seed: Random seed for reproducibility
            instruct: Not supported by F5 (ignored)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Validate the prompt before the expensive model load.
        ref_audio = voice_prompt.get("ref_audio")
        ref_text = voice_prompt.get("ref_text") or ""
        if not ref_audio or not Path(ref_audio).exists():
            raise ValueError(f"F5-TTS requires reference audio for voice cloning (missing: {ref_audio})")
        # An empty ref_text makes f5-tts auto-transcribe with Whisper
        # large-v3-turbo — a surprise ~1.6GB download. Refuse instead.
        if not ref_text.strip():
            raise ValueError("F5-TTS requires the reference transcript (profile sample reference_text)")

        await self.load_model()

        def _generate_sync():
            ref_file, trimmed_text = self._prepare_reference(ref_audio, ref_text)

            if language == "ro":
                gen_text = normalize_romanian_text(text)
                prompt_text = normalize_romanian_text(trimmed_text)
            else:
                gen_text = unicodedata.normalize("NFC", text)
                prompt_text = unicodedata.normalize("NFC", trimmed_text)

            logger.info("[F5] Generating: lang=%s", language)

            # F5TTS.infer seeds torch/numpy/random itself (seed_everything);
            # passing seed=None picks a fresh random seed.
            wav, sample_rate, _spec = self.model.infer(
                ref_file,
                prompt_text,
                gen_text,
                nfe_step=F5_NFE_STEPS,
                seed=seed,
                show_info=logger.debug,
            )

            audio = np.asarray(wav, dtype=np.float32)
            return audio, int(sample_rate)

        return await asyncio.to_thread(_generate_sync)
