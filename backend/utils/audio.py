"""
Audio processing utilities.

Includes EBU R128 loudness normalization with true-peak limiting,
matching broadcast standards. Uses pyloudnorm for LUFS measurement
and normalization — pure Python, no ffmpeg dependency.
"""

import logging
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def normalize_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    target_lufs: float = -16.0,
    true_peak_limit_db: float = -2.0,
) -> np.ndarray:
    """
    Normalize audio to target loudness (EBU R128) with true-peak limiting.

    Matches the behavior of ffmpeg's loudnorm filter:
        loudnorm=I=-16:TP=-2:LRA=11,alimiter=limit=-2dB

    Falls back to simple RMS normalization if pyloudnorm is unavailable
    or audio is too short for LUFS measurement.

    Args:
        audio: Input audio array (mono, float32)
        sample_rate: Audio sample rate
        target_lufs: Target integrated loudness in LUFS (default: -16)
        true_peak_limit_db: True-peak ceiling in dBTP (default: -2)

    Returns:
        Normalized audio array
    """
    import warnings
    audio = audio.astype(np.float32)

    if len(audio) == 0:
        return audio

    # True-peak limit as linear amplitude
    peak_limit = 10 ** (true_peak_limit_db / 20)

    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)

        # pyloudnorm requires at least 0.4s of audio for LUFS measurement
        min_samples = int(sample_rate * 0.4)
        if len(audio) < min_samples:
            logger.debug("Audio too short for LUFS normalization, using RMS fallback")
            return _normalize_rms(audio, target_db=target_lufs, peak_limit=peak_limit)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current_lufs = meter.integrated_loudness(audio)

        # If audio is essentially silent, LUFS returns -inf
        if not np.isfinite(current_lufs) or current_lufs < -70:
            logger.debug("Audio too quiet for LUFS normalization (%.1f LUFS)", current_lufs)
            return audio

        # Apply loudness normalization (suppress clipping warnings — we clip intentionally below)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Possible clipping.*")
            audio = pyln.normalize.loudness(audio, current_lufs, target_lufs)

        # True-peak limiting via clipping (simple but effective for TTS output)
        audio = np.clip(audio, -peak_limit, peak_limit)

        return audio

    except ImportError:
        logger.warning("pyloudnorm not installed, falling back to RMS normalization")
        return _normalize_rms(audio, target_db=target_lufs, peak_limit=peak_limit)


def _normalize_rms(
    audio: np.ndarray,
    target_db: float = -20.0,
    peak_limit: float = 0.85,
) -> np.ndarray:
    """
    Simple RMS-based normalization fallback.

    Used when pyloudnorm is unavailable or audio is too short for LUFS.
    """
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    target_rms = 10 ** (target_db / 20)

    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain

    audio = np.clip(audio, -peak_limit, peak_limit)
    return audio


def load_audio(
    path: str,
    sample_rate: int = 24000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with normalization.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return audio, sr


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int = 24000,
    normalize: bool = True,
) -> None:
    """
    Save audio file, optionally with loudness normalization.

    When normalize=True (the default), applies EBU R128 loudness
    normalization to -16 LUFS with -2 dBTP true-peak limiting before
    saving. This ensures consistent output volume across generations.

    Args:
        audio: Audio array
        path: Output path
        sample_rate: Sample rate
        normalize: Apply loudness normalization before saving (default: True)
    """
    if normalize:
        audio = normalize_audio(audio, sample_rate=sample_rate)
    sf.write(path, audio, sample_rate)


def validate_and_normalize_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    trim_threshold: float = 45.0,
    min_rms: float = 0.01,
) -> Tuple[bool, Optional[str]]:
    """
    Validate, auto-trim, and normalize reference audio for voice cloning.

    Does the heavy lifting so users don't have to manually prepare audio:
    - Clips slightly over max_duration are auto-trimmed (up to trim_threshold)
    - Audio is loudness-normalized to broadcast standards (EBU R128)
    - The old peak > 0.99 "clipping" rejection is gone entirely

    Args:
        audio_path: Path to audio file (will be overwritten with processed version)
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds (inclusive — 30.0s is allowed)
        trim_threshold: Auto-trim clips up to this length to max_duration.
            Clips longer than this are rejected outright.
        min_rms: Minimum RMS level (below this = silence)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        audio, sr = load_audio(audio_path)
        duration = len(audio) / sr

        if duration < min_duration:
            return False, f"Audio too short ({duration:.1f}s, minimum {min_duration}s)"

        # Auto-trim clips that are over max_duration but within trim_threshold
        if duration > max_duration:
            if duration <= trim_threshold:
                # Trim to max_duration — take the first N seconds
                max_samples = int(max_duration * sr)
                audio = audio[:max_samples]
                logger.info(
                    "Auto-trimmed reference audio from %.1fs to %.1fs",
                    duration, max_duration,
                )
                duration = max_duration
            else:
                return False, (
                    f"Audio too long ({duration:.1f}s, maximum {max_duration}s). "
                    f"Clips up to {trim_threshold:.0f}s are auto-trimmed."
                )

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < min_rms:
            return False, "Audio is too quiet or silent"

        # Normalize to consistent loudness (EBU R128, -16 LUFS)
        audio = normalize_audio(audio, sample_rate=sr)
        sf.write(audio_path, audio, sr)

        return True, None
    except Exception as e:
        return False, f"Error processing audio: {str(e)}"


# Keep old name as alias for backward compatibility
validate_reference_audio = validate_and_normalize_reference_audio
