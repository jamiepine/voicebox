"""
Audio processing utilities.
"""

import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
    peak_limit: float = 0.85,
) -> np.ndarray:
    """
    Normalize audio to target loudness with peak limiting.
    
    Args:
        audio: Input audio array
        target_db: Target RMS level in dB
        peak_limit: Peak limit (0.0-1.0)
        
    Returns:
        Normalized audio array
    """
    # Convert to float32
    audio = audio.astype(np.float32)
    
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    
    # Calculate target RMS
    target_rms = 10**(target_db / 20)
    
    # Apply gain
    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain
    
    # Peak limiting
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
) -> None:
    """
    Save audio file with atomic write and error handling.

    Writes to a temporary file first, then atomically renames to the
    target path.  This prevents corrupted/partial WAV files if the
    process is interrupted mid-write.

    Args:
        audio: Audio array
        path: Output path
        sample_rate: Sample rate

    Raises:
        OSError: If file cannot be written
    """
    from pathlib import Path
    import os

    temp_path = f"{path}.tmp"
    try:
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first (explicit format since .tmp
        # extension is not recognised by soundfile)
        sf.write(temp_path, audio, sample_rate, format='WAV')

        # Atomic rename to final path
        os.replace(temp_path, path)

    except Exception as e:
        # Clean up temp file on failure
        try:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception:
            pass  # Best effort cleanup

        raise OSError(f"Failed to save audio to {path}: {e}") from e


def trim_tts_output(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    min_silence_ms: int = 200,
    max_internal_silence_ms: int = 1000,
    fade_ms: int = 30,
) -> np.ndarray:
    """
    Trim trailing silence and post-silence hallucination from TTS output.

    Chatterbox sometimes produces ``[speech][silence][hallucinated noise]``.
    This detects internal silence gaps longer than *max_internal_silence_ms*
    and cuts the audio at that boundary, then trims trailing silence and
    applies a short cosine fade-out.

    Args:
        audio: Input audio array (mono float32)
        sample_rate: Sample rate in Hz
        frame_ms: Frame size for RMS energy calculation
        silence_threshold_db: dB threshold below which a frame is silence
        min_silence_ms: Minimum trailing silence to keep
        max_internal_silence_ms: Cut after any silence gap longer than this
        fade_ms: Cosine fade-out duration in ms

    Returns:
        Trimmed audio array
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return audio

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)

    # Compute per-frame RMS
    rms = np.array(
        [
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ]
    )
    is_speech = rms >= threshold_linear

    # Find first speech frame
    first_speech = 0
    for i, s in enumerate(is_speech):
        if s:
            first_speech = max(0, i - 1)  # keep 1 frame padding
            break

    # Walk forward from first speech; cut at long internal silence gaps
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    consecutive_silence = 0
    cut_frame = n_frames

    for i in range(first_speech, n_frames):
        if is_speech[i]:
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            if consecutive_silence >= max_silence_frames:
                cut_frame = i - consecutive_silence + 1
                break

    # Trim trailing silence from the cut point
    min_silence_frames = int(min_silence_ms / frame_ms)
    end_frame = cut_frame
    while end_frame > first_speech and not is_speech[end_frame - 1]:
        end_frame -= 1
    # Keep a short tail
    end_frame = min(end_frame + min_silence_frames, cut_frame)

    # Convert frames back to samples
    start_sample = first_speech * frame_len
    end_sample = min(end_frame * frame_len, len(audio))

    trimmed = audio[start_sample:end_sample].copy()

    # Cosine fade-out
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and len(trimmed) > fade_samples:
        fade = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        trimmed[-fade_samples:] *= fade

    return trimmed


def validate_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    min_rms: float = 0.01,
) -> Tuple[bool, Optional[str]]:
    """
    Validate reference audio for voice cloning.
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        min_rms: Minimum RMS level
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    result = validate_and_load_reference_audio(
        audio_path, min_duration, max_duration, min_rms
    )
    return (result[0], result[1])


def validate_and_load_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    min_rms: float = 0.01,
) -> Tuple[bool, Optional[str], Optional[np.ndarray], Optional[int]]:
    """
    Validate and load reference audio in a single pass.
    
    Returns:
        Tuple of (is_valid, error_message, audio_array, sample_rate)
    """
    try:
        audio, sr = load_audio(audio_path)
        duration = len(audio) / sr
        
        if duration < min_duration:
            return False, f"Audio too short (minimum {min_duration} seconds)", None, None
        if duration > max_duration:
            return False, f"Audio too long (maximum {max_duration} seconds)", None, None
        
        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, "Audio is too quiet or silent", None, None
        
        if np.abs(audio).max() > 0.99:
            return False, "Audio is clipping (reduce input gain)", None, None
        
        return True, None, audio, sr
    except Exception as e:
        return False, f"Error validating audio: {str(e)}", None, None
