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
    Save audio file.
    
    Args:
        audio: Audio array
        path: Output path
        sample_rate: Sample rate
    """
    sf.write(path, audio, sample_rate)


def prepare_for_transcription(
    audio: np.ndarray,
    sr: int = 16000,  # noqa: ARG001 — kept for API consistency
) -> np.ndarray:
    """
    Prepare audio for Whisper transcription.

    Trims leading/trailing silence and normalizes volume.
    Does NOT apply denoising (Whisper handles noise natively).

    Args:
        audio: Input audio array (16kHz mono expected)
        sr: Sample rate

    Returns:
        Preprocessed audio array
    """
    # Trim leading/trailing silence using librosa
    trimmed, _ = librosa.effects.trim(audio, top_db=30)

    # Peak normalization (scale to [-1, 1])
    peak = np.max(np.abs(trimmed))
    if peak > 0:
        trimmed = trimmed / peak

    return trimmed


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
    Trim TTS-generated audio: removes trailing silence/noise AND cuts at
    long internal silence gaps (Chatterbox hallucination boundary).

    The Chatterbox T3 token generator can overshoot: after speech ends it
    may emit silence tokens followed by garbage/hallucinated audio.  This
    function detects the first long silence gap after speech starts and
    cuts everything after it.

    Args:
        audio: Input audio array
        sample_rate: Sample rate in Hz
        frame_ms: Frame size in milliseconds for energy calculation
        silence_threshold_db: Absolute silence threshold in dB
        min_silence_ms: Minimum trailing silence before trimming
        max_internal_silence_ms: Cut after a silence gap longer than this
        fade_ms: Fade-out duration in milliseconds

    Returns:
        Trimmed audio with fade-out applied
    """
    if len(audio) == 0:
        return audio

    frame_size = int(sample_rate * frame_ms / 1000)
    min_silence_frames = int(min_silence_ms / frame_ms)
    max_internal_silence_frames = int(max_internal_silence_ms / frame_ms)
    threshold = 10 ** (silence_threshold_db / 20)
    n_frames = len(audio) // frame_size

    if n_frames == 0:
        return audio

    # Calculate RMS per frame
    rms_per_frame = np.array([
        np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ])

    is_speech = rms_per_frame > threshold

    # Find first speech frame (trim leading silence)
    first_speech_frame = 0
    for i in range(n_frames):
        if is_speech[i]:
            first_speech_frame = max(0, i - 1)  # Keep 1 frame before speech
            break

    # Scan forward from first speech: cut at any silence gap > max_internal_silence_ms
    # This catches the pattern: [speech] [long silence] [garbage/hallucination]
    speech_started = False
    consecutive_silence = 0
    cut_frame = n_frames  # Default: keep everything

    for i in range(first_speech_frame, n_frames):
        if is_speech[i]:
            speech_started = True
            consecutive_silence = 0
        else:
            if speech_started:
                consecutive_silence += 1
                if consecutive_silence >= max_internal_silence_frames:
                    # Cut at the start of this silence gap
                    cut_frame = i - consecutive_silence + 1
                    break

    # Also trim trailing silence from the cut point
    last_speech_frame = cut_frame
    for i in range(cut_frame - 1, first_speech_frame - 1, -1):
        if is_speech[i]:
            last_speech_frame = i + 1
            break

    # Only trim trailing if enough silence
    trailing_silence = cut_frame - last_speech_frame
    if trailing_silence >= min_silence_frames:
        end_frame = last_speech_frame + 1
    else:
        end_frame = cut_frame

    start_sample = first_speech_frame * frame_size
    end_sample = min(end_frame * frame_size, len(audio))
    trimmed = audio[start_sample:end_sample].copy()

    # Apply cosine fade-out to prevent clicks
    fade_samples = min(int(fade_ms * sample_rate / 1000), len(trimmed) // 4)
    if fade_samples > 0:
        fade = (np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2).astype(np.float32)
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
    try:
        audio, sr = load_audio(audio_path)
        duration = len(audio) / sr
        
        if duration < min_duration:
            return False, f"Audio too short (minimum {min_duration} seconds)"
        if duration > max_duration:
            return False, f"Audio too long (maximum {max_duration} seconds)"
        
        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, "Audio is too quiet or silent"
        
        if np.abs(audio).max() > 0.99:
            return False, "Audio is clipping (reduce input gain)"
        
        return True, None
    except Exception as e:
        return False, f"Error validating audio: {str(e)}"
