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


def trim_leading_artifact(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 10,
    burst_threshold_db: float = -38.0,
    dip_threshold_db: float = -58.0,
    min_dip_ms: int = 30,
    max_burst_ms: int = 800,
    lookahead_ms: int = 1200,
    speech_after_db: float = -45.0,
    decay_diff_db: float = 6.0,
    margin_ms: int = 20,
    fade_ms: int = 10,
) -> np.ndarray:
    """
    Remove the leading "beat" artifact some TTS engines emit at the start of
    each chunk (e.g. Qwen3-TTS: a short breath/noise burst that decays into
    silence before the real speech begins).

    Signature it looks for:  [loud burst] -> [monotonic decay] -> [deep
    silence dip] -> [real speech].  Only trims when ALL conditions hold, so
    audio that starts with real speech is left untouched.

    Args:
        audio: Input audio array (mono float32)
        sample_rate: Sample rate in Hz
        frame_ms: Frame size for RMS energy calculation
        burst_threshold_db: First frame above this dB counts as burst start
        dip_threshold_db: Frames below this dB count as the silence dip
        min_dip_ms: Minimum dip duration to qualify
        max_burst_ms: Dip must appear within this many ms of burst start
        lookahead_ms: After the dip, real speech must appear within this window
        speech_after_db: "Real speech" = frames above this dB in lookahead
        decay_diff_db: Mean RMS of burst's first half must be this many dB
                       louder than the second half (decay signature)
        margin_ms: Extra silence kept after the dip before the trim point
        fade_ms: Cosine fade-in applied at the new start

    Returns:
        Trimmed audio array (unchanged if no artifact is detected)
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len * 4:
        return audio

    n_frames = len(audio) // frame_len
    rms = np.array(
        [
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ]
    )
    db = 20.0 * np.log10(rms + 1e-12)

    burst_threshold = burst_threshold_db
    dip_threshold = dip_threshold_db
    min_dip_frames = max(1, int(min_dip_ms / frame_ms))
    max_burst_frames = int(max_burst_ms / frame_ms)
    lookahead_frames = int(lookahead_ms / frame_ms)

    # 1) Find first "loud" frame (burst start)
    loud_idx = np.where(db > burst_threshold)[0]
    if len(loud_idx) == 0:
        return audio
    burst_start = int(loud_idx[0])

    # 2) Find the first deep-silence dip after the burst (within max_burst_ms)
    dip_start = -1
    dip_len = 0
    i = burst_start + 1
    limit = min(n_frames, burst_start + max_burst_frames + min_dip_frames)
    while i < limit:
        if db[i] < dip_threshold:
            j = i
            while j < limit and db[j] < dip_threshold:
                j += 1
            if j - i >= min_dip_frames:
                dip_start = i
                dip_len = j - i
                break
            i = j
        else:
            i += 1

    if dip_start == -1:
        return audio

    # 3) Decay signature: first half of the burst must be clearly louder
    #    than the second half (the artifact decays; real speech does not).
    seg = db[burst_start:dip_start]
    if len(seg) >= 4:
        half = len(seg) // 2
        first_half = seg[:half]
        second_half = seg[half : half * 2]
        if float(np.mean(first_half) - np.mean(second_half)) < decay_diff_db:
            return audio

    # 4) Real speech must resume after the dip
    after_start = dip_start + dip_len
    window_end = min(n_frames, after_start + lookahead_frames)
    if window_end <= after_start:
        return audio
    if float(np.max(db[after_start:window_end])) < speech_after_db:
        return audio

    # 5) Trim to just past the dip (+ small margin into the silence)
    margin_frames = int(margin_ms / frame_ms)
    trim_frame = after_start + margin_frames
    start_sample = min(trim_frame * frame_len, len(audio) - 1)

    trimmed = audio[start_sample:].copy()

    # Short cosine fade-in to avoid any click at the new start
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and len(trimmed) > fade_samples:
        fade = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        trimmed[:fade_samples] *= fade

    return trimmed


def build_trim_fn(engine: str):
    """Build the per-chunk trim pipeline for an engine.

    Always removes the leading burst artifact (pattern-detected, safe for
    all engines); additionally applies ``trim_tts_output`` for engines that
    hallucinate trailing noise (e.g. Chatterbox).
    """
    # Local import to avoid any import cycle with backends.
    from ..backends import engine_needs_trim

    def _trim(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = trim_leading_artifact(audio, sample_rate)
        if engine_needs_trim(engine):
            audio = trim_tts_output(audio, sample_rate)
        return audio

    return _trim


def preprocess_reference_audio(
    audio: np.ndarray,
    sample_rate: int,
    peak_target: float = 0.95,
    trim_top_db: float = 40.0,
    edge_padding_ms: int = 100,
) -> np.ndarray:
    """
    Clean up a reference-audio sample before validation/storage.

    Removes DC offset, trims leading/trailing silence, and caps the peak so a
    slightly-hot recording doesn't get rejected downstream as "clipping". The
    goal is to accept reasonable real-world recordings — not to repair badly
    distorted ones. True clipping artifacts inside the waveform can't be
    recovered by peak scaling and will still sound bad.

    Args:
        audio: Mono audio array.
        sample_rate: Sample rate of ``audio`` in Hz.
        peak_target: Peak amplitude cap in [0, 1]. Applied only if the input
            peak exceeds this value.
        trim_top_db: Silence threshold for edge trimming, in dB below peak.
            40 dB sits below normal speech dynamic range (≈30 dB) so soft
            trailing syllables are preserved, while still catching obvious
            leading/trailing silence. Lower values are more aggressive;
            librosa's own default is 60.
        edge_padding_ms: Milliseconds of padding to add back at each edge
            *only if* trimming shortened the waveform, so TTS engines have a
            brief silence to anchor on without ever making the output longer
            than the input.

    Returns:
        Preprocessed audio array (float32).
    """
    audio = audio.astype(np.float32, copy=False)

    if audio.size == 0:
        return audio

    audio = audio - float(np.mean(audio))

    trimmed, _ = librosa.effects.trim(audio, top_db=trim_top_db)
    if 0 < trimmed.size < audio.size:
        pad_each = int(sample_rate * edge_padding_ms / 1000)
        # Never pad past the original length — for near-max-duration uploads
        # an unconditional pad would push them over the 30 s ceiling and
        # trigger a spurious "too long" rejection.
        headroom = (audio.size - trimmed.size) // 2
        pad = min(pad_each, max(headroom, 0))
        if pad > 0:
            trimmed = np.pad(trimmed, (pad, pad), mode="constant")
        audio = trimmed

    peak = float(np.abs(audio).max())
    if peak > peak_target and peak > 0:
        audio = audio * (peak_target / peak)

    return audio


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

    Applies :func:`preprocess_reference_audio` before checks so that
    slightly-hot recordings aren't rejected as clipping. Duration and RMS
    checks run on the preprocessed waveform.

    Returns:
        Tuple of (is_valid, error_message, audio_array, sample_rate)
    """
    try:
        audio, sr = load_audio(audio_path)
        audio = preprocess_reference_audio(audio, sr)
        duration = len(audio) / sr

        if duration < min_duration:
            return False, f"Audio too short (minimum {min_duration} seconds)", None, None
        if duration > max_duration:
            return False, f"Audio too long (maximum {max_duration} seconds)", None, None

        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, "Audio is too quiet or silent", None, None

        return True, None, audio, sr
    except Exception as e:
        return False, f"Error validating audio: {str(e)}", None, None
