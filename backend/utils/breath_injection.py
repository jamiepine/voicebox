"""Synthetic breath injection for humanizing TTS output."""

import numpy as np
from scipy.signal import butter, sosfilt


def _generate_breath(duration_ms: int, sample_rate: int, volume: float) -> np.ndarray:
    """Generate a synthetic breath sound using band-pass filtered noise."""
    n_samples = int(duration_ms * sample_rate / 1000)
    # White noise
    noise = np.random.randn(n_samples).astype(np.float32)
    # Band-pass filter 200Hz-2kHz (breath frequency range)
    sos = butter(4, [200, 2000], btype='bandpass', fs=sample_rate, output='sos')
    filtered = sosfilt(sos, noise).astype(np.float32)
    # Gaussian amplitude envelope (natural breath shape)
    t = np.linspace(-3, 3, n_samples)
    envelope = np.exp(-0.5 * t**2).astype(np.float32)
    return filtered * envelope * volume


def _find_silence_gaps(
    audio: np.ndarray,
    sample_rate: int,
    min_silence_ms: int = 200,
    silence_threshold_db: float = -40.0,
    frame_ms: int = 20,
) -> list[tuple[int, int]]:
    """Find silence gaps in audio, returning list of (start_sample, end_sample)."""
    frame_size = int(frame_ms * sample_rate / 1000)
    threshold = 10 ** (silence_threshold_db / 20)

    gaps = []
    gap_start = None

    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        rms = np.sqrt(np.mean(frame**2))

        if rms < threshold:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_end = i
                gap_duration_ms = (gap_end - gap_start) * 1000 / sample_rate
                if gap_duration_ms >= min_silence_ms:
                    gaps.append((gap_start, gap_end))
                gap_start = None

    return gaps


def inject_breaths(
    audio: np.ndarray,
    sample_rate: int,
    min_silence_ms: int = 200,
    breath_duration_ms: int = 150,
    breath_volume: float = 0.03,
    silence_threshold_db: float = -40.0,
) -> np.ndarray:
    """
    Inject synthetic breath sounds at silence gaps in audio.

    Detects silence gaps longer than min_silence_ms and overlays
    a subtle synthetic breath sound in the middle of each gap.
    Skips the first and last gaps to avoid leading/trailing artifacts.
    """
    gaps = _find_silence_gaps(audio, sample_rate, min_silence_ms, silence_threshold_db)

    if len(gaps) <= 2:
        return audio  # Not enough gaps (need at least one internal gap)

    result = audio.copy()
    # Skip first and last gaps
    internal_gaps = gaps[1:-1]

    rng = np.random.default_rng()

    for gap_start, gap_end in internal_gaps:
        # Random variation in breath params
        dur = breath_duration_ms + rng.integers(-30, 31)
        vol = breath_volume * (1.0 + rng.uniform(-0.3, 0.3))

        breath = _generate_breath(max(dur, 50), sample_rate, vol)

        # Center breath in the gap
        gap_center = (gap_start + gap_end) // 2
        breath_start = gap_center - len(breath) // 2
        breath_end = breath_start + len(breath)

        # Bounds check
        if breath_start < 0 or breath_end > len(result):
            continue

        # Overlay (add, don't replace)
        result[breath_start:breath_end] += breath

    return result
