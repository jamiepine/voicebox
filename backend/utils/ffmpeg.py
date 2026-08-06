"""Optional ffmpeg integration.

Voicebox does not bundle ffmpeg and must not require it: the mixdown, the
export formats, the time-stretch and the ducking all have working pure-Python
paths. ffmpeg is used only where it is genuinely better, and every call site
falls back when it is absent.

Where it wins:
  - ``loudnorm`` — EBU R128 loudness normalisation. Clips generated from
    different voices land at noticeably different levels, and peak
    normalisation (the fallback) does nothing about that.

Where it is already load-bearing, whether we like it or not:
  - Decoding ``.m4a`` / ``.aac`` / ``.webm``. libsndfile handles none of them,
    so librosa falls through to audioread, which shells out to ffmpeg. Those
    extensions are advertised by the import endpoint, so without ffmpeg they
    fail deep in the decoder with an opaque message. :func:`requires_ffmpeg`
    lets callers reject them up front instead.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Containers libsndfile cannot open, so librosa must fall back to
# audioread -> ffmpeg. Keep in sync with IMPORT_AUDIO_EXTENSIONS.
FFMPEG_ONLY_EXTENSIONS = {".m4a", ".aac", ".webm"}

# Resolved once — a PATH lookup per audio operation is wasteful, and the
# answer cannot change within a process run.
_cached_path: str | None = None
_probed = False


def ffmpeg_path() -> str | None:
    """Absolute path to ffmpeg, or None when it isn't installed."""
    global _cached_path, _probed
    if not _probed:
        _cached_path = shutil.which("ffmpeg")
        _probed = True
        logger.info("ffmpeg %s", f"found at {_cached_path}" if _cached_path else "not found on PATH")
    return _cached_path


def is_available() -> bool:
    """Whether the optional ffmpeg paths can be used."""
    return ffmpeg_path() is not None


def reset_cache() -> None:
    """Forget the cached lookup. Used by tests to exercise the fallback path."""
    global _cached_path, _probed
    _cached_path = None
    _probed = False


def requires_ffmpeg(suffix: str) -> bool:
    """Whether decoding ``suffix`` needs ffmpeg that we may not have."""
    return suffix.lower() in FFMPEG_ONLY_EXTENSIONS


def normalize_loudness(
    audio_bytes: bytes,
    suffix: str = ".wav",
    target_lufs: float = -16.0,
    true_peak: float = -1.5,
) -> bytes | None:
    """Loudness-normalise an encoded file to ``target_lufs`` (EBU R128).

    -16 LUFS is the usual target for spoken-word podcasts; -1.5 dBTP leaves
    headroom for lossy codecs, which can overshoot on decode.

    Returns:
        Normalised file bytes, or ``None`` if ffmpeg is unavailable or fails —
        callers keep their existing output in that case.
    """
    exe = ffmpeg_path()
    if exe is None:
        return None

    with tempfile.TemporaryDirectory(prefix="voicebox-loudnorm-") as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / f"out{suffix}"
        src.write_bytes(audio_bytes)

        cmd = [
            exe,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-i", str(src),
            "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11",
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("ffmpeg loudnorm failed, keeping un-normalised audio: %s", exc)
            return None

        if not dst.exists() or dst.stat().st_size == 0:
            return None
        return dst.read_bytes()
