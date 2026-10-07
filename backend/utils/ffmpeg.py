"""Optional ffmpeg integration.

Voicebox does not bundle ffmpeg and must not require it for the core paths:
the mixdown, WAV/FLAC/OGG/Opus export, the time-stretch and the ducking all
have pure-Python (libsndfile) implementations. ffmpeg is used where it is the
only option or genuinely better, and every call site either falls back or
raises a clear, actionable error.

Where it is required:
  - MP3 and M4B export (:func:`encode_audio`). Those paths shell out to
    whatever the user has installed and raise ``RuntimeError`` with an install
    hint when it is missing, so the API can turn that into a 503.

Where it is optional and falls back:
  - ``loudnorm`` — EBU R128 loudness normalisation
    (:func:`normalize_loudness`). Clips generated from different voices land
    at noticeably different levels, and peak normalisation (the fallback) does
    nothing about that.

Where it is already load-bearing, whether we like it or not:
  - Decoding ``.m4a`` / ``.aac`` / ``.webm``. libsndfile handles none of them,
    so librosa falls through to audioread, which shells out to ffmpeg. Those
    extensions are advertised by the import endpoint, so without ffmpeg they
    fail deep in the decoder with an opaque message. :func:`requires_ffmpeg`
    lets callers reject them up front instead.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# GUI-launched apps on macOS get a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin),
# so a Homebrew or MacPorts ffmpeg is invisible to shutil.which. Windows package
# managers likewise drop shims in per-user dirs the sidecar may not see.
_EXTRA_DIRS = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/local/bin",
]
if sys.platform == "win32":
    _local = os.environ.get("LOCALAPPDATA")
    if _local:
        _EXTRA_DIRS.append(str(Path(_local) / "Microsoft" / "WinGet" / "Links"))
    _EXTRA_DIRS.append(r"C:\ffmpeg\bin")

MISSING_MESSAGE = (
    "ffmpeg is required for MP3/M4B export but was not found. Install it "
    "(macOS: brew install ffmpeg; Windows: winget install Gyan.FFmpeg; "
    "Linux: apt install ffmpeg) and restart Voicebox, or export as WAV."
)

ENCODERS = {
    # The 'ipod' muxer is ffmpeg's name for the .m4b container.
    "m4b": ["-c:a", "aac", "-b:a", "128k", "-f", "ipod"],
    # 160k is the MPEG-2 LSF ceiling, so LAME keeps the 24 kHz source rate
    # instead of resampling to 44.1 kHz to satisfy a higher bitrate.
    "mp3": ["-c:a", "libmp3lame", "-b:a", "160k", "-f", "mp3"],
}

# Containers libsndfile cannot open, so librosa must fall back to
# audioread -> ffmpeg. Keep in sync with IMPORT_AUDIO_EXTENSIONS.
FFMPEG_ONLY_EXTENSIONS = {".m4a", ".aac", ".webm"}


def find_ffmpeg(name: str = "ffmpeg") -> str | None:
    """Return the path to ``name`` (ffmpeg/ffprobe) or None when not installed.

    Looks at ``VOICEBOX_FFMPEG_DIR`` first, then PATH, then the package-manager
    directories a GUI-launched process tends not to see. Uncached, so a test or
    a long-running server sees the current state; :func:`ffmpeg_path` is the
    cached variant for hot paths.
    """
    override = os.environ.get("VOICEBOX_FFMPEG_DIR")
    if override:
        candidate = Path(override) / (f"{name}.exe" if sys.platform == "win32" else name)
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    for d in _EXTRA_DIRS:
        found = shutil.which(name, path=d)
        if found:
            return found
    return None


# Resolved once — a PATH lookup per audio operation is wasteful, and the
# answer cannot change within a process run.
_cached_path: str | None = None
_probed = False


def ffmpeg_path() -> str | None:
    """Absolute path to ffmpeg, or None when it isn't installed.

    Cached result of :func:`find_ffmpeg`; :func:`reset_cache` forgets it.
    """
    global _cached_path, _probed
    if not _probed:
        _cached_path = find_ffmpeg()
        _probed = True
        logger.info("ffmpeg %s", f"found at {_cached_path}" if _cached_path else "not found")
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


def encode_audio(
    wav_path: Path,
    out_path: Path,
    fmt: str,
    metadata_path: Path | None = None,
    timeout: int = 600,
) -> None:
    """Transcode ``wav_path`` to ``fmt`` ("mp3" or "m4b") at ``out_path``.

    ``metadata_path`` is an optional FFMETADATA1 file whose tags and chapters
    are copied into the output. Raises ``ValueError`` for an unknown ``fmt``
    and ``RuntimeError`` when ffmpeg is missing, times out, or exits non-zero.
    """
    if fmt not in ENCODERS:
        raise ValueError(f"Unsupported export format: {fmt}")

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(MISSING_MESSAGE)

    cmd: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path)]
    if metadata_path is not None:
        cmd.extend(["-i", str(metadata_path), "-map_metadata", "1", "-map_chapters", "1"])
    cmd.extend(["-map", "0:a", *ENCODERS[fmt], str(out_path)])

    try:
        # Hard ceiling so a stuck ffmpeg can't pin server resources forever;
        # ten minutes covers a multi-hour audiobook on slow hardware.
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out during audio export") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited {result.returncode}: {stderr}")


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
