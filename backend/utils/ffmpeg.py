"""Locate ffmpeg and transcode WAV exports into compressed containers.

Voicebox does not bundle ffmpeg. Export paths that need it (MP3, M4B) shell out
to whatever the user has installed and raise a clear ``RuntimeError`` when it
is missing so the API can turn that into an actionable 503.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

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
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
}


def find_ffmpeg(name: str = "ffmpeg") -> Optional[str]:
    """Return the path to ``name`` (ffmpeg/ffprobe) or None when not installed."""
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


def encode_audio(
    wav_path: Path,
    out_path: Path,
    fmt: str,
    metadata_path: Optional[Path] = None,
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

    cmd: List[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path)]
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
