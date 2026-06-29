"""Unit tests for story-export helpers — chapter derivation + FFMETADATA1 emission.

These tests cover the pure-Python pieces of the m4b/mp3 export path that don't
need a database session. The ffmpeg-driven encode path is exercised at the end
under a shutil.which gate so the suite stays green on CI runners without ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from backend.services.stories import (
    _Chapter,
    _chapter_title_from_text,
    _derive_chapters_auto,
    _escape_ffmetadata,
    _ffmpeg_encode,
    _write_ffmetadata,
)


class TestChapterTitleFromText:
    def test_returns_first_sentence(self):
        assert _chapter_title_from_text("Hello world. Second sentence.") == "Hello world."

    def test_truncates_long_sentence(self):
        long = "a" * 200
        title = _chapter_title_from_text(long, max_len=80)
        assert title.endswith("…")
        assert len(title) <= 81  # 80 chars + ellipsis

    def test_handles_cjk_punctuation(self):
        assert _chapter_title_from_text("第一章。第二章。") == "第一章。"

    def test_empty_and_whitespace_fall_back(self):
        assert _chapter_title_from_text("") == "Chapter"
        assert _chapter_title_from_text("   ") == "Chapter"
        assert _chapter_title_from_text(None) == "Chapter"


class TestDeriveChaptersAuto:
    def test_orders_by_start_time(self):
        segs = [
            {"start_time_ms": 5000, "text": "Two."},
            {"start_time_ms": 0, "text": "One."},
        ]
        chapters = _derive_chapters_auto(segs, total_duration_ms=10000)
        assert [c.start_ms for c in chapters] == [0, 5000]
        assert chapters[0].title == "One."
        assert chapters[1].end_ms == 10000

    def test_fills_end_from_next_chapter(self):
        segs = [
            {"start_time_ms": 0, "text": "A."},
            {"start_time_ms": 3000, "text": "B."},
            {"start_time_ms": 7000, "text": "C."},
        ]
        chapters = _derive_chapters_auto(segs, total_duration_ms=10000)
        assert chapters[0].end_ms == 3000
        assert chapters[1].end_ms == 7000
        assert chapters[2].end_ms == 10000

    def test_dedupes_same_start_time(self):
        # Two items at the same timecode (multi-track) → one chapter.
        segs = [
            {"start_time_ms": 0, "text": "Narrator."},
            {"start_time_ms": 0, "text": "Music bed."},
            {"start_time_ms": 4000, "text": "Next beat."},
        ]
        chapters = _derive_chapters_auto(segs, total_duration_ms=8000)
        assert [c.start_ms for c in chapters] == [0, 4000]

    def test_drops_zero_duration_trailing_chapter(self):
        # An item starting at exactly the total duration would otherwise
        # produce an end_ms == start_ms chapter, which ffmpeg rejects.
        segs = [
            {"start_time_ms": 0, "text": "Body."},
            {"start_time_ms": 10000, "text": "Tail."},
        ]
        chapters = _derive_chapters_auto(segs, total_duration_ms=10000)
        assert len(chapters) == 1


class TestFFMetadataEscaping:
    @pytest.mark.parametrize(
        "raw,escaped",
        [
            ("simple", "simple"),
            ("a=b", "a\\=b"),
            ("a;b", "a\\;b"),
            ("#hash", "\\#hash"),
            ("back\\slash", "back\\\\slash"),
            ("line1\nline2", "line1\\nline2"),
        ],
    )
    def test_escapes_all_specials(self, raw, escaped):
        assert _escape_ffmetadata(raw) == escaped


class TestWriteFFMetadata:
    def test_emits_valid_chapter_block(self, tmp_path: Path):
        chapters = [
            _Chapter(start_ms=0, end_ms=3000, title="Intro"),
            _Chapter(start_ms=3000, end_ms=10000, title="Body; with = chars"),
        ]
        out = tmp_path / "meta.txt"
        _write_ffmetadata(chapters, out)
        content = out.read_text(encoding="utf-8")
        assert content.startswith(";FFMETADATA1")
        assert content.count("[CHAPTER]") == 2
        assert "START=0\nEND=3000" in content
        assert "title=Intro" in content
        # Special chars must be escaped in the metadata value.
        assert "Body\\; with \\= chars" in content


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestFFmpegEncodeIntegration:
    """End-to-end: synthesize a WAV, encode to M4B with chapters, read back with ffprobe."""

    def _make_silent_wav(self, path: Path, seconds: float = 12.0, sample_rate: int = 24000) -> None:
        import numpy as np  # imported lazily — the rest of the suite shouldn't depend on numpy

        from backend.utils.audio import save_audio

        silence = np.zeros(int(seconds * sample_rate), dtype=np.float32)
        save_audio(silence, str(path), sample_rate)

    def test_m4b_round_trip_carries_chapters(self, tmp_path: Path):
        if shutil.which("ffprobe") is None:
            pytest.skip("ffprobe not installed")

        wav_path = tmp_path / "in.wav"
        out_path = tmp_path / "out.m4b"
        self._make_silent_wav(wav_path)

        chapters = [
            _Chapter(start_ms=0, end_ms=4000, title="Opening"),
            _Chapter(start_ms=4000, end_ms=12000, title="Closing"),
        ]
        _ffmpeg_encode(wav_path, out_path, fmt="m4b", chapters=chapters)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_chapters",
                "-of",
                "default=noprint_wrappers=1",
                str(out_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        chunks = probe.stdout.split("[CHAPTER]")
        # ffprobe prints one [CHAPTER] block per chapter; expect two.
        assert sum(1 for c in chunks if c.strip()) == 2
        assert "Opening" in probe.stdout
        assert "Closing" in probe.stdout
