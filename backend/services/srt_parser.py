"""Utilities for parsing SRT files into timed dubbing segments."""

from __future__ import annotations

import re
from dataclasses import dataclass


TIMECODE_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})$")


@dataclass
class ParsedSrtSegment:
    """Normalized representation of a single SRT block."""

    srt_index: int
    start_tc: str
    end_tc: str
    start_ms: int
    end_ms: int
    target_duration_ms: int
    text_lines: list[str]
    text: str


def parse_timecode_to_ms(value: str) -> int:
    """Convert ``hh:mm:ss,ms`` into milliseconds."""
    match = TIMECODE_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timecode: {value}")
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    milliseconds = int(match.group("ms"))
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def parse_srt_text(content: str) -> list[ParsedSrtSegment]:
    """Parse a plain-text SRT document into normalized segments."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("Empty SRT file.")

    blocks = re.split(r"\n\s*\n", normalized)
    segments: list[ParsedSrtSegment] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"Malformed SRT block: {block}")

        try:
            srt_index = int(lines[0])
        except ValueError as exc:
            raise ValueError(f"Invalid SRT index: {lines[0]}") from exc

        if "-->" not in lines[1]:
            raise ValueError(f"Invalid SRT time range: {lines[1]}")

        start_tc, end_tc = [part.strip() for part in lines[1].split("-->", 1)]
        start_ms = parse_timecode_to_ms(start_tc)
        end_ms = parse_timecode_to_ms(end_tc)
        if end_ms <= start_ms:
            raise ValueError(
                f"Invalid SRT range for segment {srt_index}: end must be after start."
            )

        text_lines = lines[2:]
        text = " ".join(text_lines).strip()
        if not text:
            raise ValueError(f"Empty subtitle text in segment {srt_index}.")

        segments.append(
            ParsedSrtSegment(
                srt_index=srt_index,
                start_tc=start_tc,
                end_tc=end_tc,
                start_ms=start_ms,
                end_ms=end_ms,
                target_duration_ms=end_ms - start_ms,
                text_lines=text_lines,
                text=text,
            )
        )

    return segments
