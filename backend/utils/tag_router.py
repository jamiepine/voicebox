"""Parse text into segments, identifying paralinguistic tags for hybrid routing."""

import re
from dataclasses import dataclass
from typing import Literal

PARA_TAGS = frozenset({
    "laugh", "cough", "chuckle", "sigh", "breath",
    "sneeze", "gasp", "yawn", "cry", "groan",
    "sniff", "shush", "whimper", "scream", "whisper",
})

TAG_RE = re.compile(
    r'\[(' + '|'.join(sorted(PARA_TAGS)) + r')\]',
    re.IGNORECASE,
)


@dataclass
class TagSegment:
    type: Literal["text", "tag"]
    content: str


def parse_tagged_text(text: str) -> list[TagSegment]:
    """
    Parse text into ordered segments of plain text and paralinguistic tags.

    Example:
        "Hello [laugh] that was funny" -> [
            TagSegment(type="text", content="Hello"),
            TagSegment(type="tag", content="[laugh]"),
            TagSegment(type="text", content="that was funny"),
        ]
    """
    segments: list[TagSegment] = []
    last_end = 0

    for match in TAG_RE.finditer(text):
        # Text before this tag
        before = text[last_end:match.start()].strip()
        if before:
            segments.append(TagSegment(type="text", content=before))
        # The tag itself
        segments.append(TagSegment(type="tag", content=match.group(0)))
        last_end = match.end()

    # Remaining text after last tag
    after = text[last_end:].strip()
    if after:
        segments.append(TagSegment(type="text", content=after))

    return segments


def has_paralinguistic_tags(text: str) -> bool:
    """Check if text contains any paralinguistic tags."""
    return bool(TAG_RE.search(text))
