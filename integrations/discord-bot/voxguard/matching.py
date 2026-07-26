"""Blacklist matching.

Speech-to-text output is messy: Whisper drops punctuation, splits words oddly,
and people deliberately obfuscate ("f u c k", "fuuuck", "sh1t"). A plain
substring scan catches almost none of that, and a naive one catches far too
much — the classic failure being "Scunthorpe" tripping a filter for the town's
fourth through seventh letters.

The approach here is to normalise text and terms the same way, then match at
three decreasing levels of confidence, with an allowlist that can veto a hit.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

# Characters people substitute to slip past filters.
LEET = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "6": "g",
        "7": "t",
        "8": "b",
        "9": "g",
        "@": "a",
        "$": "s",
        "!": "i",
        "|": "i",
        "+": "t",
        "€": "e",
        "£": "l",
    }
)

_NON_WORD = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")
_REPEATS = re.compile(r"(.)\1+")

# Fuzzy matching only runs on terms at least this long — below it, near-misses
# are almost always real words ("bad" vs "bat", "hell" vs "held").
MIN_FUZZY_LEN = 5
FUZZY_THRESHOLD = 88

# Operator-supplied `re:` patterns run against every transcript line, so a
# pathological one stalls the whole voice pipeline. These bounds reject the
# shapes that cause catastrophic backtracking rather than trying to detect it
# at match time.
MAX_REGEX_LEN = 200
# A quantifier applied to an already-quantified group — (a+)+, (a*)* , (\d+)*
# and friends — is the classic exponential-backtracking construct.
_NESTED_QUANTIFIER = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*{]")
# Text scanned per line is bounded anyway, but cap it so a very long
# transcript can't multiply a merely-slow pattern into a stall.
MAX_REGEX_INPUT = 2000


def safe_regex(pattern: str) -> re.Pattern[str] | None:
    """Compile an operator-supplied pattern, or None if it looks unsafe."""
    if not pattern or len(pattern) > MAX_REGEX_LEN:
        return None
    if _NESTED_QUANTIFIER.search(pattern):
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


@dataclass(frozen=True)
class Term:
    text: str
    kind: str  # word | phrase | regex
    severity: int = 1


@dataclass(frozen=True)
class Match:
    term: str
    kind: str
    severity: int
    matched: str
    confidence: float
    method: str  # exact | obfuscated | fuzzy | regex


def normalize(text: str) -> str:
    """Fold case, strip accents and leetspeak, reduce punctuation to spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = stripped.casefold().translate(LEET)
    spaced = _NON_WORD.sub(" ", lowered)
    return _WHITESPACE.sub(" ", spaced).strip()


def squeeze(normalized: str) -> str:
    """Collapse repeated characters and remove separators.

    Applied to the term and the text alike, so "pass" and "p a s s" both become
    "pas" and still match each other. This is what catches "fuuuuck" and
    "f u c k" without needing a rule per variation.
    """
    return _REPEATS.sub(r"\1", normalized.replace(" ", ""))


def _boundary_pattern(term_norm: str) -> re.Pattern[str]:
    # Terms may be multi-word; allow any run of separators between the parts.
    parts = [re.escape(p) for p in term_norm.split(" ") if p]
    if not parts:
        return re.compile(r"(?!)")
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b")


class Matcher:
    """Compiled view of one guild's block + allow lists for a single scope."""

    def __init__(self, blocked: list[Term], allowed: list[Term] | None = None) -> None:
        self.blocked = blocked
        self.allowed = allowed or []
        # Patterns refused as unsafe, surfaced by `/blacklist list` so an
        # operator finds out rather than silently getting no matches.
        self.rejected: list[str] = []
        self._compiled: list[tuple[Term, re.Pattern[str] | None, str, str]] = []
        for term in blocked:
            if term.kind == "regex":
                compiled = safe_regex(term.text)
                if compiled is None:
                    # A bad or dangerous pattern shouldn't take the matcher
                    # down, nor stall every transcript line that follows.
                    self.rejected.append(term.text)
                    continue
                self._compiled.append((term, compiled, "", ""))
            else:
                norm = normalize(term.text)
                if not norm:
                    continue
                self._compiled.append((term, _boundary_pattern(norm), norm, squeeze(norm)))

        self._allow_norm = [normalize(a.text) for a in self.allowed if normalize(a.text)]
        self._allow_squeezed = [squeeze(a) for a in self._allow_norm]

    def __len__(self) -> int:
        return len(self._compiled)

    def _allowed_covers(self, text_norm: str, span: tuple[int, int]) -> bool:
        """True if the matched span sits inside an allowlisted word."""
        for allowed in self._allow_norm:
            for hit in _boundary_pattern(allowed).finditer(text_norm):
                if hit.start() <= span[0] and hit.end() >= span[1]:
                    return True
        return False

    def scan(self, text: str, *, min_confidence: float = 0.0) -> list[Match]:
        if not text.strip() or not self._compiled:
            return []

        text_norm = normalize(text)
        text_squeezed = squeeze(text_norm)
        tokens = text_norm.split(" ")
        results: dict[str, Match] = {}

        def offer(match: Match) -> None:
            if match.confidence < min_confidence:
                return
            existing = results.get(match.term)
            if existing is None or match.confidence > existing.confidence:
                results[match.term] = match

        for term, pattern, term_norm, term_squeezed in self._compiled:
            if term.kind == "regex":
                if pattern is None:
                    continue
                hit = pattern.search(text[:MAX_REGEX_INPUT])
                if hit:
                    offer(
                        Match(term.text, term.kind, term.severity, hit.group(0), 1.0, "regex")
                    )
                continue

            assert pattern is not None
            hit = pattern.search(text_norm)
            if hit and not self._allowed_covers(text_norm, hit.span()):
                offer(Match(term.text, term.kind, term.severity, hit.group(0), 1.0, "exact"))
                continue

            # Obfuscation pass: spacing and character repetition removed.
            if len(term_squeezed) >= 4 and term_squeezed in text_squeezed:
                if not any(term_squeezed in allowed for allowed in self._allow_squeezed):
                    offer(
                        Match(term.text, term.kind, term.severity, term_squeezed, 0.85, "obfuscated")
                    )
                    continue

            # Fuzzy pass: catches transcription slips ("shiit", "biitch").
            if term.kind == "word" and len(term_norm) >= MIN_FUZZY_LEN:
                best = 0
                best_token = ""
                for token in tokens:
                    if abs(len(token) - len(term_norm)) > 2:
                        continue
                    score = fuzz.ratio(token, term_norm)
                    if score > best:
                        best, best_token = score, token
                if best >= FUZZY_THRESHOLD and best_token not in self._allow_norm:
                    offer(
                        Match(
                            term.text,
                            term.kind,
                            term.severity,
                            best_token,
                            round(best / 100 * 0.9, 3),
                            "fuzzy",
                        )
                    )

        return sorted(results.values(), key=lambda m: (-m.severity, -m.confidence))


def parse_terms(raw: str) -> list[tuple[str, str, int]]:
    """Parse a word list into (term, kind, severity) triples.

    One term per line or comma-separated. Supported forms::

        badword                 a single word
        bad phrase here         a phrase (matched with word boundaries)
        re:^spam.*bot$          a regular expression
        badword | 3             an explicit severity (default 1)
        # comment               ignored

    Severity feeds escalation: a severity-3 hit can be configured to skip
    straight past the warning ladder.
    """
    out: list[tuple[str, str, int]] = []
    seen: set[str] = set()

    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # A line without a comma is a single term (possibly a phrase).
        chunks = [line] if "," not in line else [c for c in line.split(",")]
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or chunk.startswith("#"):
                continue

            severity = 1
            if "|" in chunk:
                head, _, tail = chunk.rpartition("|")
                if tail.strip().isdigit():
                    chunk, severity = head.strip(), max(1, min(3, int(tail.strip())))

            if chunk.lower().startswith("re:"):
                term, kind = chunk[3:].strip(), "regex"
            else:
                term = chunk
                kind = "phrase" if " " in chunk else "word"

            if not term:
                continue
            key = f"{kind}:{term.casefold()}"
            if key in seen:
                continue
            seen.add(key)
            out.append((term, kind, severity))

    return out
