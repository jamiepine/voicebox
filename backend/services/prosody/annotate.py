"""Turn dictionary entries into markup.

The dictionary does not get its own execution path. It writes the same
directives an author would have written by hand, and the ordinary
parse-compile-render pipeline takes it from there.

That is worth more than it first looks:

* a dictionary term and a hand-written span compose, because by the time the
  compiler sees them they are the same thing;
* every rule the compiler already enforces -- coalescing, orphaned punctuation,
  engine capability -- applies to dictionary output for free;
* the result is *showable*. A preview can hand back the annotated markup, so a
  rule that fires unexpectedly is visible rather than being an invisible
  difference between what was typed and what was spoken.

The forthcoming LLM annotator emits into exactly the same slot, which is what
keeps "with an LLM" and "without an LLM" the same code path downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .parser import _TAG_RE

# Strategies an entry can ask for, best-first within each engine's abilities.
RESPELL = "respell"
LANGUAGE = "language"
PHONEME = "phoneme"
STRATEGIES = (RESPELL, LANGUAGE, PHONEME)


@dataclass(frozen=True)
class TermRule:
    """One dictionary entry, reduced to what annotation needs.

    Decoupled from the ORM row so the annotator stays pure and testable, and so
    the LLM path can synthesise rules without inventing database objects.
    """

    term: str
    replacement: str
    strategy: str = RESPELL
    spoken_language: str | None = None
    phonemes: str | None = None

    def realise(self, matched: str, *, supports_phonemes: bool) -> str:
        """The markup this rule becomes on an engine with these abilities.

        Falls back to ``replacement`` whenever the preferred strategy is not
        available -- which is why ``replacement`` is always populated, even for
        rules that do not primarily respell.
        """
        if self.strategy == LANGUAGE and self.spoken_language:
            return f'<lang xml:lang="{self.spoken_language}">{matched}</lang>'
        if self.strategy == PHONEME and self.phonemes and supports_phonemes:
            return f'<phoneme alphabet="ipa" ph="{_escape(self.phonemes)}">{matched}</phoneme>'
        if self.replacement.lower() == matched.lower():
            # Nothing to change and no strategy available: leave the text alone
            # rather than wrapping it in a tag that does nothing.
            return matched
        # The alias is what gets spoken, so it has to carry the matched casing:
        # one lowercase entry must cover the term mid-sentence and at the start
        # of one.
        alias = _match_case(matched, self.replacement)
        return f'<sub alias="{_escape(alias)}">{matched}</sub>'


def _match_case(source: str, replacement: str) -> str:
    """Carry the matched text's capitalisation onto the replacement.

    "All caps" counts *cased* characters rather than using ``str.isupper()``,
    which returns True for ``C++`` -- shouting the replacement would turn
    ``C plus plus`` into ``C PLUS PLUS``.
    """
    cased = [c for c in source if c.isalpha()]
    if len(cased) > 1 and all(c.isupper() for c in cased):
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _escape(value: str) -> str:
    """Attribute values are quoted, so a quote inside one would end it early."""
    return value.replace('"', "&quot;")


def _build_pattern(terms: list[str]) -> re.Pattern | None:
    """One alternation over every term, longest first.

    Same construction as the dictionary's own matcher, and for the same reason:
    a single pass means an inserted respelling can never be re-matched by
    another rule, and the longest term wins over one contained inside it.
    """
    usable = sorted({t for t in terms if t and t.strip()}, key=len, reverse=True)
    if not usable:
        return None
    alternation = "|".join(re.escape(t) for t in usable)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE | re.UNICODE)


def _plain_regions(markup: str) -> list[tuple[int, int]]:
    """Spans of *markup* that are outside any tag and outside any tagged span.

    A term already wrapped by hand must be left alone: the author has said what
    they want, and re-wrapping it would nest a rule inside their decision. This
    also keeps the annotator from rewriting the inside of an attribute value.
    """
    regions: list[tuple[int, int]] = []
    cursor = 0
    depth = 0
    for match in _TAG_RE.finditer(markup):
        if depth == 0 and match.start() > cursor:
            regions.append((cursor, match.start()))
        name = match.group("name").lower()
        closing = bool(match.group("closing"))
        self_closing = bool(match.group("void"))
        if name != "break" and not self_closing:
            depth += -1 if closing else 1
            depth = max(depth, 0)
        cursor = match.end()
    if depth == 0 and cursor < len(markup):
        regions.append((cursor, len(markup)))
    return regions


def annotate(
    markup: str,
    rules: list[TermRule],
    *,
    supports_phonemes: bool = False,
) -> tuple[str, list[str]]:
    """Wrap dictionary terms in *markup* with the directives they imply.

    Returns the annotated markup and the terms that were matched, so a caller
    can report what fired rather than silently changing what gets spoken.
    """
    if not markup or not rules:
        return markup, []

    by_term = {r.term.lower(): r for r in rules}
    pattern = _build_pattern([r.term for r in rules])
    if pattern is None:
        return markup, []

    applied: list[str] = []
    out: list[str] = []
    cursor = 0

    for start, end in _plain_regions(markup):
        out.append(markup[cursor:start])
        segment = markup[start:end]

        def substitute(m: re.Match) -> str:
            rule = by_term.get(m.group(0).lower())
            if rule is None:
                return m.group(0)
            realised = rule.realise(m.group(0), supports_phonemes=supports_phonemes)
            if realised == m.group(0):
                return m.group(0)
            applied.append(m.group(0))
            return realised

        out.append(pattern.sub(substitute, segment))
        cursor = end

    out.append(markup[cursor:])
    return "".join(out), applied


def rules_from_entries(entries) -> list[TermRule]:
    """Adapt ORM rows to :class:`TermRule`."""
    return [
        TermRule(
            term=e.term,
            replacement=e.replacement,
            strategy=getattr(e, "strategy", None) or RESPELL,
            spoken_language=getattr(e, "spoken_language", None),
            phonemes=getattr(e, "phonemes", None),
        )
        for e in entries
    ]
