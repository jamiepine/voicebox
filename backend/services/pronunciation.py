"""Pronunciation dictionary — reusable fixes for terms the engine says wrong.

Names, acronyms, brands and loanwords come out wrong and there is no way to
correct them short of editing the text every time (#827). This maps a term to a
respelling applied just before TTS.

Respelling rather than phonemes on purpose: every engine reads plain text, so
``bandeja -> ban-DEH-ha`` works on all of them, where a phoneme string only
works on the engines that accept one. It is cruder and it is portable.

Applied at generation time, not when the text is saved, so ``generations.text``
keeps what the author wrote. Editing an entry then changes future audio without
rewriting history, and the History tab never shows a reader ``ban-DEH-ha``.
"""

import logging
import re

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import PronunciationEntry

logger = logging.getLogger(__name__)

# Paralinguistic tags like [laugh] are engine syntax, not speech. A term that
# happens to appear inside one must not be rewritten.
_TAG_RE = re.compile(r"\[[^\]]*\]")


def get_entries(
    db: Session,
    language: str | None = None,
    profile_id: str | None = None,
    include_disabled: bool = False,
) -> list[PronunciationEntry]:
    """Entries that apply to a given language and profile.

    An entry with a NULL language or profile is a wildcard, so the filters ask
    for "matches, or unset" rather than equality.
    """
    q = db.query(PronunciationEntry)
    if not include_disabled:
        q = q.filter(PronunciationEntry.enabled.is_(True))
    if language is not None:
        q = q.filter(
            or_(PronunciationEntry.language.is_(None), PronunciationEntry.language == language)
        )
    if profile_id is not None:
        q = q.filter(
            or_(PronunciationEntry.profile_id.is_(None), PronunciationEntry.profile_id == profile_id)
        )
    else:
        q = q.filter(PronunciationEntry.profile_id.is_(None))
    return q.all()


def _match_case(source: str, replacement: str) -> str:
    """Carry the matched text's capitalisation onto the replacement.

    A term at the start of a sentence is capitalised there and nowhere else, so
    storing one lowercase entry has to cover both.

    "All caps" counts *cased* characters, not ``str.isupper()``: that returns
    True for ``C++``, which has one cased letter, and shouting the replacement
    would turn ``C plus plus`` into ``C PLUS PLUS``. An acronym like ``WCAG``
    has four and is genuinely all caps.
    """
    cased = [c for c in source if c.isalpha()]
    if len(cased) > 1 and all(c.isupper() for c in cased):
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _tag_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _TAG_RE.finditer(text)]


def build_pattern(terms: list[str]) -> re.Pattern | None:
    """One alternation over every term, longest first.

    Longest-first matters twice. It lets a multi-word entry beat the
    single-word entry inside it, and because this is a single pass, a
    replacement can never be re-matched by another rule — so
    ``bandeja -> ban-DEH-ha`` and ``ha -> hah`` cannot compound into
    ``ban-DEH-hah``, which is what a loop of per-term substitutions would do.

    Lookarounds rather than ``\\b`` so terms that start or end with punctuation
    still anchor on a word boundary.
    """
    usable = [t for t in terms if t and t.strip()]
    if not usable:
        return None
    ordered = sorted(set(usable), key=len, reverse=True)
    alternation = "|".join(re.escape(t) for t in ordered)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE | re.UNICODE)


def apply_pronunciations(
    text: str,
    language: str | None,
    db: Session,
    profile_id: str | None = None,
) -> tuple[str, list[dict]]:
    """Rewrite *text* using the dictionary.

    Returns the rewritten text and a record of what was replaced, so a caller
    can log or surface it — a silent rewrite of someone's script is worse than
    no rewrite at all.
    """
    if not text or not text.strip():
        return text, []

    entries = get_entries(db, language=language, profile_id=profile_id)
    if not entries:
        return text, []

    # A profile-scoped entry beats a global one for the same term; a
    # language-specific entry beats a wildcard. Sorting the losers first lets
    # the later assignment win.
    by_term: dict = {}
    for e in sorted(
        entries,
        key=lambda e: ((e.profile_id is not None), (e.language is not None)),
    ):
        by_term[e.term.lower()] = e

    pattern = build_pattern([e.term for e in by_term.values()])
    if pattern is None:
        return text, []

    skip = _tag_spans(text)
    applied: list[dict] = []

    def substitute(m: re.Match) -> str:
        if any(start <= m.start() < end for start, end in skip):
            return m.group(0)
        entry = by_term.get(m.group(0).lower())
        if entry is None:
            return m.group(0)
        out = _match_case(m.group(0), entry.replacement)
        applied.append({"term": m.group(0), "replacement": out, "entry_id": entry.id})
        return out

    result = pattern.sub(substitute, text)
    if applied:
        logger.info(
            "Pronunciation dictionary rewrote %d term(s): %s",
            len(applied),
            ", ".join(f"{a['term']}->{a['replacement']}" for a in applied[:5]),
        )
    return result, applied


def find_duplicate(
    db: Session,
    term: str,
    language: str | None,
    profile_id: str | None,
    exclude_id: str | None = None,
) -> PronunciationEntry | None:
    """An existing entry for the same term in the same scope.

    Enforced here rather than as a unique constraint because SQL treats NULLs
    as distinct, so a constraint would happily accept two global entries for
    the same term — exactly the case worth catching.
    """
    q = db.query(PronunciationEntry).filter(PronunciationEntry.term.ilike(term))
    q = (
        q.filter(PronunciationEntry.language.is_(None))
        if language is None
        else q.filter(PronunciationEntry.language == language)
    )
    q = (
        q.filter(PronunciationEntry.profile_id.is_(None))
        if profile_id is None
        else q.filter(PronunciationEntry.profile_id == profile_id)
    )
    if exclude_id:
        q = q.filter(PronunciationEntry.id != exclude_id)
    return q.first()
