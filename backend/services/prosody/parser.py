"""Parse the SSML subset into the prosody IR.

Not an XML parser, deliberately. A script is prose: it contains ``&``, it
contains ``5 < 6``, and an XML parser rejects both. Instead this recognises a
*closed set* of known tags and treats everything else as literal text, which
makes it strictly more forgiving than XML on the input it actually gets --
nothing needs escaping unless it happens to spell one of our tags.

The subset is chosen so that the two pronunciation strategies are SSML's own
(``<sub>`` and ``<phoneme>``) rather than something invented alongside them.

Supported::

    <break time="700ms"/>            silence, also 0.7s; bare <break/> is 700ms
    <lang xml:lang="es">…</lang>     render this run in another language
    <prosody rate="0.9">…</prosody>  speaking rate
    <emphasis level="strong">…</emphasis>
    <sub alias="ban-DEH-ha">bandeja</sub>
    <phoneme alphabet="ipa" ph="…">bandeja</phoneme>

Angle brackets rather than square ones because square ones are taken:
``[laugh]`` is a Chatterbox Turbo paralinguistic tag, which is passed *to* the
engine, where a directive is intercepted *before* it. Same delimiter, opposite
behaviour -- see ``_PARA_TAG_RE`` in ``utils/chunked_tts.py``.
"""

from __future__ import annotations

import re

from .ir import Attrs, Break, Node, Span, Text

# Every tag the parser knows. Anything else stays literal text.
_VOID_TAGS = {"break"}
_SPAN_TAGS = {"lang", "prosody", "emphasis", "sub", "phoneme"}
_ALL_TAGS = _VOID_TAGS | _SPAN_TAGS

_TAG_RE = re.compile(
    r"<\s*(?P<closing>/)?\s*(?P<name>" + "|".join(sorted(_ALL_TAGS)) + r")"
    r"(?P<attrs>[^<>]*?)(?P<void>/)?\s*>",
    re.IGNORECASE,
)

_ATTR_RE = re.compile(r"""(?P<key>[\w:.-]+)\s*=\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)""")

# "700ms", "0.7s", "700" (bare numbers read as milliseconds).
_DURATION_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)?\s*$", re.IGNORECASE)

MAX_BREAK_MS = 60_000

# What a bare <break/> means. 700ms was judged a good pause on every voice
# tried; 1500ms read as too long unless the script genuinely wants a beat to
# stop and think. The natural gap after a full stop is already 210-440ms
# depending on the voice, so this adds to a pause rather than creating one.
DEFAULT_BREAK_MS = 700


class ProsodyParseError(ValueError):
    """The markup is malformed in a way that would change what gets spoken."""


def parse_duration(raw: str) -> int:
    """``"700ms"``/``"0.7s"``/``"700"`` -> milliseconds."""
    m = _DURATION_RE.match(raw or "")
    if not m:
        raise ProsodyParseError(f"Not a duration: {raw!r}. Use e.g. \"700ms\" or \"0.7s\".")
    value = float(m.group("value"))
    ms = round(value * 1000) if (m.group("unit") or "ms").lower() == "s" else round(value)
    if ms < 0 or ms > MAX_BREAK_MS:
        raise ProsodyParseError(f"Break of {ms}ms is outside 0..{MAX_BREAK_MS}ms.")
    return ms


# Attribute values are quoted, so anything writing one has to escape a quote
# inside it -- and this has to undo that, or the engine speaks the entity.
# Ampersand is unescaped last so "&amp;quot;" survives as the literal
# "&quot;" rather than collapsing into a quote.
_ENTITIES = (("&quot;", '"'), ("&apos;", "'"), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"))


def _unescape(value: str) -> str:
    for entity, char in _ENTITIES:
        value = value.replace(entity, char)
    return value


def _parse_attrs(raw: str) -> dict[str, str]:
    return {
        m.group("key").lower(): _unescape(m.group("value"))
        for m in _ATTR_RE.finditer(raw or "")
    }


def _rate_from(raw: str | None) -> float | None:
    """``"0.9"`` or ``"90%"`` -> a multiplier.

    Rejects zero and negatives rather than clamping: a rate of 0 means audio of
    infinite length, and silently substituting 1.0 would hide a typo behind
    output that sounds fine.
    """
    if raw is None:
        return None
    text = raw.strip()
    try:
        value = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    except ValueError as exc:
        raise ProsodyParseError(f"Not a rate: {raw!r}. Use e.g. \"0.9\" or \"90%\".") from exc
    if not 0.1 <= value <= 5.0:
        raise ProsodyParseError(f"Rate {value} is outside 0.1..5.0.")
    return value


def _attrs_for(tag: str, attrs: dict[str, str]) -> Attrs:
    if tag == "lang":
        code = attrs.get("xml:lang") or attrs.get("lang")
        if not code:
            raise ProsodyParseError('<lang> needs xml:lang, e.g. <lang xml:lang="es">.')
        return Attrs(language=code.strip().lower())

    if tag == "prosody":
        rate = _rate_from(attrs.get("rate"))
        if rate is None:
            raise ProsodyParseError('<prosody> supports rate, e.g. <prosody rate="0.9">.')
        return Attrs(rate=rate)

    if tag == "emphasis":
        return Attrs(emphasis=(attrs.get("level") or "moderate").strip().lower())

    if tag == "sub":
        alias = attrs.get("alias")
        if not alias or not alias.strip():
            raise ProsodyParseError('<sub> needs alias, e.g. <sub alias="ban-DEH-ha">.')
        return Attrs(spoken_as=alias.strip())

    if tag == "phoneme":
        ph = attrs.get("ph")
        if not ph or not ph.strip():
            raise ProsodyParseError('<phoneme> needs ph, e.g. <phoneme ph="banˈdexa">.')  # noqa: RUF001
        # Carried as a substitution; whether the engine can take phonemes at
        # all is a capability question the compiler answers, not the parser.
        return Attrs(spoken_as=ph.strip())

    return Attrs()


def parse(markup: str) -> list[Node]:
    """Parse *markup* into a node tree.

    Raises:
        ProsodyParseError: on a malformed or unbalanced tag. Malformed markup
            is an error rather than being passed through as text, because
            passing it through means the engine reads it aloud.
    """
    if not markup:
        return []

    root = Span(attrs=Attrs(), children=[], tag="")
    stack: list[Span] = [root]
    cursor = 0

    def emit_text(chunk: str) -> None:
        if chunk:
            stack[-1].children.append(Text(chunk))

    for match in _TAG_RE.finditer(markup):
        emit_text(markup[cursor : match.start()])
        cursor = match.end()

        name = match.group("name").lower()
        closing = bool(match.group("closing"))
        self_closing = bool(match.group("void"))
        attrs = _parse_attrs(match.group("attrs"))

        if closing:
            if len(stack) == 1 or stack[-1].tag != name:
                expected = stack[-1].tag if len(stack) > 1 else "nothing"
                raise ProsodyParseError(
                    f"</{name}> does not match the open tag ({expected})."
                )
            stack.pop()
            continue

        if name in _VOID_TAGS or self_closing:
            if name == "break":
                raw_time = attrs.get("time")
                ms = DEFAULT_BREAK_MS if raw_time is None else parse_duration(raw_time)
                stack[-1].children.append(Break(ms))
            continue

        span = Span(attrs=_attrs_for(name, attrs), children=[], tag=name)
        stack[-1].children.append(span)
        stack.append(span)

    emit_text(markup[cursor:])

    if len(stack) > 1:
        raise ProsodyParseError(f"<{stack[-1].tag}> was never closed.")

    return root.children


_STRIP_RE = re.compile(
    r"<\s*/?\s*(?:" + "|".join(sorted(_ALL_TAGS)) + r")(?:[^<>]*?)/?\s*>", re.IGNORECASE
)


def strip_markup(markup: str) -> str:
    """Remove every known tag, leaving the words.

    This is what makes LLM annotation safe to accept: strip the model's output
    and compare it to the input. If a single word moved, the model rewrote the
    script instead of annotating it, and the result is rejected. The model can
    fail to help, but it cannot mangle.

    Whitespace is normalised on both sides, since inserting a tag on its own
    line is a formatting change rather than a content one.
    """
    return re.sub(r"\s+", " ", _STRIP_RE.sub("", markup or "")).strip()


def has_markup(text: str) -> bool:
    """Whether any known tag is present, so unmarked text can skip the harness."""
    return bool(text) and _TAG_RE.search(text) is not None
