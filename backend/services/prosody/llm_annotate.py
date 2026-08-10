"""Draft prosody markup with the local LLM.

The model is an *authoring aid*, never part of rendering. It reads a script and
returns the same script with directives inserted; from there the deterministic
pipeline runs exactly as it does for markup typed by hand. That is what keeps
"with an LLM" and "without an LLM" the same code path downstream, and it is why
generation stays reproducible -- a model in the render path would make the same
script produce different audio on every run.

The invariant that makes this safe to accept
--------------------------------------------
Strip the tags from the model's output and compare to the input. If a single
word moved, the model rewrote the script instead of annotating it, and the
result is thrown away. The model can fail to help; it cannot mangle. Without
that check, an LLM quietly rephrasing a line would be discovered only by
listening to the audio.

Availability
------------
The Qwen LLM ships with most installs but is not guaranteed present, and this
must never be the reason a feature stops working. A missing model is reported,
not downloaded on demand -- the caller falls back to the dictionary and
hand-written markup, which is the whole feature minus the typing assistance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .parser import ProsodyParseError, has_markup, parse, strip_markup

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "1.7B"
# Low, because this is a structural edit rather than a creative one: the same
# script should get the same annotation.
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You annotate a script for a text-to-speech engine.

Insert tags. Never change, add, remove or reorder any word.

Tags you may use:
<lang xml:lang="es">word</lang>  a word or phrase in another language
<break time="700ms"/>            a pause

Rules:
- Wrap foreign words in <lang> with the correct language code.
- Use <break> only where the script clearly wants a beat.
- Output the annotated script and nothing else. No explanation, no quotes.
- If nothing needs annotating, output the script unchanged."""

_EXAMPLES: list[tuple[str, str]] = [
    (
        "He plays a bandeja, not a smash.",
        'He plays a <lang xml:lang="es">bandeja</lang>, not a smash.',
    ),
    (
        "The tempo was allegro throughout.",
        'The tempo was <lang xml:lang="it">allegro</lang> throughout.',
    ),
    ("Nothing unusual in this line.", "Nothing unusual in this line."),
]


@dataclass(frozen=True)
class AnnotationResult:
    """What the annotator produced, and whether it was trustworthy.

    ``markup`` is always safe to use: on rejection it is the original text, so
    a caller can use the result unconditionally and read ``rejected_reason``
    only to explain why nothing changed.
    """

    markup: str
    accepted: bool
    rejected_reason: str | None = None
    model_size: str | None = None
    attempts: int = 0

    @property
    def changed(self) -> bool:
        return self.accepted and has_markup(self.markup)


class LLMUnavailableError(RuntimeError):
    """The local LLM is not downloaded, so annotation cannot run."""


def is_llm_available(model_size: str = DEFAULT_MODEL_SIZE) -> bool:
    """Whether the LLM can run without downloading anything first.

    Deliberately does not trigger a download: annotation is optional help, and
    a feature that silently pulls gigabytes when first used is not optional.
    """
    try:
        from ..llm import get_llm_model

        backend = get_llm_model()
        if backend.is_loaded():
            return True
        return bool(backend._is_model_cached(model_size))
    except Exception:
        logger.debug("LLM availability check failed", exc_info=True)
        return False


def validate_annotation(original: str, candidate: str) -> str | None:
    """Why *candidate* is not an acceptable annotation of *original*.

    Returns ``None`` when it is acceptable. Two things have to hold: it must
    parse, and stripping it must reproduce the input word for word.
    """
    if not candidate or not candidate.strip():
        return "the model returned nothing"

    try:
        parse(candidate)
    except ProsodyParseError as exc:
        return f"the markup is malformed ({exc})"

    if strip_markup(candidate) != strip_markup(original):
        return "the model changed the words instead of only annotating them"

    return None


async def annotate_with_llm(
    text: str,
    *,
    language: str = "en",
    model_size: str = DEFAULT_MODEL_SIZE,
    max_attempts: int = 2,
) -> AnnotationResult:
    """Ask the LLM to mark up *text*.

    Retries once on a rejected candidate, because the usual failure is a model
    wrapping its answer in prose rather than misunderstanding the task, and a
    second attempt with the complaint fed back usually lands.

    Raises:
        LLMUnavailableError: if the model is not downloaded.
    """
    if not text or not text.strip():
        return AnnotationResult(markup=text, accepted=True, model_size=model_size)

    if not is_llm_available(model_size):
        raise LLMUnavailableError(
            f"The {model_size} LLM is not downloaded. Annotation is optional -- "
            "dictionary entries and hand-written markup work without it."
        )

    from ..llm import get_llm_model

    backend = get_llm_model()
    prompt = f"Script language: {language}\n\n{text}"
    last_reason = "the model produced no usable annotation"

    for attempt in range(1, max_attempts + 1):
        try:
            raw = await backend.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_tokens=min(2048, len(text) * 2 + 256),
                temperature=TEMPERATURE,
                model_size=model_size,
                examples=_EXAMPLES,
            )
        except Exception:
            logger.exception("LLM annotation call failed")
            return AnnotationResult(
                markup=text,
                accepted=False,
                rejected_reason="the LLM call failed",
                model_size=model_size,
                attempts=attempt,
            )

        reason = "the model returned nothing"
        for candidate in _candidates(raw):
            reason = validate_annotation(text, candidate)
            if reason is None:
                return AnnotationResult(
                    markup=candidate,
                    accepted=True,
                    model_size=model_size,
                    attempts=attempt,
                )

        last_reason = reason
        logger.info("Rejected LLM annotation (attempt %d): %s", attempt, reason)
        # Feed the complaint back rather than re-asking identically.
        prompt = (
            f"Script language: {language}\n\n{text}\n\n"
            f"Your previous answer was rejected: {reason}. "
            "Return the script with tags inserted and every word unchanged."
        )

    return AnnotationResult(
        markup=text,
        accepted=False,
        rejected_reason=last_reason,
        model_size=model_size,
        attempts=max_attempts,
    )


def _unfence(text: str) -> str:
    """Drop a surrounding ``` block, with or without a language hint."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _unquote(text: str) -> str:
    """Drop one pair of wrapping quotes, which small models add habitually.

    No cleverness about whether that quote also appears inside: markup is full
    of quoted attributes, so any such test would refuse the very case this
    exists for. Whether the unwrapping was right is settled by validating the
    result, not by guessing here.
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {chr(34), chr(39)}:
        return text[1:-1].strip()
    return text


def _candidates(raw: str) -> list[str]:
    """Plausible readings of a model answer, most conservative first.

    Small models wrap their output in fences, quotes, or both. Rather than
    guess which, every unwrapping is offered and the invariant picks the first
    that is a faithful annotation -- so a wrapper is only removed when doing so
    produces something demonstrably correct, and no unwrapping can launder a
    changed word into acceptance.
    """
    text = (raw or "").strip()
    out: list[str] = []
    for candidate in (text, _unfence(text), _unquote(text), _unquote(_unfence(text))):
        if candidate and candidate not in out:
            out.append(candidate)
    return out
