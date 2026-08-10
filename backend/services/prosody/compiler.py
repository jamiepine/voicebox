"""Flatten a parsed script into a RenderPlan for one specific engine.

This is where the harness earns its name. Each directive has up to two
realisations:

*native*
    tell the engine — ``instruct=`` on the engines that honour it

*structural*
    cut the text, render the run with its own settings, reassemble — which
    needs no engine support whatsoever

Pauses, language spans and rate are all structural, so they work identically on
all eight engines. Delivery cues are the only ones that depend on capability,
and where an engine cannot take them the plan says so out loud instead of
dropping them silently.

Nothing here loads a model or touches audio. A plan can be built, asserted on
and shown to a user for free, which is what makes the pipeline previewable.
"""

from __future__ import annotations

from dataclasses import replace

from .ir import Attrs, Break, Node, PlanWarning, RenderPlan, Silence, Span, Speech, Text
from .parser import parse


def _is_over_articulated(alias: str) -> bool:
    """Whether a respelling is shaped like the one that tested worst.

    ``ban-DEH-ha`` -- syllable hyphens plus capitals for stress -- was judged
    exaggerated on every voice tried, and measurably so: it ran about 30%
    longer than the same sentence unmarked. ``ban-deh-ha`` was acceptable and
    plain ``bandeha`` sounded most natural, so it is the *combination* that
    misfires, not either alone.

    Acronym expansions like ``W C A G`` are capitals without hyphens and are
    deliberately not flagged.
    """
    return "-" in alias and any(c.isupper() for c in alias[1:])


def _flatten(nodes: list[Node], inherited: Attrs, out: list[tuple[str, Attrs] | int]) -> None:
    """Walk the tree into a flat run of (text, attrs) pairs and break markers."""
    for node in nodes:
        if isinstance(node, Text):
            if node.value:
                out.append((node.value, inherited))
        elif isinstance(node, Break):
            out.append(node.ms)
        elif isinstance(node, Span):
            _flatten(node.children, inherited.merged_with(node.attrs), out)


def _resolve(items: list[tuple[str, Attrs] | int]) -> list[tuple[str, str, Attrs] | int]:
    """Apply substitutions, producing (spoken, written, attrs).

    Once ``<sub>``/``<phoneme>`` has been materialised into the spoken text it
    stops being a setting, so it must not keep the run apart from its
    neighbours afterwards. ``spoken_as`` is cleared here for exactly that
    reason -- it governs inheritance, not rendering.
    """
    out: list[tuple[str, str, Attrs] | int] = []
    for item in items:
        if isinstance(item, int):
            out.append(item)
            continue
        written, attrs = item
        spoken = written if attrs.spoken_as is None else attrs.spoken_as
        out.append((spoken, written, replace(attrs, spoken_as=None)))
    return out


def _coalesce(items: list[tuple[str, str, Attrs] | int]) -> list[tuple[str, str, Attrs] | int]:
    """Merge neighbouring runs that render identically.

    Every avoided cut is one less place the model restarts its prosody, which
    is the entire cost of this approach. It matters most for ``<sub>``: a
    respelling changes the characters, not the settings, so the sentence around
    it should stay in one piece. Cutting there would buy the seams that
    respelling exists to avoid.
    """
    merged: list[tuple[str, str, Attrs] | int] = []
    for item in items:
        if (
            isinstance(item, tuple)
            and merged
            and isinstance(merged[-1], tuple)
            and merged[-1][2] == item[2]
        ):
            prev = merged[-1]
            merged[-1] = (prev[0] + item[0], prev[1] + item[1], prev[2])
        else:
            merged.append(item)
    return merged


def compile_plan(
    markup: str,
    *,
    engine: str,
    default_language: str,
    supports_instruct: bool = False,
    engine_languages: list[str] | None = None,
    base_instruct: str | None = None,
) -> RenderPlan:
    """Compile *markup* into a plan for *engine*.

    Args:
        markup: The script, with or without directives.
        engine: Target engine id, recorded on the plan.
        default_language: Language for text outside any ``<lang>``.
        supports_instruct: From the model registry, not guessed here.
        engine_languages: What the engine can generate. A ``<lang>`` outside
            this set is a warning rather than an error -- the run still
            renders, in the engine's own language, which is what it would have
            done anyway.
        base_instruct: The request's own delivery instruction, which
            ``<emphasis>`` composes with rather than replaces.
    """
    nodes = parse(markup)
    flat = _coalesce(_resolve(_walk(nodes)))

    plan_nodes: list[Speech | Silence] = []
    warnings: list[PlanWarning] = []
    seen_unsupported_language: set[str] = set()
    emphasis_dropped = False

    for item in flat:
        if isinstance(item, int):
            if item > 0:
                plan_nodes.append(Silence(item))
            continue

        text, raw_text, attrs = item
        if not text.strip():
            # Whitespace between tags is not a run of its own, but it must not
            # be lost either -- glue it onto the previous run.
            if plan_nodes and isinstance(plan_nodes[-1], Speech):
                prev = plan_nodes[-1]
                plan_nodes[-1] = Speech(
                    text=prev.text + raw_text,
                    language=prev.language,
                    rate=prev.rate,
                    instruct=prev.instruct,
                    seed=prev.seed,
                    source_text=prev.source_text,
                )
            continue

        language = attrs.language or default_language
        if (
            engine_languages
            and language not in engine_languages
            and language not in seen_unsupported_language
        ):
            seen_unsupported_language.add(language)
            warnings.append(
                PlanWarning(
                    code="language_unsupported",
                    detail=(
                        f"Engine {engine!r} cannot generate {language!r}; that run will be "
                        f"read as {default_language!r}."
                    ),
                )
            )
            language = default_language

        instruct = base_instruct
        if attrs.emphasis:
            if supports_instruct:
                cue = f"Say this with {attrs.emphasis} emphasis."
                instruct = f"{base_instruct} {cue}".strip() if base_instruct else cue
            elif not emphasis_dropped:
                emphasis_dropped = True
                warnings.append(
                    PlanWarning(
                        code="emphasis_unsupported",
                        detail=(
                            f"Engine {engine!r} does not honour delivery instructions, so "
                            f"<emphasis> has no effect. Try qwen_custom_voice."
                        ),
                    )
                )

        if text != raw_text and _is_over_articulated(text):
            warnings.append(
                PlanWarning(
                    code="over_articulated_respelling",
                    detail=(
                        f"{text!r} combines hyphens with capitals, which makes the engine "
                        f"over-articulate: in testing that read as exaggerated and ran ~30% "
                        f"longer than the same sentence. A plain letter substitution "
                        f"(bandeja -> bandeha) sounded most natural."
                    ),
                )
            )

        plan_nodes.append(
            Speech(
                text=text,
                language=language,
                rate=attrs.rate or 1.0,
                instruct=instruct,
                source_text=raw_text if text != raw_text else None,
            )
        )

    return RenderPlan(nodes=_absorb_unspeakable(plan_nodes), warnings=warnings, engine=engine)


def _has_speech(text: str) -> bool:
    """Whether a run contains anything a model could pronounce."""
    return any(ch.isalnum() for ch in text)


def _absorb_unspeakable(nodes: list[Speech | Silence]) -> list[Speech | Silence]:
    """Fold runs with no pronounceable content into a neighbour.

    A span boundary almost always orphans its trailing punctuation --
    ``</lang>.`` leaves a run holding just ``"."``. Generating that is a wasted
    call that returns noise or silence, so it is appended to the run before it
    (or prepended to the one after, when it comes first).

    Substituted runs are never merged: a ``<sub>`` applies to specific words,
    and absorbing text into it would put words through a substitution the
    author never wrapped.
    """
    out: list[Speech | Silence] = []
    for node in nodes:
        if isinstance(node, Speech) and not _has_speech(node.text):
            prev = out[-1] if out else None
            if isinstance(prev, Speech) and prev.source_text is None:
                out[-1] = replace(prev, text=prev.text + node.text)
                continue
        out.append(node)

    # A leading orphan has no predecessor; give it to the following run.
    if len(out) > 1 and isinstance(out[0], Speech) and not _has_speech(out[0].text):
        first, second = out[0], out[1]
        if isinstance(second, Speech) and second.source_text is None:
            out = [replace(second, text=first.text + second.text), *out[2:]]

    # Everything may have been punctuation; keep it rather than return nothing.
    return out or nodes


def _walk(nodes: list[Node]) -> list[tuple[str, Attrs] | int]:
    out: list[tuple[str, Attrs] | int] = []
    _flatten(nodes, Attrs(), out)
    return out
