"""Intermediate representation for the prosody transformer.

The transformer is a harness around segment production: it never synthesises
anything, it decides where to cut, what settings each cut carries, and how the
pieces are reassembled. This module is the vocabulary those decisions are
expressed in.

Two layers:

``Directive``/``Span``
    The parse tree. Mirrors the markup, so spans nest and attributes inherit.

``RenderPlan``
    The flattened, resolved result: a list of speech runs and silences with
    every attribute already decided for one specific engine. Nothing after this
    point has to know the markup existed.

The plan is deliberately a plain data object with no model behind it. It can be
built, asserted on, diffed and shown to a user without loading 3.5 GB of
weights, which is what makes the whole pipeline testable and previewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# ── Parse tree ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Attrs:
    """Settings that flow down the span tree.

    ``None`` means "inherit from the enclosing span", which is what lets
    ``<prosody rate="0.8">`` wrap a ``<lang>`` without either having to know
    about the other.
    """

    language: str | None = None
    rate: float | None = None
    instruct: str | None = None
    emphasis: str | None = None
    # Set by <sub alias="..."> and <phoneme ph="...">: the text handed to the
    # engine differs from the text the author wrote.
    spoken_as: str | None = None

    def merged_with(self, child: Attrs) -> Attrs:
        """Child wins where it says anything; parent shows through elsewhere."""
        return Attrs(
            language=child.language if child.language is not None else self.language,
            rate=child.rate if child.rate is not None else self.rate,
            instruct=child.instruct if child.instruct is not None else self.instruct,
            emphasis=child.emphasis if child.emphasis is not None else self.emphasis,
            # Deliberately not inherited: a substitution applies to the text it
            # wraps, not to everything nested inside a parent that had one.
            spoken_as=child.spoken_as,
        )


@dataclass(frozen=True)
class Text:
    """Literal text to be spoken."""

    value: str


@dataclass(frozen=True)
class Break:
    """A silence, in milliseconds.

    No engine accepts this, so it is always realised structurally -- the
    renderer emits silence and the model never sees it. That is why pauses work
    identically on all eight engines.
    """

    ms: int


@dataclass(frozen=True)
class Span:
    """A run of nodes sharing a set of attribute overrides."""

    attrs: Attrs
    children: list[Node] = field(default_factory=list)
    # The tag this came from, kept for error messages and round-tripping.
    tag: str = ""


Node = Text | Break | Span


# ── Render plan ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Speech:
    """One generation call: text plus every setting already resolved."""

    text: str
    language: str
    rate: float = 1.0
    instruct: str | None = None
    seed: int | None = None
    # What the author wrote, when `text` was substituted by <sub>/<phoneme> or
    # a dictionary entry. Kept so a preview can show the change rather than
    # silently handing back something the author never typed.
    source_text: str | None = None


@dataclass(frozen=True)
class Silence:
    """A gap, produced by assembly rather than by the engine."""

    ms: int


PlanNode = Speech | Silence


@dataclass(frozen=True)
class PlanWarning:
    """Something the target engine cannot honour.

    Carried on the plan rather than logged and forgotten: an instruction that
    silently does nothing reads as the model refusing to follow it, which is
    the complaint behind #579.
    """

    code: str
    detail: str


@dataclass(frozen=True)
class RenderPlan:
    """Everything the renderer needs, for one engine, with nothing left to decide."""

    nodes: list[PlanNode] = field(default_factory=list)
    warnings: list[PlanWarning] = field(default_factory=list)
    engine: str = ""

    @property
    def speech_nodes(self) -> list[Speech]:
        return [n for n in self.nodes if isinstance(n, Speech)]

    @property
    def is_trivial(self) -> bool:
        """Whether this needs none of the harness: one run, no assembly.

        The overwhelmingly common case, and the caller takes the existing
        single-shot path for it, so unmarked text costs nothing.

        A substitution does *not* make a plan non-trivial. ``source_text`` is
        provenance for display -- by this point the respelling is already in
        ``text``, and one run with no silences and no rate change has nothing
        for the renderer to assemble. Treating it as non-trivial would send
        every respelled sentence through the renderer for no benefit, and would
        contradict the property that makes respelling the preferred strategy:
        that it does not cut the sentence.
        """
        return (
            len(self.nodes) == 1
            and isinstance(self.nodes[0], Speech)
            and self.nodes[0].rate == 1.0
        )

    def with_seeds(self, base_seed: int | None) -> RenderPlan:
        """Assign a deterministic per-run seed.

        Varied per run so neighbouring runs do not share RNG artefacts, but
        derived from ``base_seed`` so the same plan always renders the same
        audio. ``None`` stays ``None`` -- an unseeded plan should still vary
        between takes.
        """
        if base_seed is None:
            return self
        out: list[PlanNode] = []
        speech_index = 0
        for node in self.nodes:
            if isinstance(node, Speech):
                out.append(replace(node, seed=base_seed + speech_index))
                speech_index += 1
            else:
                out.append(node)
        return replace(self, nodes=out)
