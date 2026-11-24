"""Prosody transformer: a harness around audio segment production.

It never synthesises anything. It decides where to cut a script, what settings
each cut carries, and how the pieces are reassembled -- so pauses, per-span
language and rate work on every engine, including the ones that accept no
directives at all.

    markup ──▶ parse ──▶ compile(engine) ──▶ RenderPlan ──▶ render ──▶ audio

The plan is the seam. It is plain data with no model behind it, so it can be
built, asserted on and previewed for free.
"""

from .annotate import LANGUAGE, PHONEME, RESPELL, TermRule, annotate, rules_from_entries
from .compiler import compile_plan
from .ir import Attrs, Break, PlanWarning, RenderPlan, Silence, Speech, Text
from .parser import ProsodyParseError, has_markup, parse, strip_markup

__all__ = [
    "LANGUAGE",
    "PHONEME",
    "RESPELL",
    "Attrs",
    "Break",
    "PlanWarning",
    "ProsodyParseError",
    "RenderPlan",
    "Silence",
    "Speech",
    "TermRule",
    "Text",
    "annotate",
    "compile_plan",
    "has_markup",
    "parse",
    "rules_from_entries",
    "strip_markup",
]
