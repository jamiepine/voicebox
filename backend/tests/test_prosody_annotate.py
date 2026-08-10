"""
Tests for turning dictionary entries into markup.

The annotator is the join between the dictionary and the transformer. Its whole
value is that it produces *the same directives an author would have typed*, so
the rules worth pinning are the ones that keep it honest: it must not touch
what the author already marked up, must not reach inside a tag, and must fall
back to something every engine can read when the preferred strategy is not
available.

No database and no model — rules are plain values.

Usage:
    python -m pytest backend/tests/test_prosody_annotate.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.prosody import Speech, TermRule, annotate, compile_plan

BANDEJA = TermRule(term="bandeja", replacement="bandeha")
VIBORA = TermRule(
    term="víbora", replacement="víbora", strategy="language", spoken_language="es"
)
CHIQUITA = TermRule(
    term="chiquita",
    replacement="chi-KEE-ta",
    strategy="phoneme",
    phonemes="tʃiˈkita",  # noqa: RUF001
)


# ── Realisation per strategy ─────────────────────────────────────────


def test_respell_becomes_a_sub():
    out, applied = annotate("He plays a bandeja.", [BANDEJA])
    assert out == 'He plays a <sub alias="bandeha">bandeja</sub>.'
    assert applied == ["bandeja"]


def test_language_becomes_a_lang_span():
    out, _ = annotate("Then the víbora.", [VIBORA])
    assert out == 'Then the <lang xml:lang="es">víbora</lang>.'


def test_phoneme_is_used_where_the_engine_accepts_it():
    out, _ = annotate("A chiquita.", [CHIQUITA], supports_phonemes=True)
    assert "<phoneme" in out
    assert 'ph="tʃiˈkita"' in out  # noqa: RUF001


def test_phoneme_falls_back_to_the_respelling():
    """`replacement` is always populated precisely so there is something every
    engine can read when the preferred strategy is unavailable."""
    out, _ = annotate("A chiquita.", [CHIQUITA], supports_phonemes=False)
    assert "<phoneme" not in out
    assert 'alias="chi-KEE-ta"' in out


def test_a_rule_that_changes_nothing_emits_no_tag():
    """Wrapping text in a tag that does nothing would be noise in the markup and
    an extra cut in the plan."""
    noop = TermRule(term="padel", replacement="padel")
    out, applied = annotate("The padel court.", [noop])
    assert out == "The padel court."
    assert applied == []


# ── Matching ─────────────────────────────────────────────────────────


def test_capitalisation_is_carried_onto_the_alias():
    """The alias is what gets spoken, so one lowercase entry has to cover the
    term at the start of a sentence too."""
    out, _ = annotate("Bandeja is the shot.", [BANDEJA])
    assert 'alias="Bandeha">Bandeja<' in out


def test_all_caps_is_carried():
    out, _ = annotate("A BANDEJA lands deep.", [BANDEJA])
    assert 'alias="BANDEHA"' in out


def test_only_whole_words_match():
    out, applied = annotate("A brandejapalooza.", [BANDEJA])
    assert out == "A brandejapalooza."
    assert applied == []


def test_longer_terms_win():
    alta = TermRule(term="bandeja alta", replacement="bandeha AL-ta")
    out, _ = annotate("A bandeja alta.", [BANDEJA, alta])
    assert 'alias="bandeha AL-ta"' in out


def test_every_occurrence_is_annotated():
    out, applied = annotate("bandeja and bandeja", [BANDEJA])
    assert out.count("<sub") == 2
    assert applied == ["bandeja", "bandeja"]


# ── Not touching what the author wrote ───────────────────────────────


def test_a_hand_wrapped_term_is_left_alone():
    """The author has already said what they want; re-wrapping would nest a
    rule inside their decision."""
    original = 'a <lang xml:lang="it">bandeja</lang> b'
    out, applied = annotate(original, [BANDEJA])
    assert out == original
    assert applied == []


def test_text_outside_the_wrapped_span_is_still_annotated():
    out, applied = annotate(
        'a <lang xml:lang="it">bandeja</lang> and another bandeja', [BANDEJA]
    )
    assert out.count("<sub") == 1
    assert applied == ["bandeja"]


def test_a_term_inside_an_attribute_is_not_rewritten():
    """Rewriting inside an attribute value would corrupt the markup."""
    original = 'x <sub alias="bandeja">y</sub> z'
    out, _ = annotate(original, [BANDEJA])
    assert out == original


def test_a_break_does_not_open_a_protected_region():
    """<break/> is void, so text after it is still ordinary prose."""
    out, applied = annotate('one<break time="700ms"/>a bandeja', [BANDEJA])
    assert applied == ["bandeja"]
    assert "<sub" in out


# ── Degenerate input ─────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_text_is_unchanged(text):
    assert annotate(text, [BANDEJA]) == (text, [])


def test_no_rules_is_a_no_op():
    assert annotate("a bandeja b", []) == ("a bandeja b", [])


def test_a_quote_in_a_replacement_is_escaped():
    """An unescaped quote would end the attribute early and corrupt the tag."""
    quoted = TermRule(term="x", replacement='say "x"')
    out, _ = annotate("a x b", [quoted])
    assert "&quot;" in out
    # And it still parses back to the intended spoken text.
    plan = compile_plan(out, engine="qwen", default_language="en")
    assert 'say "x"' in plan.nodes[0].text


def test_regex_metacharacters_in_a_term_are_literal():
    rule = TermRule(term="C++", replacement="C plus plus")
    out, applied = annotate("I write C++ daily.", [rule])
    assert applied == ["C++"]
    assert 'alias="C plus plus"' in out


# ── Composition with the rest of the pipeline ────────────────────────


def test_annotated_markup_compiles_to_the_expected_plan():
    """The point of emitting markup rather than having a private path: the
    compiler's rules apply to dictionary output for free."""
    out, _ = annotate("He plays a bandeja, then a víbora.", [BANDEJA, VIBORA])
    plan = compile_plan(
        out, engine="qwen", default_language="en", engine_languages=["en", "es"]
    )

    runs = [n for n in plan.nodes if isinstance(n, Speech)]
    assert any(r.language == "es" and "víbora" in r.text for r in runs)
    assert any("bandeha" in r.text for r in runs)


def test_a_respelled_term_does_not_cut_the_sentence():
    """Respelling changes characters, not settings, so the sentence stays whole
    -- the property that makes it the preferred strategy."""
    out, _ = annotate("He plays a bandeja, not a smash.", [BANDEJA])
    plan = compile_plan(out, engine="qwen", default_language="en")
    assert len([n for n in plan.nodes if isinstance(n, Speech)]) == 1


def test_dictionary_terms_compose_with_hand_written_directives():
    out, _ = annotate(
        'He plays a bandeja.<break time="700ms"/>Then a <emphasis>smash</emphasis>.',
        [BANDEJA],
    )
    plan = compile_plan(out, engine="qwen", default_language="en", supports_instruct=True)
    assert any(getattr(n, "ms", None) == 700 for n in plan.nodes)
    assert any("bandeha" in getattr(n, "text", "") for n in plan.nodes)
