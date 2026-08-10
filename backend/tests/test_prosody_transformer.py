"""
Tests for the prosody transformer's parse and compile stages.

No model, no audio, no database — the whole point of making the RenderPlan a
plain data object is that the decisions can be asserted on for free.

The rules worth pinning are the ones that are easy to get subtly wrong:
attributes inherit but substitutions do not, neighbouring runs coalesce so the
model restarts its prosody as rarely as possible, punctuation orphaned by a
span boundary never becomes its own generation, and anything the target engine
cannot honour is said out loud rather than dropped.

Usage:
    python -m pytest backend/tests/test_prosody_transformer.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.prosody import (
    ProsodyParseError,
    RenderPlan,
    Silence,
    Speech,
    compile_plan,
    has_markup,
    parse,
    strip_markup,
)
from backend.services.prosody.parser import parse_duration


def plan(markup: str, **kwargs) -> RenderPlan:
    kwargs.setdefault("engine", "qwen")
    kwargs.setdefault("default_language", "en")
    return compile_plan(markup, **kwargs)


def texts(p: RenderPlan) -> list[str]:
    return [n.text for n in p.nodes if isinstance(n, Speech)]


def codes(p: RenderPlan) -> list[str]:
    return [w.code for w in p.warnings]


# ── Prose is not XML ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["5 < 6 and 7 > 6", "Tom & Jerry", "a <notatag> b", "if x<y then", "100% > 50%"],
)
def test_prose_that_would_break_an_xml_parser_is_literal(text):
    """The parser recognises a closed tag set and leaves everything else alone,
    so ordinary prose needs no escaping."""
    p = plan(text)
    assert texts(p) == [text]


def test_unknown_tags_are_spoken_not_stripped():
    """Silently dropping an unknown tag would change the script. It is not ours,
    so it is text."""
    text = "Use the <blink> tag"
    assert texts(plan(text)) == [text]


# ── Directives ───────────────────────────────────────────────────────


def test_a_break_becomes_silence_not_a_generation():
    """No engine accepts a pause, so it is pure assembly — which is why pauses
    work identically on all eight."""
    p = plan('One.<break time="700ms"/>Two.')
    assert p.nodes[1] == Silence(700)
    assert texts(p) == ["One.", "Two."]


@pytest.mark.parametrize(
    ("raw", "expected"), [("700ms", 700), ("0.7s", 700), ("700", 700), ("1s", 1000), ("0ms", 0)]
)
def test_duration_forms(raw, expected):
    assert parse_duration(raw) == expected


@pytest.mark.parametrize("raw", ["", "soon", "-5ms", "999s", "700m"])
def test_bad_durations_are_rejected(raw):
    with pytest.raises(ProsodyParseError):
        parse_duration(raw)


def test_a_zero_break_emits_nothing():
    p = plan('One.<break time="0ms"/>Two.')
    assert not any(isinstance(n, Silence) for n in p.nodes)


def test_a_language_span_gets_its_own_run():
    p = plan('a <lang xml:lang="es">bandeja alta</lang> b')
    langs = [(n.text.strip(), n.language) for n in p.nodes if isinstance(n, Speech)]
    assert ("bandeja alta", "es") in langs
    assert all(lang == "en" for text, lang in langs if text != "bandeja alta")


def test_rate_applies_to_its_span_only():
    p = plan('normal <prosody rate="0.8">slow</prosody> normal')
    rates = {n.text.strip(): n.rate for n in p.nodes if isinstance(n, Speech)}
    assert rates["slow"] == 0.8
    assert rates["normal"] == 1.0


@pytest.mark.parametrize(("raw", "expected"), [("0.9", 0.9), ("90%", 0.9), ("1.5", 1.5)])
def test_rate_forms(raw, expected):
    p = plan(f'<prosody rate="{raw}">x</prosody>')
    assert p.nodes[0].rate == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["0", "-1", "fast", "20"])
def test_bad_rates_are_rejected(raw):
    """Rejected rather than clamped: rate 0 is audio of infinite length, and
    quietly substituting 1.0 hides a typo behind output that sounds fine."""
    with pytest.raises(ProsodyParseError):
        plan(f'<prosody rate="{raw}">x</prosody>')


def test_sub_replaces_what_the_engine_hears_but_records_the_original():
    p = plan('a <sub alias="ban-DEH-ha">bandeja</sub> b')
    run = next(n for n in p.nodes if isinstance(n, Speech) and n.source_text)
    assert run.text == "a ban-DEH-ha b", "the engine hears the respelling"
    assert run.source_text == "a bandeja b", "the original is kept for preview"


def test_a_respelling_does_not_cut_the_sentence():
    """The whole point of <sub>: it changes characters, not settings, so the
    sentence stays in one piece. Cutting there would buy exactly the seams that
    respelling exists to avoid."""
    p = plan('The shot he plays is a <sub alias="ban-DEH-ha">bandeja</sub>, not a smash.')
    assert len([n for n in p.nodes if isinstance(n, Speech)]) == 1


def test_phoneme_is_carried_as_a_substitution():
    p = plan('a <phoneme alphabet="ipa" ph="banˈdexa">bandeja</phoneme> b')  # noqa: RUF001
    run = next(n for n in p.nodes if isinstance(n, Speech) and n.source_text)
    assert "banˈdexa" in run.text  # noqa: RUF001
    assert run.source_text == "a bandeja b"


# ── Nesting and inheritance ──────────────────────────────────────────


def test_attributes_inherit_through_nesting():
    p = plan('<prosody rate="0.8">slow <lang xml:lang="es">lento</lang></prosody>')
    lento = next(n for n in p.nodes if isinstance(n, Speech) and n.language == "es")
    assert lento.rate == 0.8, "the inner span should keep the outer rate"


def test_the_inner_span_wins_on_conflict():
    p = plan('<lang xml:lang="es">a <lang xml:lang="it">b</lang></lang>')
    langs = {n.text.strip(): n.language for n in p.nodes if isinstance(n, Speech)}
    assert langs["b"] == "it"


def test_a_substitution_does_not_leak_to_siblings():
    """<sub> applies to the words it wraps. Inheriting it would put unrelated
    text through a substitution the author never asked for -- the run merges
    with its neighbours, but only the wrapped word is replaced."""
    p = plan('<prosody rate="0.9"><sub alias="X">a</sub> b</prosody>')
    run = next(n for n in p.nodes if isinstance(n, Speech))
    assert run.text == "X b", "only the wrapped word is substituted"
    assert run.source_text == "a b"


# ── Cutting as little as possible ────────────────────────────────────


def test_identical_neighbours_coalesce():
    """Every avoided cut is one less place the model restarts its prosody."""
    p = plan("one two three")
    assert len(texts(p)) == 1


def test_orphaned_punctuation_never_becomes_its_own_run():
    """A span boundary orphans its trailing punctuation — </lang>. leaves a run
    holding just ".". Generating that is a wasted call returning noise."""
    p = plan('a <lang xml:lang="es">bandeja</lang>. b')
    assert all(any(c.isalnum() for c in t) for t in texts(p)), texts(p)


def test_leading_punctuation_is_absorbed_forward():
    p = plan('<lang xml:lang="es">.</lang> hello there')
    assert all(any(c.isalnum() for c in t) for t in texts(p)), texts(p)


def test_all_punctuation_still_produces_something():
    """Absorbing must not be able to empty the plan."""
    p = plan("...")
    assert p.nodes


def test_unmarked_text_is_a_single_trivial_run():
    """The common case must cost nothing — the renderer takes the existing
    single-shot path for these."""
    p = plan("Just a plain sentence.")
    assert p.is_trivial


def test_a_language_span_is_not_trivial():
    assert not plan('a <lang xml:lang="es">b c d</lang>').is_trivial


# ── Engine capability ────────────────────────────────────────────────


def test_emphasis_reaches_an_engine_that_honours_instruct():
    p = plan("<emphasis level=\"strong\">wow</emphasis>", supports_instruct=True)
    assert "strong" in (p.nodes[0].instruct or "")
    assert not codes(p)


def test_emphasis_on_an_engine_that_ignores_instruct_warns():
    """Dropping it silently is what makes a model look like it is refusing to
    follow instructions (#579)."""
    p = plan("<emphasis>wow</emphasis>", supports_instruct=False)
    assert "emphasis_unsupported" in codes(p)
    assert p.nodes[0].instruct is None


def test_emphasis_composes_with_the_requests_own_instruct():
    p = plan(
        "<emphasis>wow</emphasis>", supports_instruct=True, base_instruct="Speak warmly."
    )
    assert "Speak warmly." in p.nodes[0].instruct
    assert "emphasis" in p.nodes[0].instruct


def test_an_unsupported_language_falls_back_and_says_so():
    p = plan('a <lang xml:lang="sw">x y z</lang>', engine_languages=["en", "es"])
    assert "language_unsupported" in codes(p)
    assert all(n.language == "en" for n in p.nodes if isinstance(n, Speech))


def test_each_unsupported_language_warns_once():
    p = plan(
        'a <lang xml:lang="sw">x y z</lang> b <lang xml:lang="sw">p q r</lang>',
        engine_languages=["en"],
    )
    assert codes(p).count("language_unsupported") == 1


def test_a_tight_single_word_span_is_not_flagged():
    """Listening across three voices found tight spans good, so warning against
    them would steer people away from what works."""
    assert not codes(plan('a <lang xml:lang="es">bandeja</lang> b'))


def test_a_clause_length_span_is_not_flagged():
    assert not codes(plan('a <lang xml:lang="es">bandeja, no un smash,</lang> b'))


@pytest.mark.parametrize("alias", ["bandeha", "ban-deh-ha", "W C A G", "Bandeha"])
def test_a_reasonable_respelling_is_not_flagged(alias):
    """Plain substitution, syllable hyphens alone, an acronym expansion, and a
    capitalised proper noun are all fine."""
    assert not codes(plan(f'a <sub alias="{alias}">x</sub> b'))


def test_hyphens_plus_capitals_are_flagged():
    """`ban-DEH-ha` was judged exaggerated on every voice and measured ~30%
    longer than the same sentence unmarked. It is the combination that
    misfires, not either alone."""
    assert "over_articulated_respelling" in codes(plan('a <sub alias="ban-DEH-ha">x</sub> b'))


def test_a_bare_break_is_a_good_default_pause():
    """700ms was judged right on every voice; 1500ms too long unless the script
    wants a beat to stop and think."""
    p = plan("one<break/>two")
    assert Silence(700) in p.nodes


# ── Malformed markup ─────────────────────────────────────────────────


def test_an_unclosed_span_is_an_error():
    """Not passed through as text: passing it through means the engine reads
    the tag aloud."""
    with pytest.raises(ProsodyParseError):
        parse('<lang xml:lang="es">oops')


def test_a_mismatched_close_is_an_error():
    with pytest.raises(ProsodyParseError):
        parse('<lang xml:lang="es">a</prosody>')


@pytest.mark.parametrize(
    "markup", ["<lang>x</lang>", '<prosody>x</prosody>', "<sub>x</sub>", "<phoneme>x</phoneme>"]
)
def test_a_span_missing_its_required_attribute_is_an_error(markup):
    with pytest.raises(ProsodyParseError):
        parse(markup)


# ── The invariant that makes LLM annotation safe ─────────────────────


def test_strip_markup_recovers_the_words():
    """Annotation is accepted only if stripping the model's output reproduces
    the input. The model can fail to help; it cannot mangle the script."""
    original = "The shot here is a bandeja. Not a smash."
    annotated = (
        'The shot here is a <lang xml:lang="es">bandeja</lang>.'
        '<break time="700ms"/> Not a smash.'
    )
    assert strip_markup(annotated) == strip_markup(original)


def test_strip_markup_detects_a_rewritten_script():
    original = "The shot here is a bandeja."
    tampered = 'The shot here is a <lang xml:lang="es">bandeja</lang>, obviously.'
    assert strip_markup(tampered) != strip_markup(original)


def test_strip_markup_ignores_whitespace_reflow():
    """Putting a tag on its own line is formatting, not content."""
    assert strip_markup("a\n<break time=\"1s\"/>\n b") == strip_markup("a b")


def test_has_markup_detects_directives():
    assert has_markup('a <break time="1s"/>')
    assert not has_markup("a plain sentence")
    assert not has_markup("5 < 6")


# ── Seeds ────────────────────────────────────────────────────────────


def test_seeds_vary_per_run_but_stay_deterministic():
    p = plan('a b<break time="1s"/><lang xml:lang="es">c d e</lang>').with_seeds(100)
    seeds = [n.seed for n in p.nodes if isinstance(n, Speech)]
    assert seeds == [100, 101]
    assert p.with_seeds(100) == plan(
        'a b<break time="1s"/><lang xml:lang="es">c d e</lang>'
    ).with_seeds(100)


def test_an_unseeded_plan_stays_unseeded():
    """Takes should still vary when no seed was requested."""
    p = plan("a b c").with_seeds(None)
    assert all(n.seed is None for n in p.nodes if isinstance(n, Speech))
