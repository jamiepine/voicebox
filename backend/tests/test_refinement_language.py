"""The refinement prompt must pin output language to the transcript's language.

Regression tests for #603: the refinement system prompt is written in English
with English-only few-shot examples, and without an explicit language
constraint small refinement models silently translate non-English transcripts
into English. These tests pin the prompt-level guardrails (no model needed).
"""

from backend.services.refinement import RefinementFlags, build_refinement_prompt


def test_known_language_pins_output_language():
    prompt = build_refinement_prompt(RefinementFlags(), language="zh")
    assert "Chinese" in prompt
    assert "MUST be written in Chinese" in prompt
    assert "Never translate the transcript" in prompt


def test_unknown_language_falls_back_to_same_language_rule():
    prompt = build_refinement_prompt(RefinementFlags())
    assert "same language as the transcript" in prompt
    assert "Never translate the transcript" in prompt


def test_language_rule_coexists_with_flag_sections():
    prompt = build_refinement_prompt(
        RefinementFlags(smart_cleanup=True, self_correction=True, preserve_technical=True),
        language="ko",
    )
    assert "Korean" in prompt
    # The behaviour sections are still present alongside the language rule.
    assert "disfluencies" in prompt.lower()
    assert "correction cue" in prompt.lower()
    assert "dot" in prompt  # preserve_technical symbol mapping survives


def test_unmapped_language_code_passes_through():
    prompt = build_refinement_prompt(RefinementFlags(), language="sv")
    assert "MUST be written in sv" in prompt


def test_language_code_is_case_insensitive():
    prompt = build_refinement_prompt(RefinementFlags(), language="ZH")
    assert "MUST be written in Chinese" in prompt
