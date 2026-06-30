"""Unit tests for ``collapse_repetitive_artifacts``.

The eval harness (``test_refinement_samples.py``) is interactive and
LLM-dependent; these are the fast, deterministic tests for the
deterministic pre-processor that runs before the LLM ever sees a
transcript. They pin the behaviour for both the single-word loops the
original algorithm handled and the multi-word / CJK / emoji loops the
character-level pass added.
"""

from backend.services.refinement import collapse_repetitive_artifacts


# ── single-word loops (word-level pass) ─────────────────────────────────


def test_single_word_loop_stripped():
    raw = "Hello " + ("URL " * 8).strip() + " goodbye"
    assert collapse_repetitive_artifacts(raw) == "Hello goodbye"


def test_single_word_loop_with_punctuation_normalized():
    # URL, URL, URL, URL, URL, URL. — six repeats if you normalize
    # trailing punctuation; word-level pass strips them all.
    raw = "Hello URL, URL, URL, URL, URL, URL. goodbye"
    assert collapse_repetitive_artifacts(raw) == "Hello goodbye"


def test_single_word_loop_case_insensitive():
    raw = "hi " + " ".join(["Url", "URL", "url", "Url", "URL", "url"]) + " bye"
    assert collapse_repetitive_artifacts(raw) == "hi bye"


def test_short_single_word_run_preserved():
    # Five repeats — below threshold.
    raw = "no no no no no"
    assert collapse_repetitive_artifacts(raw) == raw


def test_rhetorical_repetition_preserved():
    raw = "I said no, no, no, no, no and she left"
    # Five repeats of "no" — below threshold.
    assert collapse_repetitive_artifacts(raw) == raw


# ── multi-word loops (character-level pass) ─────────────────────────────


def test_multi_word_english_loop_stripped():
    # Classic Whisper tail hallucination. Word-level pass sees no
    # consecutive identical tokens, so it's the character-level pass's
    # job to catch this.
    loop = "thanks for watching " * 6
    raw = f"Okay so the meeting is at three. {loop}"
    result = collapse_repetitive_artifacts(raw)
    assert "thanks for watching" not in result
    assert "Okay so the meeting is at three" in result


def test_three_word_loop_stripped():
    loop = "please like and " * 7
    raw = f"The point is clear. {loop}right"
    result = collapse_repetitive_artifacts(raw)
    assert "please like and" not in result
    assert "The point is clear" in result


def test_long_phrase_loop_within_60_char_cap():
    unit = "Please like and subscribe to my channel. "  # 41 chars, within cap
    raw = "End of video. " + unit * 6
    result = collapse_repetitive_artifacts(raw)
    assert unit.strip() not in result
    assert "End of video" in result


def test_multi_word_short_run_preserved():
    # Five repeats of a multi-word unit — below threshold.
    raw = "thanks for watching thanks for watching thanks for watching thanks for watching thanks for watching"
    assert collapse_repetitive_artifacts(raw) == raw


# ── CJK loops (character-level pass, no whitespace) ──────────────────────


def test_cjk_loop_stripped():
    # Common Chinese Whisper hallucination: "thanks for watching".
    # text.split() yields one token for the whole loop; only the
    # character-level pass can catch this.
    prefix = "會議在三點開始"
    loop = "謝謝觀看" * 7
    raw = prefix + loop
    result = collapse_repetitive_artifacts(raw)
    assert "謝謝觀看" not in result
    assert prefix in result


def test_japanese_loop_stripped():
    # Same pattern, kana/kanji mix. "ご視聴ありがとうございました" is a
    # frequent Japanese Whisper tail hallucination.
    loop = "ご視聴ありがとうございました" * 6
    raw = f"明日の会議は午後三時です。{loop}"
    result = collapse_repetitive_artifacts(raw)
    assert "ご視聴ありがとうございました" not in result
    assert "明日の会議は午後三時です" in result


def test_cjk_short_run_preserved():
    # Five repeats — below threshold, stays in.
    raw = "好好好好好"
    assert collapse_repetitive_artifacts(raw) == raw


# ── whitespace / edge cases ──────────────────────────────────────────────


def test_empty_string_passes_through():
    assert collapse_repetitive_artifacts("") == ""


def test_below_word_threshold_passes_through_unmodified():
    raw = "just three words"
    assert collapse_repetitive_artifacts(raw) == raw


def test_emphasis_vowel_run_preserved():
    # "wooooooow" is 1 char (plus 8 o's). Character-level min unit is 2,
    # so "oo…" doesn't get stripped and this legitimate emphasis stays.
    raw = "that's wooooooow amazing"
    assert collapse_repetitive_artifacts(raw) == raw


def test_custom_threshold_honored():
    # With min_run=3, even short rhetorical repetition should now strip.
    raw = "ha ha ha ha context"
    result = collapse_repetitive_artifacts(raw, min_run=3)
    assert "ha ha" not in result
    assert "context" in result


def test_leading_and_trailing_whitespace_stripped_after_collapse():
    # When character pass fires, the normalised result is stripped so
    # downstream prompts don't carry edge whitespace from the removal.
    loop = "loop-phrase " * 7
    raw = loop
    assert collapse_repetitive_artifacts(raw) == ""
