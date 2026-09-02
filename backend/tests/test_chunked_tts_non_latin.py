"""Regression coverage for non-Latin long-text chunking."""

import ast
from pathlib import Path

import numpy as np
import pytest

from backend.utils import chunked_tts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATIONS_ROUTE = PROJECT_ROOT / "backend" / "routes" / "generations.py"
GENERATION_SERVICE = PROJECT_ROOT / "backend" / "services" / "generation.py"
ENGINE_AWARE_GENERATOR = "generate_chunked_for_engine"
GENERATION_SERVICE_CALL_COUNT = 2
SAMPLE_RATE = 1000
CHATTERBOX_ENGINE = "chatterbox"
QWEN_ENGINE = "qwen"
TEST_UTF8_ENCODING = "utf-8"
MISSING_CHUNK_MEASURE_MESSAGE = "Chatterbox should measure chunks with UTF-8 bytes"
HINDI_SENTENCE_REPETITIONS = 7
EXPECTED_MIN_CHUNKS = 2
ARABIC_PREFIX_LENGTH = 10
ARABIC_BOUNDARY_EXTRA_CHARS = 1
SINGLE_BYTE_BUDGET = 1
HARD_CUT_BUDGET = 5
TAG_STRADDLE_BUDGET = 10
WHITESPACE_OVERHEAD = 2
BYTE_RETRY_TEXT_CHARS = 44
EMPTY_TEXT_ERROR = "Cannot generate TTS audio from empty text"
DEFAULT_MAX_CHUNK_CHARS = chunked_tts.DEFAULT_MAX_CHUNK_CHARS
HINDI_SENTENCE = (
    "\u092d\u093e\u0930\u0924 \u092e\u0947\u0902 \u0935\u0930\u094d\u0937\u093e "
    "\u090b\u0924\u0941 \u0915\u093e \u0906\u0917\u092e\u0928 \u091c\u0942\u0928 "
    "\u0915\u0947 \u092e\u0939\u0940\u0928\u0947 \u092e\u0947\u0902 "
    "\u0939\u094b\u0924\u093e \u0939\u0948\u0964"
)
HINDI_TEXT = " ".join([HINDI_SENTENCE] * HINDI_SENTENCE_REPETITIONS)
ARABIC_ALEF = "\u0627"
ARABIC_BEH = "\u0628"
ARABIC_FULL_STOP = "\u06d4"
ARABIC_QUESTION_MARK = "\u061f"
BRACKETED_ARABIC_FULL_STOP = f"[{ARABIC_FULL_STOP}]"
ASCII_HARD_CUT_TEXT = "abcdefghijklmnopqrstuvwxyz"
HINDI_NO_BOUNDARY_TEXT = HINDI_SENTENCE[0] * DEFAULT_MAX_CHUNK_CHARS
TAG_STRADDLE_TEXT = "prefix[noise]suffix"
BOUNDARY_TAG_TEXT = "[noise]suffix"
UNMATCHED_TAG_TEXT = "prefix[noise suffix"
UNMATCHED_TAG_PAST_BUDGET_TEXT = "abcdefghij[unfinished"


def _called_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
    return calls


def _utf8_byte_length(text: str) -> int:
    return len(text.encode(TEST_UTF8_ENCODING))


def test_generation_entrypoints_use_engine_aware_chunking():
    route_calls = _called_functions(GENERATIONS_ROUTE)
    service_calls = _called_functions(GENERATION_SERVICE)

    assert ENGINE_AWARE_GENERATOR in route_calls
    assert service_calls.count(ENGINE_AWARE_GENERATOR) == GENERATION_SERVICE_CALL_COUNT


def test_chatterbox_uses_utf8_byte_budget_for_non_latin_text():
    measure = getattr(chunked_tts, "chunk_measure_for_engine", None)

    assert measure is not None, MISSING_CHUNK_MEASURE_MESSAGE
    assert measure(CHATTERBOX_ENGINE) is chunked_tts.utf8_byte_length
    assert len(HINDI_TEXT) < DEFAULT_MAX_CHUNK_CHARS
    assert _utf8_byte_length(HINDI_TEXT) > DEFAULT_MAX_CHUNK_CHARS

    chunks = chunked_tts.split_text_into_chunks(
        HINDI_TEXT,
        DEFAULT_MAX_CHUNK_CHARS,
        measure=measure(CHATTERBOX_ENGINE),
    )

    assert len(chunks) >= EXPECTED_MIN_CHUNKS
    assert all(_utf8_byte_length(chunk) <= DEFAULT_MAX_CHUNK_CHARS for chunk in chunks)
    assert all(chunk.endswith("\u0964") for chunk in chunks[:-1])


def test_non_chatterbox_engines_keep_character_budget():
    assert chunked_tts.chunk_measure_for_engine(QWEN_ENGINE) is len


def test_splitter_keeps_progress_when_one_character_exceeds_byte_budget():
    chunks = chunked_tts.split_text_into_chunks(
        HINDI_SENTENCE,
        SINGLE_BYTE_BUDGET,
        measure=chunked_tts.utf8_byte_length,
    )

    assert chunks[0] == HINDI_SENTENCE[0]


def test_default_character_budget_hard_cuts_without_custom_measure():
    chunks = chunked_tts.split_text_into_chunks(ASCII_HARD_CUT_TEXT, HARD_CUT_BUDGET)

    assert chunks[0] == ASCII_HARD_CUT_TEXT[:HARD_CUT_BUDGET]


def test_byte_budget_hard_cut_stays_within_measured_segment():
    chunks = chunked_tts.split_text_into_chunks(
        HINDI_NO_BOUNDARY_TEXT,
        DEFAULT_MAX_CHUNK_CHARS,
        measure=chunked_tts.utf8_byte_length,
    )

    assert all(_utf8_byte_length(chunk) <= DEFAULT_MAX_CHUNK_CHARS for chunk in chunks)


def test_hard_cut_does_not_split_tag_straddling_budget():
    chunks = chunked_tts.split_text_into_chunks(TAG_STRADDLE_TEXT, TAG_STRADDLE_BUDGET)

    assert chunks[0] == "prefix"
    assert chunks[1].startswith("[noise]")


def test_hard_cut_keeps_boundary_starting_tag_atomic():
    chunks = chunked_tts.split_text_into_chunks(BOUNDARY_TAG_TEXT, HARD_CUT_BUDGET)

    assert chunks[0] == "[noise]"
    assert chunks[1] == "suffi"


def test_hard_cut_avoids_unmatched_tag_prefix():
    chunks = chunked_tts.split_text_into_chunks(UNMATCHED_TAG_TEXT, TAG_STRADDLE_BUDGET)

    assert chunks[0] == "prefix"


def test_unmatched_tag_after_budget_does_not_expand_chunk():
    chunks = chunked_tts.split_text_into_chunks(UNMATCHED_TAG_PAST_BUDGET_TEXT, HARD_CUT_BUDGET)

    assert chunks[0] == UNMATCHED_TAG_PAST_BUDGET_TEXT[:HARD_CUT_BUDGET]


def test_arabic_full_stop_is_a_sentence_boundary():
    text = (
        f"{ARABIC_ALEF * ARABIC_PREFIX_LENGTH}{ARABIC_FULL_STOP}"
        f"{ARABIC_BEH * DEFAULT_MAX_CHUNK_CHARS}{ARABIC_FULL_STOP}"
    )

    chunks = chunked_tts.split_text_into_chunks(
        text,
        ARABIC_PREFIX_LENGTH + len(ARABIC_FULL_STOP) + ARABIC_BOUNDARY_EXTRA_CHARS,
    )

    assert chunks[0].endswith(ARABIC_FULL_STOP)


def test_arabic_question_mark_is_a_sentence_boundary():
    text = (
        f"{ARABIC_ALEF * ARABIC_PREFIX_LENGTH}{ARABIC_QUESTION_MARK}"
        f"{ARABIC_BEH * DEFAULT_MAX_CHUNK_CHARS}{ARABIC_QUESTION_MARK}"
    )

    chunks = chunked_tts.split_text_into_chunks(
        text,
        ARABIC_PREFIX_LENGTH + len(ARABIC_QUESTION_MARK) + ARABIC_BOUNDARY_EXTRA_CHARS,
    )

    assert chunks[0].endswith(ARABIC_QUESTION_MARK)


def test_arabic_full_stop_inside_bracket_tag_is_not_a_boundary():
    first_sentence = (
        f"{BRACKETED_ARABIC_FULL_STOP}{ARABIC_ALEF * ARABIC_PREFIX_LENGTH}"
        f"{ARABIC_FULL_STOP}"
    )
    text = f"{first_sentence}{ARABIC_BEH * DEFAULT_MAX_CHUNK_CHARS}"

    chunks = chunked_tts.split_text_into_chunks(
        text,
        len(first_sentence) + ARABIC_BOUNDARY_EXTRA_CHARS,
    )

    assert chunks[0] == first_sentence


@pytest.mark.asyncio
async def test_generate_chunked_for_engine_uses_chatterbox_byte_budget():
    class FakeBackend:
        def __init__(self):
            self.calls: list[str] = []

        async def generate(self, text, *_args):
            self.calls.append(text)
            return np.full(SAMPLE_RATE, 0.2, dtype=np.float32), SAMPLE_RATE

    backend = FakeBackend()

    audio, sample_rate = await chunked_tts.generate_chunked_for_engine(
        backend,
        HINDI_TEXT,
        {},
        engine=CHATTERBOX_ENGINE,
        max_chunk_chars=DEFAULT_MAX_CHUNK_CHARS,
        crossfade_ms=0,
    )

    assert sample_rate == SAMPLE_RATE
    assert len(audio) == (len(backend.calls) * SAMPLE_RATE)
    assert len(backend.calls) >= EXPECTED_MIN_CHUNKS
    assert all(_utf8_byte_length(chunk) <= DEFAULT_MAX_CHUNK_CHARS for chunk in backend.calls)


@pytest.mark.asyncio
async def test_generate_chunked_fast_path_uses_measured_chunk_text():
    class FakeBackend:
        def __init__(self):
            self.calls: list[str] = []

        async def generate(self, text, *_args):
            self.calls.append(text)
            return np.full(SAMPLE_RATE, 0.2, dtype=np.float32), SAMPLE_RATE

    backend = FakeBackend()
    text = f" {HINDI_NO_BOUNDARY_TEXT[: DEFAULT_MAX_CHUNK_CHARS // 3]} "
    max_chunk_bytes = _utf8_byte_length(text.strip())

    await chunked_tts.generate_chunked(
        backend,
        text,
        {},
        max_chunk_chars=max_chunk_bytes,
        crossfade_ms=0,
        chunk_measure=chunked_tts.utf8_byte_length,
    )

    assert backend.calls == [text.strip()]
    assert _utf8_byte_length(backend.calls[0]) <= max_chunk_bytes
    assert _utf8_byte_length(text) == max_chunk_bytes + WHITESPACE_OVERHEAD


@pytest.mark.asyncio
async def test_generate_chunked_retries_use_configured_measure():
    class FakeBackend:
        def __init__(self):
            self.calls: list[str] = []

        async def generate(self, text, *_args):
            self.calls.append(text)
            return np.full(SAMPLE_RATE, 0.2, dtype=np.float32), SAMPLE_RATE

    backend = FakeBackend()
    text = HINDI_NO_BOUNDARY_TEXT[:BYTE_RETRY_TEXT_CHARS]

    def is_runaway(_audio, _sample_rate):
        return _utf8_byte_length(backend.calls[-1]) > chunked_tts.MIN_RUNAWAY_RETRY_CHARS

    await chunked_tts.generate_chunked(
        backend,
        text,
        {},
        max_chunk_chars=DEFAULT_MAX_CHUNK_CHARS,
        crossfade_ms=0,
        runaway_detector=is_runaway,
        chunk_measure=chunked_tts.utf8_byte_length,
    )

    assert len(backend.calls) > 1
    assert _utf8_byte_length(backend.calls[0]) > chunked_tts.MIN_RUNAWAY_RETRY_CHARS
    assert all(
        _utf8_byte_length(chunk) <= chunked_tts.MIN_RUNAWAY_RETRY_CHARS
        for chunk in backend.calls[1:]
    )


@pytest.mark.asyncio
async def test_generate_chunked_rejects_whitespace_only_text():
    class FakeBackend:
        async def generate(self, *_args):
            raise AssertionError("empty text should not reach backend")

    with pytest.raises(RuntimeError, match=EMPTY_TEXT_ERROR):
        await chunked_tts.generate_chunked(FakeBackend(), "   ", {})
