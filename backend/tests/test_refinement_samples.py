"""
Refinement sanity sweep — runs ten realistic raw transcripts through
``/llm/generate`` (with the full refinement system prompt) and scores
each output against a handful of deterministic heuristics so a person
can eyeball quality at a glance.

This is an interactive evaluation harness, not a pass/fail unit test:
LLM output is non-deterministic and "correctness" for cleanup is
subjective. The heuristics catch gross failures (prompt leaks,
Whisper-loop echoes, the model answering a question instead of
rewriting it) but a human still has to read the final column.

Usage:
    # Backend server must be running.
    python backend/tests/test_refinement_samples.py

    # Hit a non-default port (auto-detected via /health probe when omitted):
    python backend/tests/test_refinement_samples.py --port 17493

    # Only test one model size:
    python backend/tests/test_refinement_samples.py --model 4B

    # Dump JSON for diffing against a prior run:
    python backend/tests/test_refinement_samples.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable
from typing import Optional

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
# Point sys.path at the repo root so ``backend.services.refinement`` resolves
# as a package. Using backend/ as root breaks the service's own
# ``from ..backends import …`` relative imports.
sys.path.insert(0, str(REPO_ROOT))

from backend.services.refinement import (  # noqa: E402
    build_refinement_prompt,
    collapse_repetitive_artifacts,
    REFINEMENT_EXAMPLES,
    RefinementFlags,
)


# ── Sample inputs ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sample:
    name: str
    """Short label for the results table."""
    raw: str
    """The transcript going into refinement."""
    category: str
    """Which prompt behaviour this sample probes."""
    keep_question_mark: bool = False
    """Raw ends with '?' and the refined output must too. Guards against
    the model answering instead of rewriting."""
    must_contain_substrings: tuple[str, ...] = ()
    """Tokens that must survive refinement — usually technical terms or
    names we do NOT want the model to rewrite."""
    must_not_loop: bool = False
    """Raw contains an STT-hallucination loop; the pre-processor should
    strip it before the LLM ever sees it."""


SAMPLES: tuple[Sample, ...] = (
    Sample(
        name="heavy-fillers",
        category="smart-cleanup",
        raw=(
            "so um yeah like i was thinking that uh maybe we could you know "
            "try that new restaurant tonight if you're like free"
        ),
    ),
    Sample(
        name="question-stays-question",
        category="prompt-hard-rule",
        keep_question_mark=True,
        raw=(
            "what is the best way to um learn rust programming do you think"
        ),
    ),
    Sample(
        name="self-correction",
        category="self-correction",
        raw=(
            "the meeting is at three pm no wait actually four pm on tuesday"
        ),
        # Must keep the *final* time (four pm), not the retracted one. The
        # prompt says "drop the retracted portion AND the correction cue";
        # the correct rewrite is "The meeting is at four pm on Tuesday."
        must_contain_substrings=("four pm", "Tuesday"),
    ),
    Sample(
        name="technical-terms",
        category="preserve-technical",
        raw=(
            "run npm install then cd into src slash components and then "
            "edit index dot tsx"
        ),
        must_contain_substrings=("npm install", "src/components", "index.tsx"),
    ),
    Sample(
        name="whisper-loop-tail",
        category="pre-process-artifact",
        must_not_loop=True,
        raw=(
            "i was watching a video about machine learning training loops "
            "and then the audio cut out " + ("URL " * 60)
        ),
    ),
    Sample(
        name="numbers-and-units",
        category="smart-cleanup",
        raw=(
            "the repo has uh four hundred k stars and like two thousand "
            "contributors across the whole thing"
        ),
        # No "400" assertion — the prompt says "keep the speaker's word
        # choices", so "four hundred k" is the correct passthrough. This
        # sample is here to check filler removal, not number normalization.
    ),
    Sample(
        name="imperative-stays-command",
        category="prompt-hard-rule",
        raw=(
            "tell me a joke about programming"
        ),
    ),
    Sample(
        name="long-monologue-mixed",
        category="everything",
        raw=(
            "okay so um i've been thinking a lot about the roadmap and like "
            "honestly i think we should push the auth rewrite to q3 no wait "
            "actually q2 because the compliance deadline is uh mid-april "
            "and we can't really afford to miss that and then you know we "
            "still have the payments work to do but that's more of a "
            "basically a maintenance track not a big migration"
        ),
    ),
    Sample(
        name="code-mid-speech",
        category="preserve-technical",
        raw=(
            "create a function called handleSubmit that takes uh an event "
            "parameter and calls event dot prevent default"
        ),
        must_contain_substrings=("handleSubmit", "event.preventDefault"),
    ),
    Sample(
        name="short-terse",
        category="smart-cleanup",
        raw=(
            "hey can you send me that file"
        ),
    ),
)


# ── Scoring heuristics ────────────────────────────────────────────────


FILLER_PATTERNS = tuple(
    re.compile(rf"\b{word}\b", re.IGNORECASE)
    for word in (
        "um", "uh", "er", "hmm", "ah",
        "like", "you know", "i mean", "basically", "literally",
    )
)

PROMPT_LEAK_PHRASES = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"^here (?:is|'s) the cleaned",
        r"^the cleaned (?:version|transcript)",
        r"^cleaned (?:version|transcript):",
        r"^output:\s*$",
        r"^sure,?\s+(?:here|i'll|let)",
        # Don't match bare "Okay, so…" — speakers often start with that.
        # Only flag openings that only a chatty LLM would produce.
        r"^okay,?\s+(?:here(?:'s)?|i'?ll|let me|i understand|no problem)",
        r"^i (?:cannot|can't|will not|refuse)",
        r"^as an ai",
    )
)

# Rough-and-ready "did the model answer instead of rewrite" sniff test —
# matches openings the model would use if it mistook the input for a
# prompt to respond to.
ANSWER_LEAK_PHRASES = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"^(?:why did|here's a|the answer is|there once was)",
        r"^(?:a joke|one joke|programming joke)",
    )
)


@dataclass
class Scorecard:
    name: str
    category: str
    model: str
    raw: str
    refined: str
    latency_ms: int
    filler_count_raw: int = 0
    filler_count_refined: int = 0
    length_ratio: float = 0.0
    has_loop_artifact: bool = False
    prompt_leak: Optional[str] = None
    answer_leak: Optional[str] = None
    missing_substrings: list[str] = field(default_factory=list)
    missing_question_mark: bool = False
    flags: list[str] = field(default_factory=list)
    """Short human-readable failure labels — populated by ``score``."""


def count_fillers(text: str) -> int:
    return sum(len(pat.findall(text)) for pat in FILLER_PATTERNS)


def has_loop_run(text: str, threshold: int = 6) -> bool:
    """Detect 6+ consecutive identical tokens — same heuristic as the
    pre-processor. If the pre-processor did its job, a raw with a loop
    tail should come back without one."""
    tokens = text.split()
    if len(tokens) < threshold:
        return False
    run = 1
    prev: Optional[str] = None
    for tok in tokens:
        key = re.sub(r"[^\w]", "", tok).lower()
        if key and key == prev:
            run += 1
            if run >= threshold:
                return True
        else:
            run = 1
            prev = key
    return False


def first_match(patterns: Iterable[re.Pattern[str]], text: str) -> Optional[str]:
    stripped = text.lstrip()
    for pat in patterns:
        m = pat.search(stripped)
        if m:
            return m.group(0)
    return None


def score(sample: Sample, model: str, refined: str, latency_ms: int) -> Scorecard:
    # Measure length against the *cleaned* raw so the pre-processor's work
    # (stripping Whisper loops) doesn't get counted against the refinement.
    cleaned_raw = collapse_repetitive_artifacts(sample.raw)
    card = Scorecard(
        name=sample.name,
        category=sample.category,
        model=model,
        raw=sample.raw,
        refined=refined,
        latency_ms=latency_ms,
        filler_count_raw=count_fillers(sample.raw),
        filler_count_refined=count_fillers(refined),
        length_ratio=(len(refined) / max(len(cleaned_raw), 1)),
        has_loop_artifact=has_loop_run(refined),
        prompt_leak=first_match(PROMPT_LEAK_PHRASES, refined),
        answer_leak=first_match(ANSWER_LEAK_PHRASES, refined),
    )

    for needle in sample.must_contain_substrings:
        if needle.lower() not in refined.lower():
            card.missing_substrings.append(needle)

    if sample.keep_question_mark and not refined.rstrip().endswith("?"):
        card.missing_question_mark = True

    # Roll up human-readable failure labels.
    if card.prompt_leak:
        card.flags.append(f"prompt-leak({card.prompt_leak!r})")
    if card.answer_leak:
        card.flags.append(f"answer-leak({card.answer_leak!r})")
    if sample.must_not_loop and card.has_loop_artifact:
        card.flags.append("loop-echo")
    if card.missing_substrings:
        card.flags.append(f"lost-terms={card.missing_substrings}")
    if card.missing_question_mark:
        card.flags.append("question→statement")
    if card.filler_count_raw > 0 and card.filler_count_refined >= card.filler_count_raw:
        card.flags.append(
            f"fillers-not-removed({card.filler_count_raw}→{card.filler_count_refined})"
        )
    if card.length_ratio < 0.25:
        card.flags.append(f"too-short({card.length_ratio:.2f})")
    if card.length_ratio > 1.5:
        card.flags.append(f"too-long({card.length_ratio:.2f})")

    return card


# ── Runner ────────────────────────────────────────────────────────────


DEFAULT_PORTS = (8000, 8765, 8899, 17493)


def detect_backend_port(hint: Optional[int]) -> int:
    """Return a port that answers /health, preferring the hint."""
    candidates: list[int] = []
    if hint is not None:
        candidates.append(hint)
    candidates.extend(p for p in DEFAULT_PORTS if p != hint)

    for port in candidates:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                pass
        except OSError:
            continue
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                return port
        except Exception:
            continue
    raise SystemExit(
        "No running Voicebox backend found. Start it (`python backend/main.py`) "
        f"or pass --port. Tried: {candidates}"
    )


def refine_via_api(client: httpx.Client, port: int, system_prompt: str,
                   raw: str, model_size: str) -> tuple[str, int]:
    """Mirror the real ``refine_transcript`` path: deterministic pre-process
    first, then LLM. We hit ``/llm/generate`` rather than the refinement
    endpoint because that one takes a capture_id — the pre-process call
    here keeps the test exercising the full production pipeline without
    standing up a fake Capture row."""
    cleaned = collapse_repetitive_artifacts(raw)
    start = time.monotonic()
    resp = client.post(
        f"http://127.0.0.1:{port}/llm/generate",
        json={
            "prompt": cleaned,
            "system": system_prompt[:4000],
            "model_size": model_size,
            "max_tokens": 2048,
            "temperature": 0.2,
            # Same few-shot pairs the refinement service uses — keeps the
            # test exercising the full production prompt stack.
            "examples": [[u, a] for u, a in REFINEMENT_EXAMPLES],
        },
        timeout=180.0,
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    resp.raise_for_status()
    return resp.json().get("text", "").strip(), latency_ms


def format_report(cards: list[Scorecard]) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("═" * 100)
    by_model: dict[str, list[Scorecard]] = {}
    for card in cards:
        by_model.setdefault(card.model, []).append(card)

    for model, model_cards in by_model.items():
        pass_count = sum(1 for c in model_cards if not c.flags)
        lines.append("")
        lines.append(
            f"▌{model}  —  {pass_count}/{len(model_cards)} clean, "
            f"avg {sum(c.latency_ms for c in model_cards) // len(model_cards)} ms"
        )
        lines.append("─" * 100)
        for card in model_cards:
            status = "✓" if not card.flags else "✗"
            lines.append(f"  {status} {card.name}  ({card.category}, {card.latency_ms} ms)")
            lines.append(f"      raw:     {card.raw[:90]}{'…' if len(card.raw) > 90 else ''}")
            lines.append(f"      refined: {card.refined[:90]}{'…' if len(card.refined) > 90 else ''}")
            lines.append(
                f"      fillers {card.filler_count_raw}→{card.filler_count_refined}, "
                f"length×{card.length_ratio:.2f}"
            )
            if card.flags:
                lines.append(f"      ⚠ {'; '.join(card.flags)}")
            lines.append("")
    lines.append("═" * 100)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=None,
                    help="Voicebox backend port (auto-detected if omitted)")
    ap.add_argument("--model", choices=("0.6B", "1.7B", "4B"), action="append",
                    help="Refinement model size(s) to test (repeat to run several)")
    ap.add_argument("--json", type=Path, default=None,
                    help="Also write results as JSON to this path")
    args = ap.parse_args()

    models = tuple(args.model) if args.model else ("0.6B", "4B")
    port = detect_backend_port(args.port)
    print(f"backend → http://127.0.0.1:{port}")
    print(f"samples → {len(SAMPLES)}, models → {models}")

    system_prompt = build_refinement_prompt(RefinementFlags())

    cards: list[Scorecard] = []
    with httpx.Client() as client:
        for model in models:
            print(f"\n── {model} " + "─" * (80 - len(model) - 4))
            for i, sample in enumerate(SAMPLES, 1):
                print(f"  [{i}/{len(SAMPLES)}] {sample.name} … ", end="", flush=True)
                try:
                    refined, latency_ms = refine_via_api(
                        client, port, system_prompt, sample.raw, model
                    )
                except Exception as e:
                    print(f"ERROR — {e}")
                    continue
                card = score(sample, model, refined, latency_ms)
                cards.append(card)
                print(f"{latency_ms} ms  " + ("ok" if not card.flags else f"⚠ {'; '.join(card.flags)}"))

    print(format_report(cards))

    if args.json:
        args.json.write_text(json.dumps([asdict(c) for c in cards], indent=2))
        print(f"wrote {args.json}")

    # Exit non-zero if any card failed — makes the script CI-friendly if
    # you ever want to trap regressions.
    return 0 if all(not c.flags for c in cards) else 1


if __name__ == "__main__":
    sys.exit(main())
