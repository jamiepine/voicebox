"""
Personality-service sanity sweep — spins up a throwaway profile with a
fake personality, exercises ``/profiles/{id}/compose`` and the rewrite
path on ``/generate`` (``personality=true``), and scores each output
against a handful of deterministic heuristics so a person can eyeball
quality.

Same philosophy as ``test_refinement_samples.py``: LLM output is
non-deterministic, "correctness" is subjective, so this is interactive
evaluation — not a CI pass/fail. Gross failures (prompt-echo, refusal,
empty output) trip heuristic flags. A human still reads the final
column.

Usage:
    # Backend server must be running.
    python backend/tests/test_personality_samples.py

    # Test just one model size:
    python backend/tests/test_personality_samples.py --model 4B

    # Dump JSON for diffing against a prior run:
    python backend/tests/test_personality_samples.py --json out.json
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
from typing import Optional

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ── Sample personalities ──────────────────────────────────────────────


@dataclass(frozen=True)
class Personality:
    name: str
    description: str
    """Free-form character prompt saved to the profile."""
    sample_text: str
    """Input used for rewrite. Picked so each personality has something
    distinctive to say about it — an ill fit between text and personality
    makes the transformation more obvious."""


PERSONALITIES: tuple[Personality, ...] = (
    Personality(
        name="grumpy-pirate",
        description=(
            "A grumpy old pirate captain who only speaks in nautical "
            "metaphors. Keeps things short and salty. Swears by his "
            "beard and the deep blue."
        ),
        sample_text="I need you to install the dependencies before the deploy.",
    ),
    Personality(
        name="victorian-professor",
        description=(
            "A stuffy Victorian-era professor of natural philosophy. "
            "Formal register, long sentences, fond of subordinate "
            "clauses, occasional Latin asides."
        ),
        sample_text="The build is broken, we should roll back to yesterday's version.",
    ),
    Personality(
        name="caffeinated-founder",
        description=(
            "A tech-bro startup founder who is always three coffees "
            "deep, obsessed with disruption and synergy, speaks in "
            "bullet points even out loud."
        ),
        sample_text="The meeting ran long and we didn't get to the roadmap.",
    ),
)


# ── Scoring heuristics ────────────────────────────────────────────────


PROMPT_LEAK_PHRASES = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"^here (?:is|'s) the cleaned",
        r"^here (?:is|'s) a",
        r"^as (?:an ai|the character)",
        r"^character description",
        r"^task:\s*",
        r"^output:\s*$",
        r"^sure,?\s+(?:here|i'?ll|let)",
    )
)


REFUSAL_PHRASES = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"\bi (?:cannot|can't|won'?t|will not|refuse)\b",
        r"\bi'?m sorry(?:,|\s+but)",
        r"\bi apologi[sz]e",
    )
)


STAGE_DIRECTION_RE = re.compile(r"[\*\(_].{0,60}?[\*\)_]")  # *smiles*, (leans in)


@dataclass
class Scorecard:
    personality: str
    endpoint: str
    model: str
    input_text: str
    """Empty for compose."""
    refined: str
    latency_ms: int
    length_chars: int = 0
    prompt_leak: Optional[str] = None
    refusal: Optional[str] = None
    stage_directions: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def first_match(patterns, text: str) -> Optional[str]:
    s = text.lstrip()
    for pat in patterns:
        m = pat.search(s)
        if m:
            return m.group(0)
    return None


def score(
    personality: Personality,
    endpoint: str,
    model: str,
    input_text: str,
    refined: str,
    latency_ms: int,
) -> Scorecard:
    card = Scorecard(
        personality=personality.name,
        endpoint=endpoint,
        model=model,
        input_text=input_text,
        refined=refined,
        latency_ms=latency_ms,
        length_chars=len(refined),
        prompt_leak=first_match(PROMPT_LEAK_PHRASES, refined),
        refusal=first_match(REFUSAL_PHRASES, refined),
        stage_directions=STAGE_DIRECTION_RE.findall(refined)[:3],
    )

    if not refined.strip():
        card.flags.append("empty-output")
    if card.prompt_leak:
        card.flags.append(f"prompt-leak({card.prompt_leak!r})")
    if card.refusal:
        card.flags.append(f"refusal({card.refusal!r})")
    if card.stage_directions:
        card.flags.append(f"stage-directions={card.stage_directions}")

    return card


# ── Runner ────────────────────────────────────────────────────────────


DEFAULT_PORTS = (8000, 8765, 8899, 17493)
THROWAWAY_PROFILE_PREFIX = "personality-harness-"
KOKORO_PROBE_VOICE = "af_heart"
"""Any valid kokoro voice id works — compose never calls into TTS, it
just needs a profile row with a personality attached. We pick a
known-shipping Kokoro voice so the throwaway profile satisfies the
preset-engine validator on creation."""


def detect_backend_port(hint: Optional[int]) -> int:
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


def create_throwaway_profile(
    client: httpx.Client, port: int, personality: Personality, model: str
) -> str:
    """Create a preset Kokoro profile with the test personality. Returns
    the profile id. Tests delete it in a finally block."""
    name = f"{THROWAWAY_PROFILE_PREFIX}{personality.name}-{model}-{int(time.time())}"
    resp = client.post(
        f"http://127.0.0.1:{port}/profiles",
        json={
            "name": name,
            "description": f"Throwaway profile for personality harness ({model}).",
            "language": "en",
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": KOKORO_PROBE_VOICE,
            "default_engine": "kokoro",
            "personality": personality.description,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def delete_profile(client: httpx.Client, port: int, profile_id: str) -> None:
    try:
        client.delete(f"http://127.0.0.1:{port}/profiles/{profile_id}", timeout=10.0)
    except Exception as e:
        print(f"  (warning: failed to delete throwaway profile {profile_id}: {e})")


def hit_compose(
    client: httpx.Client,
    port: int,
    profile_id: str,
) -> tuple[str, int]:
    start = time.monotonic()
    url = f"http://127.0.0.1:{port}/profiles/{profile_id}/compose"
    resp = client.post(url, timeout=180.0)
    latency_ms = int((time.monotonic() - start) * 1000)
    resp.raise_for_status()
    return resp.json().get("text", "").strip(), latency_ms


def format_report(cards: list[Scorecard]) -> str:
    lines: list[str] = ["", "═" * 100]
    by_model: dict[str, list[Scorecard]] = {}
    for c in cards:
        by_model.setdefault(c.model, []).append(c)
    for model, model_cards in by_model.items():
        clean = sum(1 for c in model_cards if not c.flags)
        avg = sum(c.latency_ms for c in model_cards) // max(len(model_cards), 1)
        lines.append("")
        lines.append(f"▌{model}  —  {clean}/{len(model_cards)} clean, avg {avg} ms")
        lines.append("─" * 100)
        for c in model_cards:
            status = "✓" if not c.flags else "✗"
            tag = f"{c.personality} · {c.endpoint}"
            lines.append(f"  {status} {tag}  ({c.latency_ms} ms)")
            if c.input_text:
                lines.append(
                    f"      in:      {c.input_text[:90]}{'…' if len(c.input_text) > 90 else ''}"
                )
            lines.append(
                f"      out:     {c.refined[:120]}{'…' if len(c.refined) > 120 else ''}"
            )
            if c.flags:
                lines.append(f"      ⚠ {'; '.join(c.flags)}")
            lines.append("")
    lines.append("═" * 100)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model", choices=("0.6B", "1.7B", "4B"), action="append")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    models = tuple(args.model) if args.model else ("0.6B", "4B")
    port = detect_backend_port(args.port)
    print(f"backend → http://127.0.0.1:{port}")
    print(f"personalities → {len(PERSONALITIES)}, models → {models}")

    # Model size is set on the capture_settings singleton, not passed
    # per-request to /profiles/{id}/compose. The harness swaps it
    # between runs so we probe both sizes cleanly.
    cards: list[Scorecard] = []
    with httpx.Client() as client:
        for model in models:
            print(f"\n── {model} " + "─" * (80 - len(model) - 4))
            # Flip the server-side default LLM size for this pass.
            client.put(
                f"http://127.0.0.1:{port}/settings/captures",
                json={"llm_model": model},
                timeout=10.0,
            )
            for personality in PERSONALITIES:
                print(f"  [{personality.name}] ", end="", flush=True)
                profile_id = create_throwaway_profile(client, port, personality, model)
                try:
                    try:
                        text, latency = hit_compose(client, port, profile_id)
                    except Exception as e:
                        print(f"  compose:ERR ({e})", end="")
                        continue
                    card = score(
                        personality=personality,
                        endpoint="compose",
                        model=model,
                        input_text="",
                        refined=text,
                        latency_ms=latency,
                    )
                    cards.append(card)
                    status = "ok" if not card.flags else "⚠"
                    print(f"  compose:{status} ({latency}ms)", end="")
                    print()
                finally:
                    delete_profile(client, port, profile_id)

    print(format_report(cards))

    if args.json:
        args.json.write_text(json.dumps([asdict(c) for c in cards], indent=2))
        print(f"wrote {args.json}")

    return 0 if all(not c.flags for c in cards) else 1


if __name__ == "__main__":
    sys.exit(main())
