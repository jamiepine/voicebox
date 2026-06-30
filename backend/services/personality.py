"""
Personality-driven text generation — lets a voice profile "speak" or
restate text using an LLM that takes on the character described by the
profile's ``personality`` prompt.

Two entry points:

- :func:`compose_as_profile` — zero-input, the character produces a fresh
  utterance. Wired to the Compose button in the generate box and to the
  ``/profiles/{id}/compose`` endpoint.
- :func:`rewrite_as_profile` — takes user text, restates it in the
  character's voice while keeping every idea. Invoked by ``POST /generate``
  (and ``POST /speak``) when ``personality=true`` and the profile has a
  personality prompt set.

Both reuse the same local Qwen3 instance that refinement uses — no extra
model downloads, no extra warm-up. Temperature is tuned per mode: compose
runs hot (0.9) for variety, rewrite cool (0.3) for fidelity to the user's
ideas.
"""

from dataclasses import dataclass

from . import llm as llm_service
from .refinement import collapse_repetitive_artifacts


# Shared rules block embedded in every mode-specific system prompt. Kept
# short because small LLMs (0.6B) degrade when the system prompt is long,
# and because the per-mode instructions downstream carry the specifics.
_CHARACTER_FRAMING = """You are roleplaying a specific character described below. Stay fully in character in everything you produce.

Rules that apply to every response:
- Do not break character. Do not explain what you are doing, refuse, apologize, greet the user, or acknowledge being an AI or assistant.
- Do not narrate action ("*smiles*", "(leans back)") or stage directions. Produce speech only.
- Do not wrap the output in quotes, code fences, or labels. Output the character's words and nothing else.
- Match the character's register — if they are curt, be curt; if they ramble, ramble; if they swear, swear."""


_COMPOSE_TASK = """Task: Produce one short utterance — one or two sentences at most — that this character might say right now, unprompted. A remark, an observation, a thought out loud. No greeting, no addressing anyone by name, no "Well, …" or "So, …" opener unless it fits the character naturally. Just a natural line of speech."""


_REWRITE_TASK = """Task: The user's next message is a piece of text. Restate every idea in it using your character's voice — keep the meaning, change the wording. Do not add new ideas, do not drop any, do not reply to the text. Output only the restated version."""


@dataclass
class PersonalityResult:
    """What the three service functions return."""

    text: str
    model_size: str


def _build_system_prompt(personality: str, task: str) -> str:
    return (
        _CHARACTER_FRAMING
        + "\n\nCharacter description:\n"
        + personality.strip()
        + "\n\n"
        + task
    )


def _require_personality(personality: str | None) -> str:
    if not personality or not personality.strip():
        raise ValueError(
            "This profile has no personality set. Add one on the profile to use compose or personality-rewrite."
        )
    return personality


async def compose_as_profile(
    personality: str | None,
    model_size: str | None = None,
) -> PersonalityResult:
    """Produce a fresh utterance in the character's voice.

    No user input; the system prompt plus a trigger user turn ("Speak.")
    is all the model gets. Temperature is high so successive calls
    produce different outputs — the UI's Compose button is expected to
    be clicked repeatedly for variety.
    """
    text = _require_personality(personality)
    backend = llm_service.get_llm_model()
    resolved_size = model_size or backend.model_size

    system_prompt = _build_system_prompt(text, _COMPOSE_TASK)
    output = await backend.generate(
        prompt="Speak.",
        system=system_prompt,
        max_tokens=256,
        temperature=0.9,
        model_size=resolved_size,
    )
    return PersonalityResult(text=output.strip(), model_size=resolved_size)


async def rewrite_as_profile(
    personality: str | None,
    user_text: str,
    model_size: str | None = None,
) -> PersonalityResult:
    """Restate the user's text in the character's voice, ideas intact."""
    character = _require_personality(personality)
    cleaned = collapse_repetitive_artifacts(user_text)
    if not cleaned.strip():
        raise ValueError("Rewrite needs non-empty text to restate.")

    backend = llm_service.get_llm_model()
    resolved_size = model_size or backend.model_size

    system_prompt = _build_system_prompt(character, _REWRITE_TASK)
    output = await backend.generate(
        prompt=cleaned,
        system=system_prompt,
        max_tokens=1024,
        temperature=0.3,
        model_size=resolved_size,
    )
    return PersonalityResult(text=output.strip(), model_size=resolved_size)
