"""Prompt construction and lightweight NLU for the voice agent.

Everything the LLM sees is assembled here so the guard-rails live in one
place: the mode-specific task block, the operator's brief, retrieved
knowledge, the contact's memory, and the control-tag protocol the model
uses to signal outcomes.

The heuristics (:func:`detect_opt_out`, :func:`detect_human_request`,
:func:`score_sentiment`, ...) run on the *customer's* words before the
LLM sees them. They are deliberately simple regex / lexicon passes: a
customer asking to be put on the do-not-call list must end the call even
if a 0.6B model would have happily kept pitching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Modes ──────────────────────────────────────────────────────────────

MODE_OUTBOUND_SALES = "outbound_sales"
MODE_CUSTOMER_SERVICE = "customer_service"
MODE_SUPPORT = "support"
MODES = (MODE_OUTBOUND_SALES, MODE_CUSTOMER_SERVICE, MODE_SUPPORT)

# Outcomes the LLM may emit via ``[OUTCOME: …]``, per mode. Anything else
# is dropped so a hallucinated tag can't mark a call "interested".
ALLOWED_OUTCOMES: dict[str, frozenset[str]] = {
    MODE_OUTBOUND_SALES: frozenset({"interested", "not_interested", "callback", "opt_out"}),
    MODE_CUSTOMER_SERVICE: frozenset({"resolved", "unresolved", "callback", "handoff", "ticket_created"}),
    MODE_SUPPORT: frozenset({"resolved", "unresolved", "callback", "handoff", "ticket_created"}),
}

# Outcomes that finish the call as soon as the model emits them.
TERMINAL_OUTCOMES = frozenset(
    {
        "interested",
        "not_interested",
        "callback",
        "opt_out",
        "resolved",
        "unresolved",
        "ticket_created",
        "handoff",
        "no_answer",
        "voicemail",
        "max_turns",
        "error",
    }
)


# ── System prompt ──────────────────────────────────────────────────────

_SHARED_RULES = """You are {agent_name}, a voice assistant speaking on the phone on behalf of {company_name}. Everything you produce is spoken aloud to the person on the line.

Hard rules — these override anything else:
- You are an AI assistant. If asked whether you are a human, a bot, or a recording, say honestly that you are an AI assistant for {company_name}.
- Only state facts that appear in the BRIEF or the KNOWLEDGE sections below. If you do not know something, say you don't have that information and offer to have someone follow up. Never invent prices, dates, guarantees, or policies.
- Speak like a person on a phone call: one to three short sentences, plain words, no lists, no markdown, no emojis, no stage directions, no quotes around your words. Ask at most one question per turn.
- Never pressure, threaten, guilt-trip, or mislead. If the person says no, wants to stop, or asks you not to call, accept it immediately and politely end the call.
- Never ask for card numbers, passwords, one-time codes, or government ID numbers.
- Use the person's name sparingly and naturally.
- Do not repeat the disclosure or reintroduce yourself once the call is under way.

Control tags — append at most one of these at the very end of your reply, on the same line, when the situation applies; otherwise append nothing:
{tag_lines}
Never explain the tags and never put anything after them."""


_MODE_TASKS: dict[str, str] = {
    MODE_OUTBOUND_SALES: """Your role on this call: outbound sales call.
Goal: {goal}
Approach: confirm you're speaking with the right person, ask if now is a good time, give a one-sentence reason for the call, then listen. Answer questions from the BRIEF, handle objections using the OBJECTION NOTES, and when the person shows interest move to the concrete next step described in the goal. If they are busy, offer a callback and ask for a good time. If they are not interested, thank them and end warmly — never re-pitch after a clear no.""",
    MODE_CUSTOMER_SERVICE: """Your role on this call: inbound customer service for {company_name}.
Goal: {goal}
Approach: greet, find out what the person needs, and answer from the KNOWLEDGE and BRIEF sections. Confirm you've understood before answering. If the request needs a human or is outside what you can do, say so, explain what happens next ({escalation_promise}), and log it. Before ending, check whether there is anything else you can help with.""",
    MODE_SUPPORT: """Your role on this call: technical / product support for {company_name}.
Goal: {goal}
Approach: get a clear description of the problem (what they were doing, what happened, what they expected). Ask one diagnostic question at a time. Walk through fixes from the KNOWLEDGE section one step at a time and wait for the person to confirm each step before moving on. If a step fixes it, confirm and close. If you run out of steps or the issue needs a specialist, say so, explain what happens next ({escalation_promise}), and open a ticket. Before ending, check whether there is anything else you can help with.""",
}


_TAG_DOCS: dict[str, list[str]] = {
    MODE_OUTBOUND_SALES: [
        "[OUTCOME: interested] — the person agreed to the next step in the goal.",
        "[OUTCOME: callback] — the person asked to be called back later.",
        "[OUTCOME: not_interested] — the person clearly declined.",
        "[OUTCOME: opt_out] — the person asked not to be contacted again.",
    ],
    MODE_CUSTOMER_SERVICE: [
        "[OUTCOME: resolved] — the person's request is fully handled and they have nothing else.",
        "[OUTCOME: callback] — the person wants a call back later.",
        "[TICKET: short subject] — the request needs follow-up by a person; describe it in a few words.",
        "[HANDOFF] — the person insists on speaking to a human, or you cannot help safely.",
    ],
    MODE_SUPPORT: [
        "[OUTCOME: resolved] — the issue is fixed and the person confirmed it.",
        "[OUTCOME: callback] — the person wants a call back later.",
        "[TICKET: short subject] — the issue could not be fixed on the call; describe it in a few words.",
        "[HANDOFF] — the person insists on speaking to a human, or the problem is unsafe to handle by phone.",
    ],
}


def _section(title: str, body: str | None) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"\n\n{title}:\n{body}"


def build_system_prompt(
    *,
    mode: str,
    agent_name: str,
    company_name: str,
    brief: str,
    goal: str,
    objection_notes: str | None = None,
    persona: str | None = None,
    escalation_promise: str | None = None,
    contact_name: str | None = None,
    contact_company: str | None = None,
    contact_notes: str | None = None,
    contact_memory: str | None = None,
    knowledge: list[tuple[str, str]] | None = None,
) -> str:
    """Assemble the full system prompt for one turn.

    ``knowledge`` is a list of ``(title, content)`` pairs already narrowed
    to the current turn by :mod:`voice_agent_knowledge`.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown agent mode '{mode}'. Must be one of: {', '.join(MODES)}")

    promise = (escalation_promise or "a member of the team will follow up").strip()
    tag_lines = "\n".join(f"- {line}" for line in _TAG_DOCS[mode])
    header = _SHARED_RULES.format(
        agent_name=agent_name,
        company_name=company_name,
        tag_lines=tag_lines,
    )
    task = _MODE_TASKS[mode].format(
        goal=goal.strip(),
        company_name=company_name,
        escalation_promise=promise,
    )

    contact_bits: list[str] = []
    if contact_name:
        contact_bits.append(f"Name: {contact_name}")
    if contact_company:
        contact_bits.append(f"Company: {contact_company}")
    if contact_notes:
        contact_bits.append(f"Notes: {contact_notes.strip()}")
    if contact_memory:
        contact_bits.append(f"Previous conversations: {contact_memory.strip()}")

    knowledge_block = ""
    if knowledge:
        knowledge_block = "\n\n".join(f"## {title}\n{content.strip()}" for title, content in knowledge)

    return (
        header
        + "\n\n"
        + task
        + _section("PERSONA", persona)
        + _section("BRIEF", brief)
        + _section("OBJECTION NOTES", objection_notes)
        + _section("KNOWLEDGE", knowledge_block)
        + _section("PERSON ON THE LINE", "\n".join(contact_bits))
    )


def build_opening_line(
    *,
    mode: str,
    agent_name: str,
    company_name: str,
    disclosure: str,
    contact_name: str | None,
    custom_opening: str | None = None,
) -> str:
    """The deterministic first utterance. Always includes the disclosure so
    the AI-identity statement can't be dropped by the model."""
    greeting = f"Hi {contact_name.split()[0]}," if contact_name and contact_name.strip() else "Hi,"
    disclosure = disclosure.strip()
    if custom_opening and custom_opening.strip():
        hook = custom_opening.strip()
    elif mode == MODE_OUTBOUND_SALES:
        hook = "Do you have a quick minute?"
    elif mode == MODE_CUSTOMER_SERVICE:
        hook = "How can I help you today?"
    else:
        hook = "What can I help you sort out today?"

    if mode == MODE_OUTBOUND_SALES:
        intro = f"{greeting} this is {agent_name} calling from {company_name}."
    else:
        intro = f"{greeting} you've reached {company_name}, this is {agent_name}."
    return f"{intro} {disclosure} {hook}"


# ── Summary prompt ─────────────────────────────────────────────────────

SUMMARY_SYSTEM = """You summarise phone calls for a CRM. Given a transcript, write two to four plain sentences covering: what the person wanted or how they responded, anything they asked for, and what was agreed. Write in the third person, past tense, no headings, no bullet points, no preamble."""


def build_summary_prompt(turns: list[tuple[str, str]], contact_name: str | None) -> str:
    who = contact_name or "the customer"
    lines = [f"{'Agent' if role == 'agent' else who}: {text}" for role, text in turns]
    return "Transcript:\n" + "\n".join(lines)


# ── Control-tag parsing ────────────────────────────────────────────────

_OUTCOME_TAG = re.compile(r"\[\s*OUTCOME\s*:\s*([a-z_]+)\s*\]", re.IGNORECASE)
_TICKET_TAG = re.compile(r"\[\s*TICKET\s*(?::\s*([^\]]*))?\]", re.IGNORECASE)
_HANDOFF_TAG = re.compile(r"\[\s*HANDOFF\s*\]", re.IGNORECASE)
_END_TAG = re.compile(r"\[\s*END\s*\]", re.IGNORECASE)
_ANY_TAG = re.compile(r"\[\s*(OUTCOME|TICKET|HANDOFF|END)\b[^\]]*\]", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_STAGE_DIRECTION = re.compile(r"(\*[^*\n]{1,60}\*)|(\([^)\n]{1,60}\))")


@dataclass
class ParsedReply:
    """What the model said, minus the protocol."""

    text: str
    outcome: str | None = None
    ticket_subject: str | None = None
    handoff: bool = False
    end: bool = False
    raw: str = field(default="", repr=False)


def parse_agent_reply(raw: str, mode: str) -> ParsedReply:
    """Strip control tags and wrapper junk from an LLM reply.

    Unknown outcomes for the mode are ignored rather than trusted.
    """
    text = _THINK_BLOCK.sub("", raw or "")
    outcome: str | None = None
    ticket_subject: str | None = None
    handoff = False
    end = False

    m = _OUTCOME_TAG.search(text)
    if m:
        candidate = m.group(1).lower()
        if candidate in ALLOWED_OUTCOMES.get(mode, frozenset()):
            outcome = candidate
    m = _TICKET_TAG.search(text)
    if m:
        ticket_subject = (m.group(1) or "").strip() or None
        if ticket_subject is None:
            ticket_subject = "Follow-up requested"
    if _HANDOFF_TAG.search(text):
        handoff = True
    if _END_TAG.search(text):
        end = True

    text = _ANY_TAG.sub("", text)
    text = _STAGE_DIRECTION.sub("", text)
    # Small models sometimes label their own line or wrap it in quotes.
    text = re.sub(r"^\s*(agent|assistant|" + r"[A-Za-z]+)\s*:\s*", "", text, count=1) if _looks_labelled(text) else text
    text = text.strip().strip('"').strip("“”").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return ParsedReply(
        text=text,
        outcome=outcome,
        ticket_subject=ticket_subject,
        handoff=handoff,
        end=end,
        raw=raw or "",
    )


def _looks_labelled(text: str) -> bool:
    head = text.lstrip()[:40].lower()
    return head.startswith(("agent:", "assistant:")) or bool(re.match(r"^[a-z]{2,20}:\s", head))


# ── Customer-side heuristics ───────────────────────────────────────────

_OPT_OUT_PATTERNS = [
    r"\bstop calling\b",
    r"\bdo not call\b",
    r"\bdon'?t call\b",
    r"\bnever call\b",
    r"\bquit calling\b",
    r"\bremove (me|my number|us)\b",
    r"\btake (me|my number|us) off\b",
    r"\bunsubscribe\b",
    r"\bdo[ -]?not[ -]?call list\b",
    r"\bno more calls\b",
    r"\bnot (to )?(call|contact) (me|us) again\b",
    r"\bstop contacting\b",
    r"\blose my number\b",
]
_OPT_OUT_RE = re.compile("|".join(_OPT_OUT_PATTERNS), re.IGNORECASE)

_HUMAN_PATTERNS = [
    r"\b(speak|talk|put me through|transfer me|connect me)\b.{0,30}\b(human|person|someone real|real person|representative|rep|agent|operator|manager|supervisor)\b",
    r"\b(human|real person|representative|operator|supervisor|manager)\b.{0,20}\b(please|now)\b",
    r"\bare you (a )?(bot|robot|machine|recording|computer|ai)\b",
    r"\bi want a (human|person|representative)\b",
    r"\bget me (a|an|the) (human|person|representative|manager|supervisor)\b",
]
_HUMAN_RE = re.compile("|".join(_HUMAN_PATTERNS), re.IGNORECASE)
# Asking *whether* we're a bot is not a handoff request — it's answered
# honestly by the model. Only demands to reach a person count.
_ARE_YOU_BOT_RE = re.compile(r"\bare you (a )?(bot|robot|machine|recording|computer|ai)\b", re.IGNORECASE)

_GOODBYE_RE = re.compile(
    r"^\s*(ok(ay)?[,.! ]*)?(thanks?( you)?[,.! ]*)?((that'?s|that is|that will be|that'?ll be) (all|it|everything)|nothing else|no,? (that'?s|that is) (all|it)|i'?m (all )?(good|set|done)|bye|goodbye|good ?bye|have a (good|nice|great) (day|one))[.! ]*$",
    re.IGNORECASE,
)

_CALLBACK_RE = re.compile(
    r"\b(call (me )?back|ring (me )?back|later (today|this week|tomorrow)|not a good time|bad time|busy right now|in a meeting|can you call (me )?(later|tomorrow|next week)|try (me )?(again )?(later|tomorrow|next week))\b",
    re.IGNORECASE,
)

_NOT_INTERESTED_RE = re.compile(
    r"\b(not interested|no thanks|no thank you|don'?t need|not for me|we'?re (all )?set|already have (one|that|it)|not looking|no,? i'?m (fine|good|okay|ok))\b",
    re.IGNORECASE,
)

_INTERESTED_RE = re.compile(
    r"\b(sign me up|i'?m interested|sounds (good|great|interesting)|tell me more|let'?s do (it|that)|book (it|me|that)|yes,? please|go ahead|send me (the|that|it|more)|how do i (sign up|get started|buy)|i'?d like (to|that))\b",
    re.IGNORECASE,
)

_VOICEMAIL_RE = re.compile(
    r"\b(leave (a|your) message|after the (tone|beep)|not available (right now|at the moment)|can'?t (take|come to) (your|the) (call|phone)|you have reached the voicemail|mailbox)\b",
    re.IGNORECASE,
)


def detect_opt_out(text: str) -> bool:
    return bool(_OPT_OUT_RE.search(text or ""))


def detect_human_request(text: str) -> bool:
    """True when the caller demands a person (not merely asks if we're a bot)."""
    if not text:
        return False
    if _ARE_YOU_BOT_RE.search(text) and not re.search(
        r"\b(want|need|get me|speak|talk|transfer)\b", text, re.IGNORECASE
    ):
        return False
    return bool(_HUMAN_RE.search(text))


def detect_goodbye(text: str) -> bool:
    return bool(_GOODBYE_RE.match(text or ""))


def detect_voicemail(text: str) -> bool:
    return bool(_VOICEMAIL_RE.search(text or ""))


def classify_customer_intent(text: str, mode: str) -> str | None:
    """Coarse intent from the customer's words. Returns an outcome name or
    None. Opt-out always wins; the rest are mode-specific.

    Only ``opt_out`` and ``handoff`` are acted on directly — the other
    labels are hints the caller may use when the LLM emitted no tag.
    """
    if detect_opt_out(text):
        return "opt_out"
    if detect_human_request(text):
        return "handoff"
    if _CALLBACK_RE.search(text or ""):
        return "callback"
    if mode == MODE_OUTBOUND_SALES:
        if _NOT_INTERESTED_RE.search(text or ""):
            return "not_interested"
        if _INTERESTED_RE.search(text or ""):
            return "interested"
    return None


# Tiny valence lexicon. Enough to spot an escalating caller without
# shipping another model; not meant to be a research-grade classifier.
_NEGATIVE_WORDS = {
    "angry",
    "furious",
    "ridiculous",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "useless",
    "unacceptable",
    "frustrated",
    "frustrating",
    "annoyed",
    "annoying",
    "disgusted",
    "disgusting",
    "pathetic",
    "joke",
    "scam",
    "fraud",
    "lie",
    "lied",
    "liar",
    "sue",
    "lawyer",
    "complaint",
    "complain",
    "cancel",
    "refund",
    "broken",
    "hate",
    "stupid",
    "idiot",
    "waste",
    "wasting",
    "fail",
    "failed",
    "fails",
    "wrong",
    "problem",
    "problems",
    "issue",
    "issues",
    "unhappy",
    "disappointed",
    "disappointing",
    "rubbish",
    "garbage",
    "crap",
    "damn",
    "hell",
}
_POSITIVE_WORDS = {
    "great",
    "good",
    "thanks",
    "thank",
    "perfect",
    "awesome",
    "excellent",
    "wonderful",
    "helpful",
    "appreciate",
    "appreciated",
    "brilliant",
    "lovely",
    "fantastic",
    "happy",
    "pleased",
    "glad",
    "works",
    "worked",
    "fixed",
    "solved",
    "sorted",
    "yes",
    "sure",
    "okay",
    "ok",
    "fine",
    "cool",
    "nice",
    "amazing",
    "love",
    "interested",
}
_INTENSIFIERS = {"very", "so", "really", "extremely", "absolutely", "totally", "completely"}
_NEGATORS = {
    "not",
    "no",
    "never",
    "don't",
    "dont",
    "isn't",
    "isnt",
    "wasn't",
    "didn't",
    "doesn't",
    "can't",
    "cannot",
    "won't",
}
_WORD_RE = re.compile(r"[a-z']+")


def score_sentiment(text: str) -> float:
    """Return a valence in [-1, 1]. 0 for neutral / empty."""
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return 0.0
    score = 0.0
    boost = 1.0
    negate = False
    for w in words:
        if w in _INTENSIFIERS:
            boost = 1.5
            continue
        if w in _NEGATORS:
            negate = True
            continue
        sign = -1.0 if negate else 1.0
        if w in _NEGATIVE_WORDS:
            score -= 1.0 * boost * sign
        elif w in _POSITIVE_WORDS:
            score += 1.0 * boost * sign
        boost = 1.0
        negate = False
    # Shouting and repeated punctuation read as heat.
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        score -= 1.0
    if "!!" in text or "?!" in text:
        score -= 0.5
    denom = max(3.0, len(words) ** 0.5 * 2)
    return max(-1.0, min(1.0, score / denom))


# ── Canned closings ────────────────────────────────────────────────────
# Spoken without the LLM for the cases where getting the words exactly
# right matters more than sounding spontaneous.


def opt_out_closing(company_name: str) -> str:
    return (
        f"Understood. I'm adding your number to {company_name}'s do-not-call list right now, "
        "so you won't hear from us again. Sorry for the interruption, and have a good day."
    )


def handoff_closing(escalation_promise: str | None) -> str:
    promise = (
        (escalation_promise or "a member of our team will follow up with you as soon as possible").strip().rstrip(".")
    )
    return f"Of course. I'll pass this to a person — {promise}. Thanks for your patience, and goodbye for now."


def max_turns_closing(escalation_promise: str | None) -> str:
    promise = (escalation_promise or "someone from the team will follow up").strip().rstrip(".")
    return f"I want to make sure you get properly looked after, so I'm going to hand this over — {promise}. Thanks for your time today."


def goodbye_closing(agent_mode: str) -> str:
    if agent_mode == MODE_OUTBOUND_SALES:
        return "Thanks for your time, have a good day."
    return "Glad I could help. Thanks for calling, and have a good day."
