"""Knowledge retrieval for the voice agent.

A per-turn "RAG-lite": articles are scored by weighted keyword overlap
against the customer's last few utterances and the top few are pasted
into the system prompt. No embeddings, no extra model download — the
knowledge base for one agent is small (tens to hundreds of FAQ /
troubleshooting entries) and a phone turn is one sentence, so lexical
matching is both fast and good enough, and it keeps the whole stack
local like the rest of Voicebox.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..database import KnowledgeArticle

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'+-]{1,}")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "so",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "not",
        "no",
        "yes",
        "ok",
        "okay",
        "please",
        "thanks",
        "thank",
        "hi",
        "hello",
        "hey",
        "um",
        "uh",
        "like",
        "just",
        "really",
        "very",
        "about",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "how",
        "all",
        "any",
        "some",
        "more",
        "most",
        "other",
        "than",
        "then",
        "too",
        "also",
        "get",
        "got",
        "getting",
        "go",
        "going",
        "went",
        "come",
        "came",
        "want",
        "wanted",
        "need",
        "needed",
        "know",
        "think",
        "say",
        "said",
        "tell",
        "told",
        "one",
        "two",
        "three",
        "now",
        "still",
        "again",
        "into",
        "over",
        "under",
        "out",
        "up",
        "down",
        "off",
    ]
)

# Title and tag hits are worth more than body hits — an operator who titled
# an article "Password reset" wants it surfaced for "reset my password".
_TITLE_WEIGHT = 3.0
_TAG_WEIGHT = 2.5
_BODY_WEIGHT = 1.0

DEFAULT_TOP_K = 3
MAX_CONTEXT_CHARS = 3500


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


@dataclass
class ScoredArticle:
    article: KnowledgeArticle
    score: float


def _idf(articles: list[KnowledgeArticle]) -> dict[str, float]:
    """Inverse document frequency over the agent's own articles, so a word
    that appears in every entry ("account") doesn't dominate."""
    n = len(articles)
    df: Counter[str] = Counter()
    for a in articles:
        df.update(set(tokenize(f"{a.title} {a.tags or ''} {a.content}")))
    return {t: math.log((n + 1) / (c + 0.5)) + 1.0 for t, c in df.items()}


def rank_articles(
    articles: list[KnowledgeArticle],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[ScoredArticle]:
    """Score every article against ``query`` and return the best ``top_k``
    with a positive score."""
    q_tokens = tokenize(query)
    if not q_tokens or not articles:
        return []
    q_counts = Counter(q_tokens)
    idf = _idf(articles)

    scored: list[ScoredArticle] = []
    for a in articles:
        title_tokens = Counter(tokenize(a.title))
        tag_tokens = Counter(tokenize((a.tags or "").replace(",", " ")))
        body_tokens = Counter(tokenize(a.content))
        body_len = max(1, sum(body_tokens.values()))
        score = 0.0
        for tok, q_n in q_counts.items():
            w = idf.get(tok, 1.0)
            if tok in title_tokens:
                score += _TITLE_WEIGHT * w * q_n
            if tok in tag_tokens:
                score += _TAG_WEIGHT * w * q_n
            if tok in body_tokens:
                # Saturating term frequency (BM25-ish) so a long article
                # that repeats a word doesn't crowd out a focused one.
                tf = body_tokens[tok]
                score += _BODY_WEIGHT * w * q_n * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * body_len / 200.0))
        if score > 0:
            scored.append(ScoredArticle(article=a, score=score))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]


def retrieve_for_turn(
    db: Session,
    agent_id: str,
    recent_customer_text: list[str],
    *,
    top_k: int = DEFAULT_TOP_K,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> list[tuple[str, str]]:
    """Return ``(title, content)`` pairs to inject into the system prompt.

    The most recent customer utterance is weighted by being included
    twice — the current question matters more than what led up to it.
    """
    articles = db.query(KnowledgeArticle).filter(KnowledgeArticle.agent_id == agent_id).all()
    if not articles:
        return []
    if not recent_customer_text:
        return []
    query = " ".join([*recent_customer_text[-3:], recent_customer_text[-1]])
    ranked = rank_articles(articles, query, top_k=top_k)

    out: list[tuple[str, str]] = []
    budget = max_chars
    for item in ranked:
        content = item.article.content.strip()
        if len(content) > budget:
            content = content[: max(0, budget - 1)].rstrip() + "…"
        if not content:
            break
        out.append((item.article.title.strip(), content))
        budget -= len(content)
        if budget <= 0:
            break
    return out
