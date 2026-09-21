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

import ipaddress
import math
import re
import socket
import uuid
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import urlparse

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


# ── v2: import & chunking ──────────────────────────────────────────────


MAX_CHUNK_CHARS = 1200
MAX_FETCH_BYTES = 5 * 1024 * 1024
FETCH_TIMEOUT_S = 20.0


class _TextExtractor(HTMLParser):
    """Turn a page into readable text: headings become ``## `` lines,
    block elements become paragraph breaks, script/style/nav are dropped."""

    _SKIP: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe"}
    )
    _BLOCK: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "li", "br", "tr", "section", "article", "blockquote", "pre", "td", "th", "dt", "dd"}
    )
    _HEADINGS: ClassVar[frozenset[str]] = frozenset({"h1", "h2", "h3", "h4"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._heading: str | None = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._HEADINGS:
            self._heading = tag
            self.parts.append("\n\n## ")
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._HEADINGS:
            self._heading = None
            self.parts.append("\n")
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text) with light structure preserved."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return parser.title.strip(), text


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str | None, str]]:
    """Split text into (heading, body) chunks of at most ``max_chars``.

    Headings (``## …`` lines or Markdown ``#`` lines) start a new chunk and
    label it; long sections are cut at paragraph, then sentence, boundaries.
    """
    chunks: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    def _flush() -> None:
        body = "\n\n".join(p for p in buffer if p.strip()).strip()
        buffer.clear()
        if not body:
            return
        for piece in _split_long(body, max_chars):
            chunks.append((heading, piece))

    for para in re.split(r"\n\s*\n", text or ""):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"^#{1,4}\s+(.+)$", para.splitlines()[0])
        if m:
            _flush()
            heading = m.group(1).strip()[:120]
            rest = "\n".join(para.splitlines()[1:]).strip()
            if rest:
                buffer.append(rest)
            continue
        if sum(len(p) for p in buffer) + len(para) > max_chars:
            _flush()
        buffer.append(para)
    _flush()
    return chunks


def _split_long(body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    out: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if len(current) + len(sentence) + 1 > max_chars and current:
            out.append(current.strip())
            current = ""
        if len(sentence) > max_chars:
            # A single monstrous sentence: hard cut.
            for i in range(0, len(sentence), max_chars):
                out.append(sentence[i : i + max_chars].strip())
            continue
        current = f"{current} {sentence}".strip()
    if current:
        out.append(current.strip())
    return out


def import_text(
    db: Session,
    agent_id: str,
    title: str,
    text: str,
    *,
    source: str | None = None,
    tags: list[str] | None = None,
) -> list[KnowledgeArticle]:
    """Chunk ``text`` into articles titled after the document (and its
    headings) and store them for the agent."""
    title = (title or "Imported document").strip()[:150]
    tag_str = ",".join(t.strip() for t in (tags or []) if t.strip()) or None
    rows: list[KnowledgeArticle] = []
    pieces = chunk_text(text)
    for index, (heading, body) in enumerate(pieces, start=1):
        if heading:
            name = f"{title} — {heading}" if heading.lower() != title.lower() else title
        elif len(pieces) > 1:
            name = f"{title} (part {index})"
        else:
            name = title
        rows.append(
            KnowledgeArticle(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                title=name[:200],
                content=body,
                tags=tag_str,
                source=source,
            )
        )
    db.add_all(rows)
    db.commit()
    for r in rows:
        db.refresh(r)
    return rows


def check_fetch_url(url: str) -> None:
    """Refuse anything but http(s) and the cloud-metadata / link-local range.

    Voicebox is loopback-bound by default and knowledge sources are
    operator-configured, so private LAN hosts stay allowed; the metadata
    range is the one address a server-side fetch must never reach.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must start with http:// or https://")
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host '{host}'") from exc
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if addr.is_link_local or addr.is_reserved or addr.is_multicast:
            raise ValueError("That address is not allowed.")


async def fetch_url_text(url: str) -> tuple[str, str]:
    """Download a page and return (title, text)."""
    check_fetch_url(url)
    import httpx  # lazy

    async with (
        httpx.AsyncClient(timeout=FETCH_TIMEOUT_S, follow_redirects=True, max_redirects=3) as client,
        client.stream("GET", url, headers={"User-Agent": "voicebox-voice-agent"}) as resp,
    ):
        if resp.status_code >= 400:
            raise ValueError(f"Fetch failed with HTTP {resp.status_code}")
        content_type = resp.headers.get("content-type", "")
        chunks: list[bytes] = []
        size = 0
        async for chunk in resp.aiter_bytes():
            size += len(chunk)
            if size > MAX_FETCH_BYTES:
                raise ValueError("Page is larger than 5 MB.")
            chunks.append(chunk)
    raw = b"".join(chunks)
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", content_type)
    if m:
        charset = m.group(1)
    try:
        body = raw.decode(charset, errors="replace")
    except LookupError:
        body = raw.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or body.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        title, text = html_to_text(body)
    else:
        title, text = "", body
    if not title:
        title = urlparse(url).path.rstrip("/").split("/")[-1] or urlparse(url).hostname or "Imported page"
    if not text.strip():
        raise ValueError("No readable text found at that URL.")
    return title, text


async def import_url(db: Session, agent_id: str, url: str, tags: list[str] | None = None) -> list[KnowledgeArticle]:
    title, text = await fetch_url_text(url)
    return import_text(db, agent_id, title, text, source=url, tags=tags)


def search(db: Session, agent_id: str, query: str, *, top_k: int = 5) -> list[ScoredArticle]:
    """Preview what the agent would retrieve for ``query``."""
    articles = db.query(KnowledgeArticle).filter(KnowledgeArticle.agent_id == agent_id).all()
    return rank_articles(articles, query, top_k=top_k)
