import os
import re
import sqlite3
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_SOLUTION_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_UDAHUB_DB = os.path.join(_SOLUTION_ROOT, "data", "core", "udahub.db")

# Articles scoring below this threshold are considered irrelevant and excluded.
# Score = title_match×3 + tag_match×2 + content_match×1 per query keyword.
# A score of 3.0 requires at least one title match or 1.5 tag matches — a meaningful signal.
MIN_RELEVANCE_SCORE = 3.0

# Included in the tool response when retrieval fails; the Resolver detects this
# marker to trigger deterministic escalation without relying on LLM confidence.
ESCALATION_MARKER = "<<ESCALATION_REQUIRED>>"

_STOP_WORDS = {
    "i", "me", "my", "the", "a", "an", "is", "are", "was", "to", "do",
    "how", "what", "can", "you", "your", "in", "on", "at", "it", "its",
    "this", "that", "and", "or", "but", "for", "with", "about", "have",
    "has", "not", "be", "been", "will", "would", "could", "should", "get",
}


def _score_article(query_words: set[str], title: str, content: str, tags: str) -> float:
    """Weighted keyword relevance: title matches count 3x, tag matches 2x, content 1x."""
    title_lower = title.lower()
    content_lower = (content or "").lower()
    tags_lower = (tags or "").lower()

    score = 0.0
    for word in query_words:
        if word in title_lower:
            score += 3.0
        if word in tags_lower:
            score += 2.0
        if word in content_lower:
            score += 1.0
    return score


@tool
def search_knowledge_base(query: str) -> str:
    """Search the CultPass knowledge base for articles relevant to the customer's issue.

    Args:
        query: The customer's question or issue description.

    Returns:
        Up to 3 most relevant knowledge base articles above the relevance threshold,
        or an escalation signal if no sufficiently relevant articles are found.
    """
    if not os.path.exists(_UDAHUB_DB):
        logger.error("KB database not found at %s", _UDAHUB_DB)
        return f"Knowledge base database not found. {ESCALATION_MARKER}"

    try:
        conn = sqlite3.connect(_UDAHUB_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT article_id, title, content, tags FROM knowledge")
        all_articles = cursor.fetchall()
        conn.close()
    except Exception as e:
        logger.error("DB error in search_knowledge_base: %s", e)
        return f"Error accessing knowledge base: {e}. {ESCALATION_MARKER}"

    if not all_articles:
        logger.warning("Knowledge base is empty")
        return f"Knowledge base is empty. {ESCALATION_MARKER}"

    clean_query = re.sub(r"[^\w\s]", "", query.lower())
    query_words = {w for w in clean_query.split() if w not in _STOP_WORDS and len(w) > 2}

    if not query_words:
        return f"Query too generic to search effectively. Please provide more details. {ESCALATION_MARKER}"

    scored = []
    for article_id, title, content, tags in all_articles:
        score = _score_article(query_words, title, content, tags)
        if score >= MIN_RELEVANCE_SCORE:
            scored.append((score, title, content, tags))

    scored.sort(reverse=True, key=lambda x: x[0])

    if not scored:
        logger.info("KB search: no articles above threshold (score≥%.1f) for '%s'", MIN_RELEVANCE_SCORE, query[:60])
        return (
            f"No relevant knowledge base articles found for this query "
            f"(minimum relevance score {MIN_RELEVANCE_SCORE} not met). "
            f"This issue requires escalation to a human support agent. {ESCALATION_MARKER}"
        )

    top = scored[:3]
    parts = []
    for score, title, content, tags in top:
        parts.append(f"### {title} [score={score:.1f}]\n{content}\n_(Tags: {tags})_")

    logger.info(
        "KB search: %d relevant article(s) found for '%s' (top score=%.1f)",
        len(top), query[:60], top[0][0],
    )
    return "\n\n---\n\n".join(parts)
