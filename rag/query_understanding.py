"""Lightweight query understanding for the RAG demo."""

from __future__ import annotations

import re
from typing import Dict, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "please",
    "research",
    "show",
    "study",
    "studies",
    "tell",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "using",
    "what",
    "which",
    "with",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_quoted_phrases(query: str) -> List[str]:
    return [match.strip() for match in re.findall(r'"([^"]+)"', query) if match.strip()]


def _extract_semantic_chunks(query: str) -> List[str]:
    chunks = re.split(r"[;,:()\-]|(?:\b(?:and|or|with|using|about|for)\b)", query, flags=re.IGNORECASE)
    cleaned = []
    for chunk in chunks:
        chunk = _normalize_whitespace(chunk)
        if len(chunk.split()) >= 2:
            cleaned.append(chunk)
    return cleaned


def _extract_keywords(query: str, max_keywords: int = 8) -> List[str]:
    phrases = _extract_quoted_phrases(query)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query.lower())

    keywords: List[str] = []
    seen = set()

    for phrase in phrases:
        normalized = _normalize_whitespace(phrase)
        lowered = normalized.lower()
        if lowered not in seen:
            keywords.append(normalized)
            seen.add(lowered)

    for chunk in _extract_semantic_chunks(query):
        lowered = chunk.lower()
        if lowered not in seen and len(lowered) <= 48:
            keywords.append(chunk)
            seen.add(lowered)
        if len(keywords) >= max_keywords:
            break

    for token in tokens:
        if token in STOPWORDS:
            continue
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= max_keywords:
            break

    return keywords[:max_keywords]


def understand_query(query: str, mode: str = "original", max_keywords: int = 8) -> Dict[str, object]:
    """
    Produce a lightweight, explainable query-understanding result.

    Supported modes:
    - original: use the original question directly
    - keywords: build a keyword-focused retrieval query
    """
    original_query = _normalize_whitespace(query)
    keywords = _extract_keywords(original_query, max_keywords=max_keywords)
    keyword_query = " ".join(keywords)

    if mode not in {"original", "keywords"}:
        raise ValueError(f"Unsupported query understanding mode: {mode}")

    retrieval_query = original_query if mode == "original" or not keyword_query else keyword_query

    return {
        "mode": mode,
        "original_query": original_query,
        "rewritten_query": retrieval_query,
        "keyword_query": keyword_query,
        "keywords": keywords,
    }
