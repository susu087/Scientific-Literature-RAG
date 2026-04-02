"""Context building utilities for the minimal RAG demo."""

from __future__ import annotations

from typing import Dict, List, Tuple


def truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_plain_document(doc: Dict[str, object], per_doc_max_chars: int) -> str:
    title = truncate_text(str(doc.get("title", "")), max(80, per_doc_max_chars // 4))
    abstract = truncate_text(str(doc.get("abstract", "")), per_doc_max_chars)
    return f'{doc["doc_id"]} {title}\n{abstract}'


def _format_structured_document(doc: Dict[str, object], per_doc_max_chars: int) -> str:
    title = truncate_text(str(doc.get("title", "")), max(80, per_doc_max_chars // 4))
    abstract = truncate_text(str(doc.get("abstract", "")), per_doc_max_chars)
    score = doc.get("score")
    score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"
    return (
        f'{doc["doc_id"]}\n'
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        f'Rank: {doc.get("rank", "N/A")}\n'
        f"Score: {score_text}"
    )


def build_context(
    documents: List[Dict[str, object]],
    strategy: str = "structured",
    top_n: int = 5,
    per_doc_max_chars: int = 1200,
) -> Tuple[str, List[Dict[str, object]]]:
    """
    Build prompt context from retrieved documents.

    Strategies:
    - plain: title + abstract concatenation
    - structured: explicit fields with rank and score
    """
    selected_docs = documents[:top_n]
    if strategy not in {"plain", "structured"}:
        raise ValueError(f"Unsupported context strategy: {strategy}")

    if strategy == "plain":
        chunks = [_format_plain_document(doc, per_doc_max_chars) for doc in selected_docs]
    else:
        chunks = [_format_structured_document(doc, per_doc_max_chars) for doc in selected_docs]

    return "\n\n".join(chunks), selected_docs
