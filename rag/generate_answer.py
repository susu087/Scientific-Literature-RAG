"""Answer generation interface for the minimal RAG demo."""

from __future__ import annotations

import re
from typing import Dict, List

from rag.prompts import MOCK_ANSWER_TEMPLATE, build_answer_messages


def _first_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0].strip() if parts else text


def _clean_evidence_sentence(text: str) -> str:
    text = _first_sentence(text)
    text = re.sub(
        r"^(this paper|this study|we|the paper)\s+(studies|study|discuss|discusses|propose|proposes|present|presents)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text[:1].lower() + text[1:] if text else ""


def _generate_mock_answer(question: str, documents: List[Dict[str, object]]) -> Dict[str, object]:
    if not documents:
        return {
            "backend": "mock",
            "answer": "The retrieved context is empty, so I cannot provide a grounded answer.",
            "references": [],
        }

    lead_docs = documents[: min(3, len(documents))]
    evidence_parts = []
    references = []

    for doc in lead_docs:
        citation = str(doc["doc_id"])
        title = str(doc.get("title", "")).strip()
        abstract_sentence = _clean_evidence_sentence(str(doc.get("abstract", "")))
        if abstract_sentence:
            evidence_parts.append(f'"{title}" discusses {abstract_sentence} {citation}')
        else:
            evidence_parts.append(f'"{title}" appears highly relevant {citation}')
        references.append(
            {
                "doc_id": citation,
                "corpusid": doc.get("corpusid"),
                "title": title,
                "score": doc.get("score"),
            }
        )

    summary = " ; ".join(evidence_parts)
    citation_text = " ".join(ref["doc_id"] for ref in references)
    answer = MOCK_ANSWER_TEMPLATE.format(summary=summary, citations=citation_text).strip()

    return {
        "backend": "mock",
        "answer": answer,
        "references": references,
    }


def _generate_local_placeholder(question: str, context: str, documents: List[Dict[str, object]]) -> Dict[str, object]:
    mock_result = _generate_mock_answer(question, documents)
    mock_result["backend"] = "local"
    mock_result["messages"] = build_answer_messages(question, context)
    mock_result["answer"] = (
        "Local generation backend is not implemented yet. "
        "Using the mock grounded answer for now.\n\n"
        f'{mock_result["answer"]}'
    )
    return mock_result


def generate_answer(
    question: str,
    context: str,
    documents: List[Dict[str, object]],
    backend: str = "mock",
) -> Dict[str, object]:
    if backend == "mock":
        return _generate_mock_answer(question, documents)
    if backend == "local":
        return _generate_local_placeholder(question, context, documents)
    raise ValueError(f"Unsupported answer backend: {backend}")
