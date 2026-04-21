"""Answer generation interface for the minimal RAG demo."""

from __future__ import annotations

import re
import json
import os
import time
import urllib.error
import urllib.request
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


def _call_openai_compatible_chat(
    messages: List[Dict[str, str]],
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> str:
    api_key = os.environ.get("AIHUBMIX_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("AIHUBMIX_API_KEY or OPENAI_API_KEY is not set.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            return response_data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            if attempt == max_retries - 1:
                raise RuntimeError(f"API HTTP error {exc.code}: {error_body}") from exc
        except Exception:
            if attempt == max_retries - 1:
                raise
        time.sleep(2 * (attempt + 1))

    raise RuntimeError("API request failed after retries.")


def _generate_api_answer(
    question: str,
    context: str,
    documents: List[Dict[str, object]],
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> Dict[str, object]:
    answer = _call_openai_compatible_chat(
        messages=build_answer_messages(question, context),
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )

    references = [
        {
            "doc_id": str(doc.get("doc_id")),
            "corpusid": doc.get("corpusid"),
            "title": doc.get("title", ""),
            "score": doc.get("score"),
        }
        for doc in documents
    ]
    return {
        "backend": "api",
        "answer": answer,
        "references": references,
        "messages": build_answer_messages(question, context),
        "model": model,
        "base_url": base_url,
    }


def generate_answer(
    question: str,
    context: str,
    documents: List[Dict[str, object]],
    backend: str = "mock",
    model: str = "gpt-4o-mini",
    base_url: str = "https://aihubmix.com/v1",
    temperature: float = 0.0,
    max_tokens: int = 800,
    timeout: int = 120,
    max_retries: int = 3,
) -> Dict[str, object]:
    if backend == "mock":
        return _generate_mock_answer(question, documents)
    if backend == "local":
        return _generate_local_placeholder(question, context, documents)
    if backend in {"api", "openai", "aihubmix"}:
        return _generate_api_answer(
            question=question,
            context=context,
            documents=documents,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported answer backend: {backend}")
