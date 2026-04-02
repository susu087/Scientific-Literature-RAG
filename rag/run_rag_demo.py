"""Command-line entrypoint for the minimal end-to-end RAG prototype."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.build_context import build_context
from rag.generate_answer import generate_answer
from rag.query_understanding import understand_query


def _format_query_understanding(result: Dict[str, object]) -> str:
    return (
        f'Mode: {result["mode"]}\n'
        f'Original Query: {result["original_query"]}\n'
        f'Rewritten Query: {result["rewritten_query"]}\n'
        f'Keywords: {", ".join(result["keywords"])}'
    )


def _format_documents(title: str, documents: List[Dict[str, object]], limit: int = 5) -> str:
    lines = [title]
    for doc in documents[:limit]:
        score = doc.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"
        lines.append(
            f'{doc["doc_id"]} corpusid={doc.get("corpusid")} score={score_text} '
            f'title={doc.get("title", "")}'
        )
    return "\n".join(lines)


def _format_references(references: List[Dict[str, object]]) -> str:
    if not references:
        return "References:\nNone"
    lines = ["References:"]
    for ref in references:
        score = ref.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"
        lines.append(
            f'{ref["doc_id"]} corpusid={ref.get("corpusid")} score={score_text} title={ref.get("title", "")}'
        )
    return "\n".join(lines)


def format_rag_result(result: Dict[str, Any], show_context_chars: int = 2000) -> str:
    lines = [
        "Question:",
        str(result["question"]),
        "",
        "Query Understanding:",
        _format_query_understanding(result["query_understanding"]),
        "",
        _format_documents("Top Retrieved Papers:", result["retrieved_docs"]),
        "",
        _format_documents("Top Reranked Papers:", result["reranked_docs"]),
        "",
        "Context:",
    ]

    context_text = str(result["context"])
    preview = context_text[:show_context_chars]
    if len(context_text) > show_context_chars:
        preview += "..."
    lines.append(preview)
    lines.extend(
        [
            "",
            "Answer:",
            str(result["answer"]),
            "",
            _format_references(result["references"]),
        ]
    )
    return "\n".join(lines)


def run_rag_case(
    question: str,
    query_mode: str = "keywords",
    context_strategy: str = "structured",
    answer_backend: str = "mock",
    top_k: int = 100,
    rerank_top_k: int = 50,
    context_top_n: int = 5,
    per_doc_max_chars: int = 1200,
    retrieval_index_path: str = os.path.join("retrieval_indices", "LitSearch.title_abstract.e5"),
    dataset_path: str = "princeton-nlp/LitSearch",
    local_corpus_path: str = os.path.join("data", "corpus_sample.json"),
    reranker_model: str = "BAAI/bge-reranker-base",
    reranker_batch_size: int = 4,
    reranker_max_length: int = 512,
    device: str = "cuda",
) -> Dict[str, Any]:
    from rag.retrieval_pipeline import E5Retriever, LocalReranker, run_retrieval_and_rerank

    query_result = understand_query(question, mode=query_mode)
    retrieval_query = str(query_result["rewritten_query"])

    retriever = E5Retriever(
        index_path=retrieval_index_path,
        dataset_path=dataset_path,
        local_fallback_path=local_corpus_path,
        device=device,
    )
    reranker = LocalReranker(
        model_name=reranker_model,
        batch_size=reranker_batch_size,
        max_length=reranker_max_length,
        device=device,
    )
    pipeline_result = run_retrieval_and_rerank(
        query_text=retrieval_query,
        retriever=retriever,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        reranker=reranker,
    )

    context_text, context_docs = build_context(
        pipeline_result["reranked_docs"],
        strategy=context_strategy,
        top_n=context_top_n,
        per_doc_max_chars=per_doc_max_chars,
    )
    answer_result = generate_answer(
        question=question,
        context=context_text,
        documents=context_docs,
        backend=answer_backend,
    )

    return {
        "question": question,
        "query_understanding": query_result,
        "retrieval_query": retrieval_query,
        "retrieved_docs": pipeline_result["retrieved_docs"],
        "reranked_docs": pipeline_result["reranked_docs"],
        "context": context_text,
        "context_docs": context_docs,
        "answer": answer_result["answer"],
        "references": answer_result["references"],
        "answer_backend": answer_result["backend"],
        "config": {
            "query_mode": query_mode,
            "context_strategy": context_strategy,
            "answer_backend": answer_backend,
            "top_k": top_k,
            "rerank_top_k": rerank_top_k,
            "context_top_n": context_top_n,
            "per_doc_max_chars": per_doc_max_chars,
            "retrieval_index_path": retrieval_index_path,
            "dataset_path": dataset_path,
            "local_corpus_path": local_corpus_path,
            "reranker_model": reranker_model,
            "reranker_batch_size": reranker_batch_size,
            "reranker_max_length": reranker_max_length,
            "device": device,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal end-to-end RAG demo for LitSearch.")
    parser.add_argument("--question", type=str, required=True, help="User question for the RAG demo.")
    parser.add_argument("--query_mode", type=str, default="keywords", choices=["original", "keywords"])
    parser.add_argument("--context_strategy", type=str, default="structured", choices=["plain", "structured"])
    parser.add_argument("--answer_backend", type=str, default="mock", choices=["mock", "local"])
    parser.add_argument("--top_k", type=int, default=100, help="Number of E5 retrieved documents.")
    parser.add_argument("--rerank_top_k", type=int, default=50, help="Rerank depth. Keep this at 50 for the main setup.")
    parser.add_argument("--context_top_n", type=int, default=5, help="Number of reranked docs used to build context.")
    parser.add_argument("--per_doc_max_chars", type=int, default=1200)
    parser.add_argument("--retrieval_index_path", type=str, default=os.path.join("retrieval_indices", "LitSearch.title_abstract.e5"))
    parser.add_argument("--dataset_path", type=str, default="princeton-nlp/LitSearch")
    parser.add_argument("--local_corpus_path", type=str, default=os.path.join("data", "corpus_sample.json"))
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--reranker_batch_size", type=int, default=4)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show_context_chars", type=int, default=2000)
    parser.add_argument("--save_json", type=str, default="")
    args = parser.parse_args()

    result = run_rag_case(
        question=args.question,
        query_mode=args.query_mode,
        context_strategy=args.context_strategy,
        answer_backend=args.answer_backend,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        per_doc_max_chars=args.per_doc_max_chars,
        context_top_n=args.context_top_n,
        retrieval_index_path=args.retrieval_index_path,
        dataset_path=args.dataset_path,
        local_corpus_path=args.local_corpus_path,
        reranker_model=args.reranker_model,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        device=args.device,
    )
    print(format_rag_result(result, show_context_chars=args.show_context_chars))

    if args.save_json:
        save_path = os.path.abspath(args.save_json)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
