"""Command-line entrypoint for the minimal end-to-end RAG prototype."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.build_context import build_context
from rag.generate_answer import generate_answer
from rag.query_understanding import understand_query
from rag.retrieval_pipeline import E5Retriever, LocalReranker, run_retrieval_and_rerank


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal end-to-end RAG demo for LitSearch.")
    parser.add_argument("--question", type=str, required=True, help="User question for the RAG demo.")
    parser.add_argument("--query_mode", type=str, default="original", choices=["original", "keywords"])
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
    args = parser.parse_args()

    query_result = understand_query(args.question, mode=args.query_mode)
    retrieval_query = str(query_result["rewritten_query"])

    retriever = E5Retriever(
        index_path=args.retrieval_index_path,
        dataset_path=args.dataset_path,
        local_fallback_path=args.local_corpus_path,
    )
    reranker = LocalReranker(
        model_name=args.reranker_model,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
        device=args.device,
    )
    pipeline_result = run_retrieval_and_rerank(
        query_text=retrieval_query,
        retriever=retriever,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        reranker=reranker,
    )

    context_text, context_docs = build_context(
        pipeline_result["reranked_docs"],
        strategy=args.context_strategy,
        top_n=args.context_top_n,
        per_doc_max_chars=args.per_doc_max_chars,
    )
    answer_result = generate_answer(
        question=args.question,
        context=context_text,
        documents=context_docs,
        backend=args.answer_backend,
    )

    print("Question:")
    print(args.question)
    print()

    print("Query Understanding:")
    print(_format_query_understanding(query_result))
    print()

    print(_format_documents("Top Retrieved Papers:", pipeline_result["retrieved_docs"]))
    print()

    print(_format_documents("Top Reranked Papers:", pipeline_result["reranked_docs"]))
    print()

    print("Context:")
    preview = context_text[: args.show_context_chars]
    if len(context_text) > args.show_context_chars:
        preview += "..."
    print(preview)
    print()

    print("Answer:")
    print(answer_result["answer"])
    print()

    print(_format_references(answer_result["references"]))


if __name__ == "__main__":
    main()
