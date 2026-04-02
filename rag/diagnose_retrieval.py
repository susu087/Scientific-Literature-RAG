"""Minimal diagnostics for the current RAG retrieval pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import datasets


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.retrieval_pipeline import DEFAULT_INDEX_PATH, RagE5
from utils import utils


DEFAULT_Q1 = (
    "Are there any research papers on methods to compress large-scale language models "
    "using knowledge distillation techniques?"
)


def inspect_corpus_source(
    dataset_path: str,
    local_fallback_path: str,
) -> Tuple[List[dict], bool, str]:
    """Replicate the current fallback logic and report which source is used."""
    try:
        dataset = datasets.load_dataset(dataset_path, "corpus_clean", split="full")
        records = [dict(item) for item in dataset]
        return records, False, dataset_path
    except Exception:
        records = utils.read_json(local_fallback_path, silent=True)
        return records, True, local_fallback_path


def resolve_q1_question(test_questions_path: str) -> str:
    if os.path.exists(test_questions_path):
        with open(test_questions_path, "r", encoding="utf-8") as file:
            items = json.load(file)
        for item in items:
            case_id = str(item.get("id") or item.get("case_id") or "")
            if case_id == "Q1":
                return str(item.get("question", DEFAULT_Q1))
    return DEFAULT_Q1


def load_index(index_path: str, device: str) -> RagE5:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file does not exist: {index_path}")
    return RagE5(index_name="LitSearch.title_abstract", device=device).load(index_path)


def build_corpus_lookup(records: List[dict]) -> Dict[int, dict]:
    return {utils.get_clean_corpusid(record): record for record in records}


def retrieve_topk(
    index: RagE5,
    corpus_lookup: Dict[int, dict],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    from eval.retrieval.kv_store import TextType
    from sklearn.metrics.pairwise import cosine_similarity

    encoded_query = index._encode(query, TextType.QUERY)
    similarities = cosine_similarity([encoded_query], index.encoded_keys)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        corpusid = index.values[idx]
        record = corpus_lookup.get(corpusid, {})
        results.append(
            {
                "rank": rank,
                "score": float(similarities[idx]),
                "corpusid": corpusid,
                "title": record.get("title", ""),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose current retrieval/index loading behavior.")
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset_path", type=str, default="princeton-nlp/LitSearch")
    parser.add_argument("--local_corpus_path", type=str, default=os.path.join("data", "corpus_sample.json"))
    parser.add_argument("--test_questions_path", type=str, default="test_questions.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--show_index_values", type=int, default=10)
    args = parser.parse_args()

    index_path = os.path.abspath(args.index_path)
    local_corpus_path = os.path.abspath(args.local_corpus_path)
    test_questions_path = os.path.abspath(args.test_questions_path)

    records, fallback_used, corpus_source = inspect_corpus_source(args.dataset_path, local_corpus_path)
    corpus_lookup = build_corpus_lookup(records)
    index = load_index(index_path, device=args.device)
    q1_question = resolve_q1_question(test_questions_path)
    top_results = retrieve_topk(index, corpus_lookup, q1_question, top_k=args.top_k)

    print("=== Retrieval Diagnostics ===")
    print(f"Loaded index file: {index_path}")
    print(f"Index file exists: {os.path.exists(index_path)}")
    print(f"Fallback used corpus_sample.json: {fallback_used}")
    print(f"Corpus source: {corpus_source}")
    print(f"Current corpus size: {len(records)}")
    print(f"Index size: {len(index.values)}")
    print()

    print(f"First {args.show_index_values} corpusids in index.values:")
    for idx, corpusid in enumerate(index.values[: args.show_index_values], start=1):
        print(f"{idx}. {corpusid}")
    print()

    print("Q1 query:")
    print(q1_question)
    print()

    print(f"Q1 top{args.top_k} retrieval results:")
    for item in top_results:
        print(
            f'[D{item["rank"]}] corpusid={item["corpusid"]} '
            f'score={item["score"]:.6f} title={item["title"]}'
        )


if __name__ == "__main__":
    main()
