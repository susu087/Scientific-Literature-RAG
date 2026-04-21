from litsearch_aligned_metrics import (
    build_aligned_summary,
    load_json_or_jsonl,
    print_comparison_table,
)


RESULT_FILES = [
    "results/retrieval/LitSearch.title_abstract.bm25.jsonl",
    "results/retrieval/LitSearch.title_abstract.e5.jsonl",
    "results/reranking/LitSearch.title_abstract.e5.reranked_top50.jsonl",
    "results/reranking/LitSearch.title_abstract.e5.partition_k10_top3.reranked_top50.jsonl",
]

LABELS = [
    "BM25",
    "E5",
    "E5+Rerank(top50)",
    "E5+Partition+Rerank(top50)",
]


def main() -> None:
    summaries = [build_aligned_summary(load_json_or_jsonl(path)) for path in RESULT_FILES]
    print_comparison_table(LABELS, summaries)


if __name__ == "__main__":
    main()
