from litsearch_aligned_metrics import (
    build_aligned_summary,
    load_json_or_jsonl,
    print_comparison_table,
)


E5_FILE = r"results/retrieval/LitSearch.title_abstract.e5.jsonl"
RERANK_FILE = r"results/reranking/LitSearch.title_abstract.e5.reranked_top50.jsonl"


def main() -> None:
    e5_rows = build_aligned_summary(load_json_or_jsonl(E5_FILE))
    rerank_rows = build_aligned_summary(load_json_or_jsonl(RERANK_FILE))
    print_comparison_table(["E5", "E5+Rerank(top50)"], [e5_rows, rerank_rows])


if __name__ == "__main__":
    main()
