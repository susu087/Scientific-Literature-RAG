from litsearch_aligned_metrics import (
    build_aligned_summary,
    load_json_or_jsonl,
    print_comparison_table,
)


E5_FILE = r"results/retrieval/LitSearch.title_abstract.e5.jsonl"
RERANK20_FILE = r"results/reranking/LitSearch.title_abstract.e5.reranked_top20.jsonl"
RERANK50_FILE = r"results/reranking/LitSearch.title_abstract.e5.reranked_top50.jsonl"
RERANK100_FILE = r"results/reranking/LitSearch.title_abstract.e5.reranked_top100.jsonl"


def main() -> None:
    summaries = [
        build_aligned_summary(load_json_or_jsonl(E5_FILE)),
        build_aligned_summary(load_json_or_jsonl(RERANK20_FILE)),
        build_aligned_summary(load_json_or_jsonl(RERANK50_FILE)),
        build_aligned_summary(load_json_or_jsonl(RERANK100_FILE)),
    ]
    print_comparison_table(
        ["E5", "Rerank(top20)", "Rerank(top50)", "Rerank(top100)"],
        summaries,
    )


if __name__ == "__main__":
    main()
