from litsearch_aligned_metrics import build_aligned_summary, load_json_or_jsonl, print_single_system_table


RESULTS_FILE = r"results/retrieval/LitSearch.title_abstract.bm25.jsonl"


def main() -> None:
    rows = build_aligned_summary(load_json_or_jsonl(RESULTS_FILE))
    print_single_system_table("BM25", rows)


if __name__ == "__main__":
    main()
