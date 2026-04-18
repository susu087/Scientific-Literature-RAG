import argparse
from pathlib import Path

from litsearch_aligned_metrics import build_aligned_summary, load_json_or_jsonl, print_single_system_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LitSearch-paper-aligned metrics for one result file.")
    parser.add_argument(
        "--results_file",
        type=str,
        default=r"results/reranking/LitSearch.title_abstract.e5.reranked_top100.jsonl",
        help="Retrieval or reranking results file.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="System",
        help="Display name for the system.",
    )
    args = parser.parse_args()

    rows = build_aligned_summary(load_json_or_jsonl(args.results_file))
    label = args.label if args.label != "System" else Path(args.results_file).stem
    print_single_system_table(label, rows)


if __name__ == "__main__":
    main()
