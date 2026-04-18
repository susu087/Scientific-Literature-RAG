import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SYSTEM_GROUPS = [
    ("Inline-citation", "Broad", 20),
    ("Inline-citation", "Specific", 5),
    ("Inline-citation", "Specific", 20),
    ("Author-written", "Broad", 20),
    ("Author-written", "Specific", 5),
    ("Author-written", "Specific", 20),
    ("Avg.", "Broad", 20),
    ("Avg.", "Specific", 5),
    ("Avg.", "Specific", 20),
]


def load_json_or_jsonl(path: str) -> List[dict]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if file_path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def recall_at_k(relevant_ids: Sequence[int], retrieved_ids: Sequence[int], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def source_group(query_set: str) -> str:
    if query_set.startswith("inline_"):
        return "Inline-citation"
    if query_set.startswith("manual_"):
        return "Author-written"
    return "Unknown"


def specificity_group(specificity: int) -> str:
    if specificity == 0:
        return "Broad"
    if specificity == 1:
        return "Specific"
    return "Unknown"


def filter_items(items: Iterable[dict], source: str, specificity: str) -> List[dict]:
    filtered = []
    for item in items:
        item_source = source_group(str(item.get("query_set", "")))
        item_specificity = specificity_group(int(item.get("specificity", -1)))

        source_match = source == "Avg." or item_source == source
        specificity_match = item_specificity == specificity
        if source_match and specificity_match:
            filtered.append(item)
    return filtered


def compute_group_recall(items: Iterable[dict], k: int) -> Dict[str, float]:
    selected = list(items)
    if not selected:
        return {"count": 0, f"R@{k}": 0.0}

    score = 0.0
    for item in selected:
        score += recall_at_k(item.get("corpusids", []), item.get("retrieved", []), k)

    return {
        "count": len(selected),
        f"R@{k}": score / len(selected),
    }


def build_aligned_summary(items: Sequence[dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for source, specificity, k in SYSTEM_GROUPS:
        filtered = filter_items(items, source, specificity)
        metric = compute_group_recall(filtered, k)
        rows.append(
            {
                "source_group": source,
                "specificity": specificity,
                "metric": f"R@{k}",
                "count": metric["count"],
                "value": metric[f"R@{k}"],
            }
        )
    return rows


def print_single_system_table(label: str, rows: Sequence[Dict[str, object]]) -> None:
    print(f"\n{label}")
    print("=" * 72)
    print(f"{'Source':<18}{'Specificity':<12}{'Metric':<8}{'Count':>8}{'Value':>12}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['source_group']:<18}"
            f"{row['specificity']:<12}"
            f"{row['metric']:<8}"
            f"{int(row['count']):>8}"
            f"{float(row['value']):>12.4f}"
        )


def print_comparison_table(labels: Sequence[str], summaries: Sequence[Sequence[Dict[str, object]]]) -> None:
    if not summaries:
        return

    print("=" * (46 + 14 * len(labels)))
    header = f"{'Source':<18}{'Specificity':<12}{'Metric':<8}"
    for label in labels:
        header += f"{label:>14}"
    print(header)
    print("-" * (46 + 14 * len(labels)))

    for idx in range(len(summaries[0])):
        base_row = summaries[0][idx]
        line = (
            f"{base_row['source_group']:<18}"
            f"{base_row['specificity']:<12}"
            f"{base_row['metric']:<8}"
        )
        for summary in summaries:
            line += f"{float(summary[idx]['value']):>14.4f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LitSearch-paper-aligned retrieval metrics.")
    parser.add_argument("--results_files", nargs="+", required=True, help="One or more retrieval/reranking result files.")
    parser.add_argument("--labels", nargs="*", default=[], help="Optional labels for the result files.")
    args = parser.parse_args()

    labels = list(args.labels) if args.labels else [Path(path).stem for path in args.results_files]
    if len(labels) != len(args.results_files):
        raise ValueError("The number of labels must match the number of results files.")

    summaries = []
    for label, path in zip(labels, args.results_files):
        rows = build_aligned_summary(load_json_or_jsonl(path))
        summaries.append(rows)
        if len(args.results_files) == 1:
            print_single_system_table(label, rows)

    if len(args.results_files) > 1:
        print_comparison_table(labels, summaries)


if __name__ == "__main__":
    main()
