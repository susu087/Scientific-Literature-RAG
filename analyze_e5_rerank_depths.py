import json
from collections import defaultdict

E5_FILE = r"results/retrieval/LitSearch.title_abstract.e5.jsonl"
RERANK20_FILE = r"results/reranking/LitSearch.title_abstract.e5.reranked_top20.jsonl"
RERANK50_FILE = r"results/reranking/LitSearch.title_abstract.e5.reranked_top50.jsonl"
RERANK100_FILE = r"results/reranking/LitSearch.title_abstract.e5.reranked_top100.jsonl"


def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if path.endswith(".jsonl"):
        data = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data

    return json.loads(text)


def recall_at_k(relevant_ids, retrieved_ids, k):
    if not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    hits = len(set(relevant_ids) & set(topk))
    return hits / len(set(relevant_ids))


def hit_at_k(relevant_ids, retrieved_ids, k):
    topk = retrieved_ids[:k]
    return 1.0 if len(set(relevant_ids) & set(topk)) > 0 else 0.0


def mrr_at_k(relevant_ids, retrieved_ids, k):
    topk = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(topk, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_metrics(items):
    r10 = r50 = r100 = 0.0
    h10 = h100 = 0.0
    m10 = m100 = 0.0
    valid_count = 0

    for item in items:
        relevant_ids = item.get("corpusids", [])
        retrieved_ids = item.get("retrieved", [])

        if not isinstance(relevant_ids, list) or not isinstance(retrieved_ids, list):
            continue

        valid_count += 1
        r10 += recall_at_k(relevant_ids, retrieved_ids, 10)
        r50 += recall_at_k(relevant_ids, retrieved_ids, 50)
        r100 += recall_at_k(relevant_ids, retrieved_ids, 100)
        h10 += hit_at_k(relevant_ids, retrieved_ids, 10)
        h100 += hit_at_k(relevant_ids, retrieved_ids, 100)
        m10 += mrr_at_k(relevant_ids, retrieved_ids, 10)
        m100 += mrr_at_k(relevant_ids, retrieved_ids, 100)

    if valid_count == 0:
        return {
            "count": 0,
            "Recall@10": 0.0,
            "Recall@50": 0.0,
            "Recall@100": 0.0,
            "Hit@10": 0.0,
            "Hit@100": 0.0,
            "MRR@10": 0.0,
            "MRR@100": 0.0,
        }

    return {
        "count": valid_count,
        "Recall@10": r10 / valid_count,
        "Recall@50": r50 / valid_count,
        "Recall@100": r100 / valid_count,
        "Hit@10": h10 / valid_count,
        "Hit@100": h100 / valid_count,
        "MRR@10": m10 / valid_count,
        "MRR@100": m100 / valid_count,
    }


def group_by_query_set(data):
    groups = defaultdict(list)
    for item in data:
        query_set = item.get("query_set", "unknown")
        groups[query_set].append(item)
    return groups


def print_block(name, e5_metrics, r20_metrics, r50_metrics, r100_metrics):
    metric_names = [
        "Recall@10",
        "Recall@50",
        "Recall@100",
        "Hit@10",
        "Hit@100",
        "MRR@10",
        "MRR@100",
    ]

    print(f"\n[{name}]")
    print(
        f"样本数: E5={e5_metrics['count']}, "
        f"R20={r20_metrics['count']}, "
        f"R50={r50_metrics['count']}, "
        f"R100={r100_metrics['count']}"
    )
    print("-" * 118)
    print(
        f"{'Metric':<12}"
        f"{'E5':>10}"
        f"{'R20':>12}"
        f"{'R50':>12}"
        f"{'R100':>12}"
        f"{'Δ20':>10}"
        f"{'Δ50':>10}"
        f"{'Δ100':>10}"
    )
    print("-" * 118)

    for metric in metric_names:
        e5_val = e5_metrics[metric]
        r20_val = r20_metrics[metric]
        r50_val = r50_metrics[metric]
        r100_val = r100_metrics[metric]

        print(
            f"{metric:<12}"
            f"{e5_val:>10.4f}"
            f"{r20_val:>12.4f}"
            f"{r50_val:>12.4f}"
            f"{r100_val:>12.4f}"
            f"{(r20_val - e5_val):>10.4f}"
            f"{(r50_val - e5_val):>10.4f}"
            f"{(r100_val - e5_val):>10.4f}"
        )


def main():
    e5_data = load_json_or_jsonl(E5_FILE)
    r20_data = load_json_or_jsonl(RERANK20_FILE)
    r50_data = load_json_or_jsonl(RERANK50_FILE)
    r100_data = load_json_or_jsonl(RERANK100_FILE)

    print("E5 vs Rerank(top20/top50/top100)")
    print("=" * 118)

    print_block(
        "overall",
        compute_metrics(e5_data),
        compute_metrics(r20_data),
        compute_metrics(r50_data),
        compute_metrics(r100_data),
    )

    e5_groups = group_by_query_set(e5_data)
    r20_groups = group_by_query_set(r20_data)
    r50_groups = group_by_query_set(r50_data)
    r100_groups = group_by_query_set(r100_data)

    all_groups = sorted(
        set(e5_groups.keys()) |
        set(r20_groups.keys()) |
        set(r50_groups.keys()) |
        set(r100_groups.keys())
    )

    print("\nBy query_set")
    print("=" * 118)

    for group_name in all_groups:
        print_block(
            group_name,
            compute_metrics(e5_groups.get(group_name, [])),
            compute_metrics(r20_groups.get(group_name, [])),
            compute_metrics(r50_groups.get(group_name, [])),
            compute_metrics(r100_groups.get(group_name, [])),
        )


if __name__ == "__main__":
    main()