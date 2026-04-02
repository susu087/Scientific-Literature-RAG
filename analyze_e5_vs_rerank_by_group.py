import json
from collections import defaultdict


# ====== 这里改成你的两个结果文件 ======
e5_file = r"results/retrieval/LitSearch.title_abstract.e5.jsonl"
rerank_file = r"results/reranking/LitSearch.title_abstract.e5.reranked.jsonl"

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


def print_metric_block(name, base_metrics, rerank_metrics):
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
    print(f"样本数: E5={base_metrics['count']}, Rerank={rerank_metrics['count']}")
    print("-" * 62)
    print(f"{'Metric':<12}{'E5':>12}{'E5+Rerank':>14}{'Delta':>12}")
    print("-" * 62)

    for metric in metric_names:
        base_val = base_metrics[metric]
        rerank_val = rerank_metrics[metric]
        delta = rerank_val - base_val
        print(f"{metric:<12}{base_val:>12.4f}{rerank_val:>14.4f}{delta:>12.4f}")


def main():
    e5_data = load_json_or_jsonl(e5_file)
    rerank_data = load_json_or_jsonl(rerank_file)

    if not e5_data:
        print("原始 E5 文件为空或读取失败。")
        return
    if not rerank_data:
        print("Rerank 文件为空或读取失败。")
        return

    # 总体指标
    e5_overall = compute_metrics(e5_data)
    rerank_overall = compute_metrics(rerank_data)

    print("E5 vs E5+Rerank metrics on LitSearch")
    print("=" * 62)
    print_metric_block("overall", e5_overall, rerank_overall)

    # 分组指标
    e5_groups = group_by_query_set(e5_data)
    rerank_groups = group_by_query_set(rerank_data)

    all_group_names = sorted(set(e5_groups.keys()) | set(rerank_groups.keys()))

    print("\n\nBy query_set")
    print("=" * 62)

    for group_name in all_group_names:
        e5_metrics = compute_metrics(e5_groups.get(group_name, []))
        rerank_metrics = compute_metrics(rerank_groups.get(group_name, []))
        print_metric_block(group_name, e5_metrics, rerank_metrics)


if __name__ == "__main__":
    main()