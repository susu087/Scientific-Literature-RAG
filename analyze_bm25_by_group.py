import json
from collections import defaultdict

file_path = r"results/retrieval/LitSearch.title_abstract.e5.jsonl"

def recall_at_k(gold, retrieved, k):
    gold_set = set(gold)
    ret_set = set(retrieved[:k])
    if not gold_set:
        return 0.0
    return len(gold_set & ret_set) / len(gold_set)

def hit_at_k(gold, retrieved, k):
    gold_set = set(gold)
    ret_set = set(retrieved[:k])
    return 1.0 if gold_set & ret_set else 0.0

def mrr_at_k(gold, retrieved, k):
    gold_set = set(gold)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0

def first_hit_rank(gold, retrieved):
    gold_set = set(gold)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold_set:
            return rank
    return None

groups = defaultdict(list)
all_items = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        all_items.append(item)
        groups[item["query_set"]].append(item)

print("=" * 60)
print("E5 grouped evaluation on LitSearch")
print("=" * 60)

for group_name, items in groups.items():
    r10, r50, r100 = [], [], []
    h10, h100 = [], []
    m10, m100 = [], []

    for item in items:
        gold = item["corpusids"]
        retrieved = item["retrieved"]

        r10.append(recall_at_k(gold, retrieved, 10))
        r50.append(recall_at_k(gold, retrieved, 50))
        r100.append(recall_at_k(gold, retrieved, 100))

        h10.append(hit_at_k(gold, retrieved, 10))
        h100.append(hit_at_k(gold, retrieved, 100))

        m10.append(mrr_at_k(gold, retrieved, 10))
        m100.append(mrr_at_k(gold, retrieved, 100))

    print(f"\n[{group_name}]")
    print(f"Count      : {len(items)}")
    print(f"Recall@10  : {sum(r10)/len(r10):.4f}")
    print(f"Recall@50  : {sum(r50)/len(r50):.4f}")
    print(f"Recall@100 : {sum(r100)/len(r100):.4f}")
    print(f"Hit@10     : {sum(h10)/len(h10):.4f}")
    print(f"Hit@100    : {sum(h100)/len(h100):.4f}")
    print(f"MRR@10     : {sum(m10)/len(m10):.4f}")
    print(f"MRR@100    : {sum(m100)/len(m100):.4f}")

# 成功样例：按首个命中排名升序
success_cases = []
failure_cases = []

for item in all_items:
    gold = item["corpusids"]
    retrieved = item["retrieved"]
    rank = first_hit_rank(gold, retrieved)

    if rank is not None:
        success_cases.append({
            "query_set": item["query_set"],
            "query": item["query"],
            "gold": gold,
            "first_hit_rank": rank
        })
    else:
        failure_cases.append({
            "query_set": item["query_set"],
            "query": item["query"],
            "gold": gold,
            "top10": retrieved[:10]
        })

success_cases = sorted(success_cases, key=lambda x: x["first_hit_rank"])
failure_cases = failure_cases[:5]

print("\n" + "=" * 60)
print("Top 5 successful cases")
print("=" * 60)
for i, case in enumerate(success_cases[:5], start=1):
    print(f"\nCase {i}")
    print(f"query_set      : {case['query_set']}")
    print(f"first_hit_rank : {case['first_hit_rank']}")
    print(f"gold           : {case['gold']}")
    print(f"query          : {case['query']}")

print("\n" + "=" * 60)
print("Top 5 failure cases")
print("=" * 60)
for i, case in enumerate(failure_cases, start=1):
    print(f"\nCase {i}")
    print(f"query_set : {case['query_set']}")
    print(f"gold      : {case['gold']}")
    print(f"top10     : {case['top10']}")
    print(f"query     : {case['query']}")