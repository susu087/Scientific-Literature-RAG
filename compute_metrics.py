import json

file_path = r"results/reranking/LitSearch.title_abstract.e5.reranked_top100.jsonl"

def recall_at_k(gold, retrieved, k):
    gold_set = set(gold)
    ret_set = set(retrieved[:k])
    if len(gold_set) == 0:
        return 0.0
    return len(gold_set & ret_set) / len(gold_set)

def hit_at_k(gold, retrieved, k):
    gold_set = set(gold)
    ret_set = set(retrieved[:k])
    return 1.0 if len(gold_set & ret_set) > 0 else 0.0

def mrr_at_k(gold, retrieved, k):
    gold_set = set(gold)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0

recall_10 = []
recall_50 = []
recall_100 = []
hit_10 = []
hit_100 = []
mrr_10 = []
mrr_100 = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        gold = item["corpusids"]
        retrieved = item["retrieved"]

        recall_10.append(recall_at_k(gold, retrieved, 10))
        recall_50.append(recall_at_k(gold, retrieved, 50))
        recall_100.append(recall_at_k(gold, retrieved, 100))

        hit_10.append(hit_at_k(gold, retrieved, 10))
        hit_100.append(hit_at_k(gold, retrieved, 100))

        mrr_10.append(mrr_at_k(gold, retrieved, 10))
        mrr_100.append(mrr_at_k(gold, retrieved, 100))

print("E5 metrics on LitSearch")
print(f"Recall@10 : {sum(recall_10)/len(recall_10):.4f}")
print(f"Recall@50 : {sum(recall_50)/len(recall_50):.4f}")
print(f"Recall@100: {sum(recall_100)/len(recall_100):.4f}")
print(f"Hit@10    : {sum(hit_10)/len(hit_10):.4f}")
print(f"Hit@100   : {sum(hit_100)/len(hit_100):.4f}")
print(f"MRR@10    : {sum(mrr_10)/len(mrr_10):.4f}")
print(f"MRR@100   : {sum(mrr_100)/len(mrr_100):.4f}")