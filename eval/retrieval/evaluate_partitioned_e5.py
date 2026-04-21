import argparse
import json
import os
from typing import Dict, List, Sequence

import datasets
import numpy as np
from tqdm import tqdm

from eval.retrieval.e5 import E5
from eval.retrieval.kv_store import TextType
from utils import utils


def load_partition_file(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def cosine_scores(query_embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ query_embedding, dtype=np.float32)


def select_partitions(query_embedding: np.ndarray, centroids: np.ndarray, top_partitions: int) -> List[int]:
    scores = cosine_scores(query_embedding, centroids)
    top_n = min(top_partitions, len(scores))
    return scores.argsort()[-top_n:][::-1].astype(int).tolist()


def collect_candidate_indices(partitions: Sequence[Dict], selected_partition_ids: Sequence[int]) -> List[int]:
    candidate_indices = []
    for partition_id in selected_partition_ids:
        candidate_indices.extend(partitions[partition_id]["doc_indices"])
    return list(dict.fromkeys(candidate_indices))


def partitioned_query(
    index: E5,
    query_text: str,
    centroids: np.ndarray,
    partitions: Sequence[Dict],
    top_partitions: int,
    top_k: int,
) -> Dict[str, object]:
    query_embedding = np.asarray(index._encode(query_text, TextType.QUERY), dtype=np.float32)
    selected_partition_ids = select_partitions(query_embedding, centroids, top_partitions)
    candidate_indices = collect_candidate_indices(partitions, selected_partition_ids)

    candidate_embeddings = np.asarray(index.encoded_keys, dtype=np.float32)[candidate_indices]
    scores = cosine_scores(query_embedding, candidate_embeddings)
    top_n = min(top_k, len(candidate_indices))
    local_top_indices = scores.argsort()[-top_n:][::-1]

    retrieved = [int(index.values[candidate_indices[int(local_idx)]]) for local_idx in local_top_indices]
    return {
        "retrieved": retrieved,
        "selected_partitions": selected_partition_ids,
        "num_candidates": len(candidate_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate partition-aware E5 retrieval on LitSearch.")
    parser.add_argument("--partition_file", type=str, required=True)
    parser.add_argument("--index_name", type=str, default="LitSearch.title_abstract.e5")
    parser.add_argument("--index_root_dir", type=str, default="retrieval_indices")
    parser.add_argument("--dataset_path", type=str, default="princeton-nlp/LitSearch")
    parser.add_argument("--top_partitions", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()

    partition_payload = load_partition_file(args.partition_file)
    centroids = np.asarray(partition_payload["centroids"], dtype=np.float32)
    partitions = partition_payload["partitions"]

    index_path = os.path.join(args.index_root_dir, args.index_name)
    index = E5(None).load(index_path)

    query_set = [query for query in datasets.load_dataset(args.dataset_path, "query", split="full")]
    for query in tqdm(query_set, desc="Partitioned E5 retrieval", dynamic_ncols=True):
        result = partitioned_query(
            index=index,
            query_text=query["query"],
            centroids=centroids,
            partitions=partitions,
            top_partitions=args.top_partitions,
            top_k=args.top_k,
        )
        query["retrieved"] = result["retrieved"]
        query["partition_info"] = {
            "partition_file": args.partition_file,
            "top_partitions": args.top_partitions,
            "selected_partitions": result["selected_partitions"],
            "num_candidates": result["num_candidates"],
        }

    output_file = args.output_file
    if not output_file:
        base_name = os.path.basename(args.partition_file).replace(".json", "")
        output_file = os.path.join("results", "retrieval", f"{base_name}.jsonl")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    utils.write_json(query_set, output_file)


if __name__ == "__main__":
    main()
