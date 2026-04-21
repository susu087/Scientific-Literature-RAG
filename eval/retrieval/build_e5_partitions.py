import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.cluster import KMeans


def load_raw_index(index_path: str) -> Dict[str, Any]:
    with open(index_path, "rb") as file:
        return pickle.load(file)


def build_partition_payload(index_data: Dict[str, Any], num_partitions: int, random_state: int) -> Dict[str, Any]:
    encoded_keys = np.asarray(index_data["encoded_keys"], dtype=np.float32)
    corpusids = [int(value) for value in index_data["values"]]

    model = KMeans(
        n_clusters=num_partitions,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(encoded_keys)

    partitions = []
    for partition_id in range(num_partitions):
        doc_indices = np.where(labels == partition_id)[0].astype(int).tolist()
        partitions.append(
            {
                "partition_id": partition_id,
                "size": len(doc_indices),
                "doc_indices": doc_indices,
                "corpusids": [corpusids[index] for index in doc_indices],
            }
        )

    centroids = model.cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    return {
        "index_name": index_data.get("index_name"),
        "index_type": index_data.get("index_type"),
        "num_partitions": num_partitions,
        "random_state": random_state,
        "total_documents": len(corpusids),
        "centroids": centroids.tolist(),
        "partitions": partitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KMeans partitions from an existing E5 index.")
    parser.add_argument("--index_name", type=str, default="LitSearch.title_abstract.e5")
    parser.add_argument("--index_root_dir", type=str, default="retrieval_indices")
    parser.add_argument("--num_partitions", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()

    index_path = os.path.join(args.index_root_dir, args.index_name)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"E5 index file does not exist: {index_path}")

    output_file = args.output_file
    if not output_file:
        output_file = os.path.join(
            "retrieval_partitions",
            f"{args.index_name}.kmeans_k{args.num_partitions}.json",
        )

    index_data = load_raw_index(index_path)
    payload = build_partition_payload(
        index_data=index_data,
        num_partitions=args.num_partitions,
        random_state=args.random_state,
    )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(payload, file)

    print(f"Loaded index: {index_path}")
    print(f"Total documents: {payload['total_documents']}")
    print(f"Built partitions: {args.num_partitions}")
    print(f"Saved partition file to: {output_file}")


if __name__ == "__main__":
    main()
