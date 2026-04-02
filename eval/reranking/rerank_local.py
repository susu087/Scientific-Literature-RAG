import os
import copy
import json
import argparse
import datasets
import torch

from tqdm import tqdm
from sentence_transformers import CrossEncoder

from utils import utils


def local_rerank_pipeline(
    model: CrossEncoder,
    item: dict,
    rank_start: int,
    rank_end: int,
    batch_size: int,
) -> dict:
    docs = item["documents"][rank_start:rank_end]

    if len(docs) == 0:
        return item

    pairs = [(item["query"], doc["content"]) for doc in docs]

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    sorted_indices = sorted(
        range(len(docs)),
        key=lambda i: float(scores[i]),
        reverse=True
    )

    reranked_docs = []
    for new_rank, idx in enumerate(sorted_indices, start=1):
        doc = copy.deepcopy(docs[idx])
        doc["rerank_score"] = float(scores[idx])
        doc["rerank_rank"] = new_rank
        reranked_docs.append(doc)

    item["documents"][rank_start:rank_end] = reranked_docs
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_file", type=str, required=True)

    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_k", default=20, type=int, help="Max number of retrieved documents to rerank")
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir", type=str, required=False, default="results/reranking/")
    parser.add_argument("--dataset_path", required=False, default="princeton-nlp/LitSearch")
    args = parser.parse_args()

    corpus_data = datasets.load_dataset(
    path=args.dataset_path,
    name="corpus_clean",
    split="full"
    )

    retrieval_results = utils.read_json(args.retrieval_results_file)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = CrossEncoder(
        args.reranker_model,
        device=device,
        max_length=args.max_length,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    input_name = os.path.basename(args.retrieval_results_file)

    if input_name.endswith(".jsonl"):
        output_name = input_name.replace(
            ".jsonl",
            f".reranked_top{args.max_k}.jsonl"
        )
    elif input_name.endswith(".json"):
        output_name = input_name.replace(
            ".json",
            f".reranked_top{args.max_k}.json"
        )
    else:
        output_name = input_name + f".reranked_top{args.max_k}.jsonl"

    output_file = os.path.join(args.output_dir, output_name)

    index_type = os.path.basename(args.retrieval_results_file).split(".")[1]

    if index_type == "title_abstract":
        corpusid_to_text = {
            utils.get_clean_corpusid(item): utils.get_clean_title_abstract(item)
            for item in corpus_data
        }
    elif index_type == "full_paper":
        corpusid_to_text = {
            utils.get_clean_corpusid(item): utils.get_clean_full_paper(item)
            for item in corpus_data
        }
    else:
        raise ValueError(f"Invalid index type: {index_type}")

    # put retrieval results into format required by reranking pipeline
    reranking_inputs = []
for query_info in retrieval_results:
    topk_corpusids = query_info["retrieved"][:args.max_k]

    reranking_inputs.append({
        "query": query_info["query"],
        "documents": [
            {
                "content": corpusid_to_text[retrieved_corpusid],
                "corpusid": retrieved_corpusid
            }
            for retrieved_corpusid in topk_corpusids
        ]
    })

if os.path.exists(output_file):
    reranking_outputs = utils.read_json(output_file)
else:
    reranking_outputs = copy.deepcopy(retrieval_results)
    utils.write_json(reranking_outputs, output_file, silent=True)

for item_idx, item in enumerate(tqdm(reranking_inputs, desc=f"Reranking top{args.max_k}", dynamic_ncols=True)):
    if "pre_reranked" not in reranking_outputs[item_idx]:
        reranked_item = local_rerank_pipeline(
            model=model,
            item=item,
            rank_start=0,
            rank_end=len(item["documents"]),
            batch_size=args.batch_size,
        )

        original_retrieved = copy.deepcopy(reranking_outputs[item_idx]["retrieved"])
        reranked_topk = [document["corpusid"] for document in reranked_item["documents"]]
        remaining = original_retrieved[args.max_k:]

        reranking_outputs[item_idx]["pre_reranked"] = original_retrieved
        reranking_outputs[item_idx]["retrieved"] = reranked_topk + remaining

        # 仍然保留逐条保存，避免中途中断丢结果
        utils.write_json(reranking_outputs, output_file, silent=True)

print(f"Saved reranked results to: {output_file}")