"""Thin wrapper around the existing E5 retrieval and local rerank logic."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import datasets
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from eval.retrieval.e5 import E5
from eval.retrieval.kv_store import TextType
from utils import utils


DEFAULT_INDEX_PATH = os.path.join("retrieval_indices", "LitSearch.title_abstract.e5")


def load_corpus_records(
    dataset_path: str = "princeton-nlp/LitSearch",
    local_fallback_path: str = os.path.join("data", "corpus_sample.json"),
) -> List[dict]:
    try:
        dataset = datasets.load_dataset(dataset_path, "corpus_clean", split="full")
        return [dict(item) for item in dataset]
    except Exception:
        if not os.path.exists(local_fallback_path):
            raise
        return utils.read_json(local_fallback_path, silent=True)


def build_corpus_lookup(records: Sequence[dict]) -> Dict[int, dict]:
    lookup = {}
    for record in records:
        corpusid = utils.get_clean_corpusid(record)
        lookup[corpusid] = record
    return lookup


class E5Retriever:
    """Reusable E5 retrieval wrapper with score-aware output."""

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        dataset_path: str = "princeton-nlp/LitSearch",
        local_fallback_path: str = os.path.join("data", "corpus_sample.json"),
    ) -> None:
        self.index_path = index_path
        self.dataset_path = dataset_path
        self.local_fallback_path = local_fallback_path
        self.corpus_records = load_corpus_records(dataset_path, local_fallback_path)
        self.corpus_lookup = build_corpus_lookup(self.corpus_records)
        self.index = self._load_or_build_index()

    def _build_in_memory_index(self) -> E5:
        index = E5(index_name="LitSearch.title_abstract")
        key_value_pairs = {
            utils.get_clean_title_abstract(record): utils.get_clean_corpusid(record)
            for record in self.corpus_records
        }
        index.create_index(key_value_pairs)
        return index

    def _load_or_build_index(self) -> E5:
        if os.path.exists(self.index_path):
            return E5(index_name="LitSearch.title_abstract").load(self.index_path)
        return self._build_in_memory_index()

    def retrieve(self, query_text: str, top_k: int = 100) -> List[Dict[str, object]]:
        encoded_query = self.index._encode(query_text, TextType.QUERY)
        similarities = cosine_similarity([encoded_query], self.index.encoded_keys)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            corpusid = self.index.values[idx]
            record = self.corpus_lookup.get(corpusid, {})
            results.append(
                {
                    "doc_id": f"[D{rank}]",
                    "rank": rank,
                    "score": float(similarities[idx]),
                    "retrieval_score": float(similarities[idx]),
                    "corpusid": corpusid,
                    "title": record.get("title", ""),
                    "abstract": record.get("abstract", ""),
                    "content": utils.get_clean_title_abstract(record) if record else "",
                }
            )
        return results


class LocalReranker:
    """
    Local reranker aligned with eval/reranking/rerank_local.py.

    We intentionally keep this as a thin copy inside rag/ because the original
    script mixes CLI code at module scope and is not safe to import directly.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 4,
        max_length: int = 512,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(
        self,
        query_text: str,
        documents: Sequence[Dict[str, object]],
        max_k: int = 50,
    ) -> List[Dict[str, object]]:
        rerank_docs = list(documents[:max_k])
        if not rerank_docs:
            return []

        pairs = [(query_text, str(doc.get("content", ""))) for doc in rerank_docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        sorted_indices = sorted(
            range(len(rerank_docs)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )

        reranked = []
        for new_rank, original_idx in enumerate(sorted_indices, start=1):
            doc = dict(rerank_docs[original_idx])
            doc["doc_id"] = f"[D{new_rank}]"
            doc["rank"] = new_rank
            doc["score"] = float(scores[original_idx])
            doc["rerank_score"] = float(scores[original_idx])
            reranked.append(doc)

        for doc in documents[max_k:]:
            remainder = dict(doc)
            remainder["doc_id"] = f'[D{len(reranked) + 1}]'
            remainder["rank"] = len(reranked) + 1
            reranked.append(remainder)

        return reranked


def run_retrieval_and_rerank(
    query_text: str,
    retriever: E5Retriever,
    top_k: int = 100,
    rerank_top_k: int = 50,
    reranker: Optional[LocalReranker] = None,
) -> Dict[str, List[Dict[str, object]]]:
    retrieved_docs = retriever.retrieve(query_text, top_k=top_k)
    reranker = reranker or LocalReranker()
    reranked_docs = reranker.rerank(query_text, retrieved_docs, max_k=rerank_top_k)
    return {
        "retrieved_docs": retrieved_docs,
        "reranked_docs": reranked_docs,
    }
