"""Thin wrapper around the existing E5 retrieval and local rerank logic."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import datasets
import numpy as np
import sentence_transformers
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from eval.retrieval.kv_store import KVStore, TextType
from utils import utils


DEFAULT_INDEX_PATH = os.path.join("retrieval_indices", "LitSearch.title_abstract.e5")
DEFAULT_BM25_INDEX_PATH = os.path.join("retrieval_indices", "LitSearch.title_abstract.bm25")


class RagE5(KVStore):
    """Device-aware E5 index wrapper for the RAG prototype."""

    def __init__(
        self,
        index_name: str,
        model_path: str = "intfloat/e5-large-v2",
        device: str = "cuda",
    ) -> None:
        super().__init__(index_name, "e5")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.model_path = model_path
        self.device = device
        self._model = sentence_transformers.SentenceTransformer(
            model_path,
            device=device,
            cache_folder=os.environ.get("HF_HOME"),
        )

    def _format_text(self, text: str, type: TextType) -> str:
        if type == TextType.KEY:
            return "passage: " + text
        if type == TextType.QUERY:
            return "query: " + text
        raise ValueError("Invalid TextType")

    def _encode_batch(self, texts: List[str], type: TextType, show_progress_bar: bool = True) -> List[np.ndarray]:
        texts = [self._format_text(text, type) for text in texts]
        return self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        ).astype(np.float16)

    def _query(self, encoded_query: np.ndarray, n: int) -> List[int]:
        cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        return top_indices

    def load(self, file_path: str) -> "RagE5":
        if len(self.keys) > 0:
            raise ValueError("Index is not empty. Clear it before loading.")
        with open(file_path, "rb") as file:
            pickle_data = pickle.load(file)
        for key, value in pickle_data.items():
            setattr(self, key, value)
        self._model = sentence_transformers.SentenceTransformer(
            self.model_path,
            device=self.device,
            cache_folder=os.environ.get("HF_HOME"),
        )
        return self


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
        device: str = "cuda",
    ) -> None:
        self.index_path = index_path
        self.dataset_path = dataset_path
        self.local_fallback_path = local_fallback_path
        self.device = device
        self.corpus_records = load_corpus_records(dataset_path, local_fallback_path)
        self.corpus_lookup = build_corpus_lookup(self.corpus_records)
        self.index = self._load_or_build_index()

    def _build_in_memory_index(self) -> RagE5:
        index = RagE5(index_name="LitSearch.title_abstract", device=self.device)
        key_value_pairs = {
            utils.get_clean_title_abstract(record): utils.get_clean_corpusid(record)
            for record in self.corpus_records
        }
        index.create_index(key_value_pairs)
        return index

    def _load_or_build_index(self) -> RagE5:
        if os.path.exists(self.index_path):
            return RagE5(index_name="LitSearch.title_abstract", device=self.device).load(self.index_path)
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


class BM25Retriever:
    """BM25 retrieval wrapper used by the final hybrid RAG pipeline."""

    def __init__(
        self,
        index_path: str = DEFAULT_BM25_INDEX_PATH,
        corpus_lookup: Optional[Dict[int, dict]] = None,
    ) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"BM25 index not found: {index_path}")
        from eval.retrieval.bm25 import BM25

        self.index_path = index_path
        self.index = BM25(None).load(index_path)
        self.corpus_lookup = corpus_lookup or {}

    def retrieve(self, query_text: str, top_k: int = 100) -> List[Dict[str, object]]:
        encoded_query = self.index._encode(query_text, TextType.QUERY)
        scores = self.index.index.get_scores(encoded_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            corpusid = int(self.index.values[int(idx)])
            record = self.corpus_lookup.get(corpusid, {})
            results.append(
                {
                    "doc_id": f"[D{rank}]",
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "bm25_score": float(scores[int(idx)]),
                    "corpusid": corpusid,
                    "title": record.get("title", ""),
                    "abstract": record.get("abstract", ""),
                    "content": utils.get_clean_title_abstract(record) if record else "",
                }
            )
        return results


def _doc_key(doc: Dict[str, object]) -> int:
    return int(doc["corpusid"])


def _reset_doc_ranks(documents: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked = []
    for rank, doc in enumerate(documents, start=1):
        item = dict(doc)
        item["doc_id"] = f"[D{rank}]"
        item["rank"] = rank
        ranked.append(item)
    return ranked


def weighted_rrf_fuse_docs(
    ranked_lists: Sequence[Sequence[Dict[str, object]]],
    weights: Sequence[float],
    top_k: int = 200,
    rrf_k: int = 60,
) -> List[Dict[str, object]]:
    score_by_id: Dict[int, float] = {}
    best_doc_by_id: Dict[int, Dict[str, object]] = {}
    best_rank_by_id: Dict[int, int] = {}

    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs, start=1):
            corpusid = _doc_key(doc)
            score_by_id[corpusid] = score_by_id.get(corpusid, 0.0) + weight / (rrf_k + rank)
            best_rank_by_id[corpusid] = min(best_rank_by_id.get(corpusid, rank), rank)
            if corpusid not in best_doc_by_id:
                best_doc_by_id[corpusid] = dict(doc)
            else:
                merged = dict(best_doc_by_id[corpusid])
                merged.update({k: v for k, v in doc.items() if k.endswith("_score")})
                best_doc_by_id[corpusid] = merged

    sorted_ids = sorted(
        score_by_id,
        key=lambda corpusid: (-score_by_id[corpusid], best_rank_by_id.get(corpusid, 10**9), corpusid),
    )[:top_k]

    fused = []
    for corpusid in sorted_ids:
        doc = dict(best_doc_by_id[corpusid])
        doc["score"] = float(score_by_id[corpusid])
        doc["rrf_score"] = float(score_by_id[corpusid])
        fused.append(doc)
    return _reset_doc_ranks(fused)


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


def reason_to_rank_lite_documents(
    query_text: str,
    documents: Sequence[Dict[str, object]],
    top_n: int = 10,
    model: str = "gpt-4o-mini",
    base_url: str = "https://aihubmix.com/v1",
    direct_weight: float = 0.55,
    comparative_weight: float = 0.35,
    prior_weight: float = 0.10,
) -> List[Dict[str, object]]:
    if top_n <= 0 or not documents:
        return list(documents)

    from eval.reranking.reason_to_rank_lite import reason_rerank_one_query

    corpusids = [_doc_key(doc) for doc in documents]
    corpusid_to_text = {_doc_key(doc): str(doc.get("content", "")) for doc in documents}
    result = reason_rerank_one_query(
        query=query_text,
        corpusids=corpusids,
        corpusid_to_text=corpusid_to_text,
        top_n=top_n,
        max_doc_chars=1200,
        model=model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=3500,
        timeout=120,
        max_retries=3,
        direct_weight=direct_weight,
        comparative_weight=comparative_weight,
        prior_weight=prior_weight,
    )
    doc_by_id = {_doc_key(doc): dict(doc) for doc in documents}
    reranked = []
    for corpusid in result["reranked"]:
        if corpusid not in doc_by_id:
            continue
        doc = dict(doc_by_id[corpusid])
        reason_item = result["reason_items"].get(str(corpusid), {})
        if reason_item:
            doc["reason_score"] = reason_item.get("reason_score")
            doc["reason"] = reason_item.get("reason")
        reranked.append(doc)
    return _reset_doc_ranks(reranked)


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


def run_final_hybrid_reason_pipeline(
    query_text: str,
    e5_retriever: E5Retriever,
    bm25_index_path: str = DEFAULT_BM25_INDEX_PATH,
    top_k: int = 100,
    rerank_top_k: int = 50,
    rrf_k: int = 60,
    e5_weight: float = 1.0,
    bm25_weight: float = 1.0,
    reranker: Optional[LocalReranker] = None,
    reason_top_n: int = 10,
    reason_model: str = "gpt-4o-mini",
    reason_base_url: str = "https://aihubmix.com/v1",
) -> Dict[str, List[Dict[str, object]]]:
    e5_docs = e5_retriever.retrieve(query_text, top_k=top_k)
    bm25_retriever = BM25Retriever(index_path=bm25_index_path, corpus_lookup=e5_retriever.corpus_lookup)
    bm25_docs = bm25_retriever.retrieve(query_text, top_k=top_k)
    fused_docs = weighted_rrf_fuse_docs(
        ranked_lists=[e5_docs, bm25_docs],
        weights=[e5_weight, bm25_weight],
        top_k=top_k,
        rrf_k=rrf_k,
    )

    reranker = reranker or LocalReranker()
    local_reranked_docs = reranker.rerank(query_text, fused_docs, max_k=rerank_top_k)
    reason_reranked_docs = reason_to_rank_lite_documents(
        query_text=query_text,
        documents=local_reranked_docs,
        top_n=reason_top_n,
        model=reason_model,
        base_url=reason_base_url,
    )

    return {
        "retrieved_docs": fused_docs,
        "local_reranked_docs": local_reranked_docs,
        "reranked_docs": reason_reranked_docs,
    }
