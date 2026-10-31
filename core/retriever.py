"""Configurable dense, BM25, and reciprocal-rank hybrid retrieval."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable

from core.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    RETRIEVAL_CANDIDATE_K,
    RERANKER_MODEL,
    TOP_K,
)
from core.embeddings import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    build_embedding_backend,
    close_embedding_backend,
    lexical_index_name,
    validate_embeddings,
)
from core.ingest import get_existing_collection
from core.lexical import LexicalIndex
from core.logging_utils import log_event

logger = logging.getLogger(__name__)
_RERANKER_LOCK = threading.Lock()
_RERANKERS = {}
_RERANKER_ERRORS = set()
_RRF_K = 60
_MAX_QUERY_BATCH_SIZE = 32
MAX_RETRIEVAL_REQUEST_QUESTIONS = int(os.getenv("MAX_RETRIEVAL_EVAL_QUESTIONS", "1000") or 1000)


class RetrievalError(RuntimeError):
    """Safe user-facing error for search failures."""


def _load_reranker(model_name: str):
    with _RERANKER_LOCK:
        if model_name in _RERANKER_ERRORS:
            return None
        if model_name in _RERANKERS:
            return _RERANKERS[model_name]
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, max_length=512)
            _RERANKERS[model_name] = model
            logger.info("Loaded retrieval reranker %s.", model_name)
            return model
        except Exception as exc:
            _RERANKER_ERRORS.add(model_name)
            log_event(
                logger,
                logging.WARNING,
                "reranker_unavailable",
                error_type=type(exc).__name__,
            )
            return None


def dense_ranking(dense_hits: list[dict], lexical_hits: list[dict]) -> list[dict]:
    return [dict(hit, retrieval_method="dense") for hit in dense_hits]


def bm25_ranking(dense_hits: list[dict], lexical_hits: list[dict]) -> list[dict]:
    return [dict(hit, retrieval_method="bm25") for hit in lexical_hits]


def hybrid_ranking(dense_hits: list[dict], lexical_hits: list[dict]) -> list[dict]:
    candidates = {item["chunk_id"]: dict(item) for item in dense_hits}
    for rank, hit in enumerate(lexical_hits, start=1):
        candidate = candidates.get(hit["chunk_id"])
        if candidate is None:
            candidate = dict(hit)
            candidate["vector_rank"] = None
            candidate["distance"] = None
            candidate["hybrid_score"] = 0.0
            candidates[hit["chunk_id"]] = candidate
            candidate["retrieval_method"] = "bm25"
        else:
            candidate["retrieval_method"] = "hybrid"
        candidate["lexical_rank"] = rank
        candidate["lexical_score"] = hit.get("lexical_score")
        candidate["hybrid_score"] = float(candidate.get("hybrid_score") or 0.0) + 1 / (_RRF_K + rank)
    return sorted(
        candidates.values(),
        key=lambda item: (
            -float(item.get("hybrid_score") or 0.0),
            item.get("distance") if item.get("distance") is not None else float("inf"),
        ),
    )


RETRIEVER_REGISTRY: dict[str, Callable[[list[dict], list[dict]], list[dict]]] = {
    "dense": dense_ranking,
    "bm25": bm25_ranking,
    "hybrid": hybrid_ranking,
}


def register_retriever(
    name: str,
    ranking_function: Callable[[list[dict], list[dict]], list[dict]],
) -> None:
    """Register a search strategy that consumes dense and/or BM25 candidates."""
    normalized = (name or "").strip().lower()
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError("Retriever names may contain letters, digits, and underscores.")
    if normalized in RETRIEVER_REGISTRY:
        raise ValueError(f"Retriever '{normalized}' is already registered.")
    if not callable(ranking_function):
        raise TypeError("A retriever ranking function must be callable.")
    RETRIEVER_REGISTRY[normalized] = ranking_function


class Retriever:
    def __init__(
        self,
        top_k: int = TOP_K,
        *,
        chroma_dir=None,
        collection_name=None,
        candidate_k: int = RETRIEVAL_CANDIDATE_K,
        method: str = "hybrid",
        embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
        embedding_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        api_key: str | None = None,
        use_reranker: bool = True,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.top_k = min(100, max(1, int(top_k)))
        self.candidate_k = min(500, max(self.top_k, int(candidate_k)))
        self.method = self._validate_method(method)
        self.chroma_dir = Path(chroma_dir) if chroma_dir is not None else CHROMA_DIR
        self.collection_name = collection_name or COLLECTION_NAME
        self.embedding_provider = (embedding_provider or DEFAULT_EMBEDDING_PROVIDER).strip().lower()
        self.embedding_model = (embedding_model or DEFAULT_LOCAL_EMBEDDING_MODEL).strip()
        self._api_key = api_key or ""
        self.use_reranker = bool(use_reranker)
        self.reranker_model = (reranker_model or RERANKER_MODEL).strip()
        if not self.reranker_model or len(self.reranker_model) > 200:
            raise ValueError("Enter a valid reranker model ID (1–200 characters).")
        self.collection = get_existing_collection(
            chroma_dir=self.chroma_dir,
            collection_name=self.collection_name,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
        )
        self.lexical_index = LexicalIndex(
            self.chroma_dir
            / lexical_index_name(
                self.embedding_provider,
                self.embedding_model,
                collection_base_name=self.collection_name,
            )
        )
        self._lexical_error_logged = False
        self._embedding_backend = None
        self.last_method_errors = {}

    @staticmethod
    def _validate_method(method: str) -> str:
        value = (method or "").strip().lower()
        if value not in RETRIEVER_REGISTRY:
            available = ", ".join(RETRIEVER_REGISTRY)
            raise ValueError(f"Unknown retriever '{value}'. Choose one of: {available}.")
        return value

    def _get_embedding_backend(self):
        if self._embedding_backend is None:
            self._embedding_backend = build_embedding_backend(
                self.embedding_provider,
                self.embedding_model,
                self._api_key,
            )
        return self._embedding_backend

    def is_ready(self) -> bool:
        return self.collection is not None and int(self.collection.count()) > 0

    @staticmethod
    def warm_up_reranker(model_name: str = RERANKER_MODEL) -> bool:
        """Load the local cross-encoder before the first user query."""
        return _load_reranker(model_name) is not None

    @staticmethod
    def _query_variants(query: str):
        """Retained for compatibility; retrieval searches the exact query."""
        q = (query or "").strip()
        return [q] if q else []

    def _dense_hits(self, query_vectors: list[list[float]], candidate_count: int) -> list[list[dict]]:
        if not query_vectors:
            return []
        try:
            response = self.collection.query(
                query_embeddings=query_vectors,
                n_results=candidate_count,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RetrievalError(
                "Dense search failed. Check that the selected embedding model matches the indexed documents."
            ) from None

        all_hits = []
        ids_by_query = response.get("ids") or []
        docs_by_query = response.get("documents") or []
        metas_by_query = response.get("metadatas") or []
        distances_by_query = response.get("distances") or []
        for query_index, ids in enumerate(ids_by_query):
            docs = docs_by_query[query_index] if query_index < len(docs_by_query) else []
            metas = metas_by_query[query_index] if query_index < len(metas_by_query) else []
            distances = distances_by_query[query_index] if query_index < len(distances_by_query) else []
            hits = []
            for rank, (chunk_id, text, metadata, distance) in enumerate(
                zip(ids or [], docs or [], metas or [], distances or []), start=1
            ):
                text = str(text or "").strip()
                if not text:
                    continue
                metadata = metadata or {}
                hits.append({
                    "chunk_id": str(chunk_id),
                    "text": text,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "location": metadata.get("location"),
                    "file_id": metadata.get("file_id"),
                    "distance": float(distance),
                    "vector_rank": rank,
                    "lexical_rank": None,
                    "hybrid_score": 1.0 / (_RRF_K + rank),
                    "retrieval_method": "dense",
                })
            all_hits.append(hits)
        return all_hits

    def _lexical_hits(self, queries: list[str], candidate_count: int) -> list[list[dict]]:
        if not queries:
            return []
        try:
            self.lexical_index.ensure_synced(self.collection)
            return [self.lexical_index.query(query, candidate_count) for query in queries]
        except Exception as exc:
            if not self._lexical_error_logged:
                log_event(
                    logger,
                    logging.WARNING,
                    "bm25_index_unavailable",
                    error_type=type(exc).__name__,
                )
                self._lexical_error_logged = True
            raise RetrievalError(
                "BM25 search is unavailable for this index. Re-index the documents and retry."
            ) from None

    @staticmethod
    def _normalize_lexical_hits(hits: list[dict]) -> list[dict]:
        normalized = []
        for rank, hit in enumerate(hits, start=1):
            text = str(hit.get("text") or "").strip()
            if not text:
                continue
            normalized.append({
                "chunk_id": str(hit.get("chunk_id")),
                "text": text,
                "source": hit.get("source"),
                "page": hit.get("page"),
                "location": hit.get("location"),
                "file_id": hit.get("file_id"),
                "distance": None,
                "vector_rank": None,
                "lexical_rank": rank,
                "lexical_score": float(hit.get("lexical_score") or 0.0),
                "hybrid_score": 1.0 / (_RRF_K + rank),
                "retrieval_method": "bm25",
            })
        return normalized

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not self.use_reranker or not candidates:
            return candidates
        reranker = _load_reranker(self.reranker_model)
        if reranker is None:
            return candidates
        try:
            pairs = [(query, item["text"]) for item in candidates]
            with _RERANKER_LOCK:
                scores = reranker.predict(
                    pairs,
                    batch_size=16,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)
            return sorted(
                candidates,
                key=lambda item: (
                    -item["rerank_score"],
                    -float(item.get("hybrid_score") or 0.0),
                    item.get("distance") if item.get("distance") is not None else float("inf"),
                ),
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "reranking_failed",
                error_type=type(exc).__name__,
            )
            return candidates

    def retrieve_many(
        self,
        queries,
        *,
        methods=None,
        batch_size: int = _MAX_QUERY_BATCH_SIZE,
    ) -> dict[str, list[list[dict]]]:
        """Retrieve a batch of queries, sharing embeddings and vector/lexical calls.

        This keeps dense and hybrid comparison on the same query vectors and
        bounds both provider batches and reranker work.
        """
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, (list, tuple)):
            raise ValueError("Retrieval queries must be supplied as a list of text values.")
        if len(queries) > MAX_RETRIEVAL_REQUEST_QUESTIONS:
            raise ValueError(
                f"A retrieval request can contain at most {MAX_RETRIEVAL_REQUEST_QUESTIONS} questions."
            )
        clean_queries = []
        for query in queries or []:
            if not isinstance(query, str) or len(query) > 8000:
                raise ValueError("Each retrieval question must be text of 8,000 characters or less.")
            clean_queries.append(query.strip())
        selected = [self._validate_method(method) for method in (methods or [self.method])]
        selected = list(dict.fromkeys(selected))
        self.last_method_errors = {}
        if not selected:
            raise ValueError("Select at least one retriever.")
        if not 1 <= int(batch_size) <= _MAX_QUERY_BATCH_SIZE:
            raise ValueError(f"Query batches must contain 1–{_MAX_QUERY_BATCH_SIZE} questions.")
        results = {method: [] for method in selected}
        if not self.is_ready():
            return results

        for start in range(0, len(clean_queries), int(batch_size)):
            batch_queries = clean_queries[start : start + int(batch_size)]
            if not batch_queries:
                continue
            collection_size = int(self.collection.count())
            if collection_size <= 0:
                for method in selected:
                    results[method].extend([[] for _ in batch_queries])
                continue
            candidate_count = min(self.candidate_k, collection_size)

            dense_by_query = [[] for _ in batch_queries]
            needs_vector_search = any(method != "bm25" for method in selected)
            needs_lexical_search = any(method != "dense" for method in selected)
            if needs_vector_search:
                nonempty_indexes = [i for i, query in enumerate(batch_queries) if query]
                if nonempty_indexes:
                    try:
                        vectors = validate_embeddings(
                            self._get_embedding_backend().embed_queries(
                                [batch_queries[i] for i in nonempty_indexes]
                            ),
                            len(nonempty_indexes),
                        )
                        found = self._dense_hits(vectors, candidate_count)
                        for index, hits in zip(nonempty_indexes, found):
                            dense_by_query[index] = hits
                    except Exception as exc:
                        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "Dense retrieval failed."
                        for method in selected:
                            if method != "bm25":
                                self.last_method_errors[method] = message[:300]

            lexical_by_query = [[] for _ in batch_queries]
            if needs_lexical_search:
                nonempty_indexes = [i for i, query in enumerate(batch_queries) if query]
                if nonempty_indexes:
                    try:
                        found = self._lexical_hits(
                            [batch_queries[i] for i in nonempty_indexes],
                            candidate_count,
                        )
                        for index, hits in zip(nonempty_indexes, found):
                            lexical_by_query[index] = self._normalize_lexical_hits(hits)
                    except RetrievalError as exc:
                        if "bm25" in selected:
                            self.last_method_errors["bm25"] = str(exc)
                        if "hybrid" in selected:
                            self.last_method_errors["hybrid"] = (
                                "BM25 index unavailable; hybrid retrieval fell back to dense ranking."
                            )
                        for method in selected:
                            if method not in {"dense", "bm25", "hybrid"}:
                                self.last_method_errors[method] = str(exc)

            for method in selected:
                ranking_fn = RETRIEVER_REGISTRY[method]
                ranked_queries = []
                for query, dense_hits, lexical_hits in zip(
                    batch_queries, dense_by_query, lexical_by_query
                ):
                    if not query:
                        ranked_queries.append([])
                        continue
                    if method in self.last_method_errors and method != "hybrid":
                        ranked_queries.append([])
                        continue
                    if method == "hybrid" and not dense_hits and not lexical_hits and "hybrid" in self.last_method_errors:
                        ranked_queries.append([])
                        continue
                    candidates = ranking_fn(dense_hits, lexical_hits)
                    ranked_queries.append(self._rerank(query, candidates)[: self.top_k])
                results[method].extend(ranked_queries)
        return results

    def retrieve(self, query: str):
        """Return top-ranked chunks with source and retrieval score metadata."""
        results = self.retrieve_many([query], methods=[self.method])[self.method][0]
        error = self.last_method_errors.get(self.method)
        if error and self.method != "hybrid":
            raise RetrievalError(error)
        if error and self.method == "hybrid" and not results:
            raise RetrievalError(error)
        return results

    def close(self) -> None:
        """Drop in-memory provider references when a Streamlit workspace is cleared."""
        close_embedding_backend(self._embedding_backend)
        self._embedding_backend = None
        self._api_key = ""
