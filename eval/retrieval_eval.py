"""Fast, label-aware retrieval metrics for fair search-strategy comparison."""

from __future__ import annotations

import math

from core.retriever import RETRIEVER_REGISTRY
from core.golden import validate_golden_dataset

METRICS = ("Precision@K", "Recall@K", "MRR", "nDCG@K")
MAX_RETRIEVAL_EVAL_QUESTIONS = 500


def _references(item: dict) -> dict:
    ids = item.get("relevant_chunk_ids") or []
    docs = item.get("relevant_documents") or []
    if not isinstance(ids, list):
        ids = [ids]
    if not isinstance(docs, list):
        docs = [docs]
    source = item.get("source")
    page = item.get("page")
    source_file_id = item.get("source_file_id")
    location = item.get("source_reference") or item.get("location")
    sources = item.get("relevant_sources") or []
    locations = item.get("relevant_locations") or []
    if isinstance(sources, str):
        sources = [sources]
    if isinstance(locations, str):
        locations = [locations]
    if source:
        sources = [*sources, source]
    if location:
        locations = [*locations, location]
    for reference in docs:
        if isinstance(reference, str):
            if reference not in sources and reference not in locations and reference not in ids:
                sources.append(reference)
        elif isinstance(reference, dict):
            if reference.get("chunk_id"):
                ids.append(reference["chunk_id"])
            if reference.get("source"):
                sources.append(reference["source"])
            if reference.get("file_id"):
                source_file_id = reference["file_id"]
            if reference.get("location"):
                locations.append(reference["location"])
            if reference.get("page") is not None:
                page = reference["page"]
    ids = list(dict.fromkeys(str(value) for value in ids if value))
    sources = list(dict.fromkeys(str(value) for value in sources if value))
    locations = list(dict.fromkeys(str(value) for value in locations if value))
    return {
        "ids": ids,
        "sources": sources,
        "locations": locations,
        "page": page,
        "file_id": source_file_id,
    }


def _matches(candidate: dict, references: dict) -> bool:
    if references["file_id"] and candidate.get("file_id") != references["file_id"]:
        return False
    if candidate.get("chunk_id") in references["ids"]:
        return True
    if references["file_id"] and not (
        references["ids"] or references["sources"] or references["locations"]
    ):
        return True
    source = str(candidate.get("source") or "")
    location = str(candidate.get("location") or "")
    page = candidate.get("page")
    page_matches = references["page"] is None or str(page) == str(references["page"])
    source_matches = not references["sources"] or source in references["sources"]
    location_matches = not references["locations"] or location in references["locations"]
    if references["sources"] and references["locations"]:
        return source_matches and location_matches
    if references["sources"]:
        return source_matches and page_matches
    if references["locations"]:
        return location_matches and (not references["page"] or page_matches)
    return False


def _relevant_count(retriever, references: dict) -> int:
    if not (
        references["ids"] or references["sources"] or references["locations"]
        or references["file_id"]
    ):
        return 0
    if len(references["sources"]) <= 1 and len(references["locations"]) <= 1:
        count = retriever.lexical_index.count_matching(
            source=references["sources"][0] if references["sources"] else None,
            page=references["page"],
            location=references["locations"][0] if references["locations"] else None,
            chunk_ids=references["ids"],
            file_id=references["file_id"],
        )
    else:
        count = len(references["ids"])
        for source in references["sources"]:
            count += retriever.lexical_index.count_matching(
                source=source, page=references["page"], file_id=references["file_id"]
            )
        for location in references["locations"]:
            count += retriever.lexical_index.count_matching(
                location=location, file_id=references["file_id"]
            )
    return max(1, count)


def score_retrieval(
    item: dict,
    retrieved: list[dict],
    *,
    retriever,
    k: int,
) -> dict:
    references = _references(item)
    if not (
        references["ids"] or references["sources"] or references["locations"]
        or references["file_id"]
    ):
        return {name: None for name in METRICS} | {"label_available": False}

    ranked = list(retrieved or [])[:k]
    relevance = [1 if _matches(candidate, references) else 0 for candidate in ranked]
    hits = sum(relevance)
    relevant_total = _relevant_count(retriever, references)
    first_rank = next((index for index, relevant in enumerate(relevance, start=1) if relevant), None)
    precision = hits / k if k else 0.0
    recall = min(1.0, hits / relevant_total) if relevant_total else 0.0
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(relevant / math.log2(index + 1) for index, relevant in enumerate(relevance, start=1))
    ideal_hits = min(relevant_total, k)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        "Precision@K": precision,
        "Recall@K": recall,
        "MRR": mrr,
        "nDCG@K": ndcg,
        "label_available": True,
        "relevant_results": hits,
        "relevant_documents": relevant_total,
        "first_relevant_rank": first_rank,
    }


def run_retrieval_comparison(
    golden_data: list[dict],
    retriever,
    *,
    methods: list[str],
    k: int | None = None,
    batch_size: int = 32,
) -> dict:
    """Score chosen retrievers over the exact same ordered dataset and K."""
    selected = list(dict.fromkeys(str(method).strip().lower() for method in methods))
    if not selected:
        raise ValueError("Select at least one retriever to compare.")
    unknown = [method for method in selected if method not in RETRIEVER_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown retriever: {unknown[0]}.")
    golden_data = validate_golden_dataset(golden_data)
    if len(golden_data) > MAX_RETRIEVAL_EVAL_QUESTIONS:
        raise ValueError(f"A comparison can contain at most {MAX_RETRIEVAL_EVAL_QUESTIONS} questions.")
    selected_k = int(k if k is not None else retriever.top_k)
    if not 1 <= selected_k <= 100:
        raise ValueError("K must be between 1 and 100.")
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Retrieval comparison batch size must be a whole number.") from exc
    if not 1 <= batch_size <= 32:
        raise ValueError("Retrieval comparison batches must contain 1–32 questions.")

    summaries = {
        method: {"metrics": {}, "queries": [], "errors": [], "label_coverage": 0}
        for method in selected
    }
    for start in range(0, len(golden_data), batch_size):
        batch = golden_data[start : start + batch_size]
        questions = [item.get("input", "") for item in batch]
        try:
            retrieved_by_method = retriever.retrieve_many(questions, methods=selected, batch_size=batch_size)
        except Exception as exc:
            safe_message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "Retrieval failed."
            for method in selected:
                summaries[method]["errors"].append({"type": type(exc).__name__, "message": safe_message[:300]})
                summaries[method]["queries"].extend([
                    {"input": str(item.get("input", ""))[:200], "error": safe_message[:300]}
                    for item in batch
                ])
            continue

        for method, message in getattr(retriever, "last_method_errors", {}).items():
            if method in summaries and not any(
                error["message"] == message for error in summaries[method]["errors"]
            ):
                summaries[method]["errors"].append({
                    "type": "RetrievalError",
                    "message": str(message)[:300],
                })

        for method in selected:
            method_results = retrieved_by_method.get(method) or [[] for _ in batch]
            for item, retrieved in zip(batch, method_results):
                scores = score_retrieval(item, retrieved, retriever=retriever, k=selected_k)
                summaries[method]["queries"].append({
                    "input": str(item.get("input", ""))[:200],
                    "retrieved": len(retrieved or []),
                    **scores,
                })
                if scores["label_available"]:
                    summaries[method]["label_coverage"] += 1

    for method, summary in summaries.items():
        for metric in METRICS:
            values = [
                row[metric]
                for row in summary["queries"]
                if row.get(metric) is not None and not row.get("error")
            ]
            summary["metrics"][metric] = sum(values) / len(values) if values else None
        summary["evaluated_questions"] = sum(
            1 for row in summary["queries"] if row.get("label_available")
        )
        summary["total_questions"] = len(golden_data)

    return {
        "k": selected_k,
        "total_questions": len(golden_data),
        "methods": summaries,
    }
