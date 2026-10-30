"""Credential-free cache for completed evaluation results."""

import json
import threading
from pathlib import Path

from core.cache import request_key
from core.config import METRIC_THRESHOLD

_CACHE_LOCK = threading.RLock()
_CACHE_VERSION = 1


def evaluation_key(
    golden_data: list[dict],
    options: dict,
    answer_settings: dict,
    judge_settings: dict,
    retrieval_settings: dict,
) -> str:
    """Build a stable key for one complete evaluation configuration."""
    payload = {
        "version": _CACHE_VERSION,
        "golden_data": golden_data,
        "options": options,
        "metric_threshold": METRIC_THRESHOLD,
        "answer": {
            "provider": answer_settings.get("provider", ""),
            "model": answer_settings.get("model", ""),
        },
        "judge": {
            "provider": judge_settings.get("provider", ""),
            "model": judge_settings.get("model", ""),
        },
        "retrieval": {
            key: retrieval_settings.get(key)
            for key in (
                "method",
                "embedding_provider",
                "embedding_model",
                "use_reranker",
                "reranker_model",
                "top_k",
                "candidate_k",
            )
        },
    }
    return request_key(
        provider="evaluation",
        model="evaluation-result",
        prompt=json.dumps(payload, sort_keys=True, ensure_ascii=False),
        parameters={"version": _CACHE_VERSION},
    )


def get_cached_evaluation(cache_file: str | Path, key: str) -> dict | None:
    path = Path(cache_file)
    with _CACHE_LOCK:
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
    result = cache.get(key) if isinstance(cache, dict) else None
    return result if isinstance(result, dict) else None


def set_cached_evaluation(cache_file: str | Path, key: str, result: dict) -> None:
    if not isinstance(result, dict):
        return
    path = Path(cache_file)
    with _CACHE_LOCK:
        try:
            cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError, json.JSONDecodeError):
            cache = {}
        if not isinstance(cache, dict):
            cache = {}
        cache[key] = result
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(cache, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)


def clear_evaluation_cache(cache_file: str | Path) -> None:
    """Invalidate completed evaluations after the indexed corpus changes."""
    try:
        Path(cache_file).unlink(missing_ok=True)
    except OSError:
        pass
