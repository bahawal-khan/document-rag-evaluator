"""
Simple exact-match cache: if the same (normalized) question is asked again,
the cached answer is returned instantly — no new LLM call, ~0 latency, ~0 cost.

A production system might use semantic caching (embedding-similarity based),
but exact-match is simple, predictable, and reliable enough at this scale.
"""

import json
import hashlib
import os
import threading
from pathlib import Path
from core.config import CACHE_FILE

_CACHE_LOCK = threading.RLock()


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _key(question: str, context: str = "") -> str:
    return request_key(
        provider="legacy",
        model="legacy",
        prompt=_normalize(question),
        parameters={},
        context=context,
    )


def request_key(
    *,
    provider: str,
    model: str,
    prompt: str,
    parameters: dict | None = None,
    context=None,
) -> str:
    """Build a stable, credential-free key for one complete model request."""
    payload = {
        "provider": str(provider or "").strip().lower(),
        "model": str(model or "").strip(),
        "prompt": prompt if isinstance(prompt, str) else str(prompt),
        "parameters": parameters or {},
        "context": context or "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_cache(cache_file: str | Path | None = None) -> dict:
    path = Path(cache_file) if cache_file is not None else CACHE_FILE
    with _CACHE_LOCK:
        if path.exists():
            try:
                cache = json.loads(path.read_text(encoding="utf-8"))
                return cache if isinstance(cache, dict) else {}
            except (OSError, ValueError, json.JSONDecodeError):
                return {}
        return {}


def is_usable_cache_entry(entry) -> bool:
    """Reject malformed entries and partial answers left by older versions."""
    if not isinstance(entry, dict):
        return False
    answer = entry.get("answer")
    return (
        isinstance(answer, str)
        and bool(answer.strip())
        and "response cut off due to the time limit" not in answer.casefold()
    )


def get_cached(
    question: str,
    cache_file: str | Path | None = None,
    *,
    cache_context: str = "",
):
    cache = load_cache(cache_file)
    return cache.get(_key(question, cache_context))


def set_cached(
    question: str,
    entry: dict,
    cache_file: str | Path | None = None,
    *,
    cache_context: str = "",
):
    path = Path(cache_file) if cache_file is not None else CACHE_FILE
    if not is_usable_cache_entry(entry):
        return
    with _CACHE_LOCK:
        cache = load_cache(path)
        cache[_key(question, cache_context)] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def get_cached_request(
    provider: str,
    model: str,
    prompt: str,
    *,
    parameters: dict | None = None,
    context=None,
    cache_file: str | Path | None = None,
):
    cache = load_cache(cache_file)
    return cache.get(request_key(
        provider=provider,
        model=model,
        prompt=prompt,
        parameters=parameters,
        context=context,
    ))


def set_cached_request(
    provider: str,
    model: str,
    prompt: str,
    entry: dict,
    *,
    parameters: dict | None = None,
    context=None,
    cache_file: str | Path | None = None,
):
    if not is_usable_cache_entry(entry):
        return
    path = Path(cache_file) if cache_file is not None else CACHE_FILE
    with _CACHE_LOCK:
        cache = load_cache(path)
        cache[request_key(
            provider=provider,
            model=model,
            prompt=prompt,
            parameters=parameters,
            context=context,
        )] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def cache_stats(cache_file: str | Path | None = None):
    cache = load_cache(cache_file)
    return {"cached_questions": len(cache)}


def clear_cache(cache_file: str | Path | None = None):
    path = Path(cache_file) if cache_file is not None else CACHE_FILE
    if path.exists():
        path.unlink()
