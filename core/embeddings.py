"""Session-scoped embedding adapters with profile-isolated vector collections.

Provider credentials are passed directly to SDK clients and are never placed
in Chroma metadata, environment variables, logs, or on-disk cache keys.
"""

from __future__ import annotations

from functools import lru_cache
import os
import threading
import hashlib
import logging
import math
import time
from typing import Protocol

from core.rate_limit import RateLimiter, get_retry_wait, is_rate_limit_error
from core.logging_utils import log_event

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_PROVIDER = "local"
EMBEDDING_MODELS = {
    "local": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    "openai": ["text-embedding-3-small", "text-embedding-3-large"],
    "gemini": ["gemini-embedding-001"],
}
MAX_EMBEDDING_BATCH_SIZE = 64
EMBEDDING_REQUEST_TIMEOUT_SECONDS = 30
MAX_EMBEDDING_RETRIES = 3


class EmbeddingError(RuntimeError):
    """Safe, user-facing embedding error that does not include request details."""


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


@lru_cache(maxsize=4)
def _load_local_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


_LOCAL_MODEL_LOCK = threading.Lock()


def _local_model(model_name: str):
    """Load each local model once per process, even when many users ask at the same moment.

    ``lru_cache`` alone does not lock: eight simultaneous first requests would load
    eight copies of the model and could exhaust a 2.7 GB host.
    """
    with _LOCAL_MODEL_LOCK:
        return _load_local_model(model_name)


_local_model.cache_clear = _load_local_model.cache_clear


def _safe_provider_error(provider: str, error: BaseException) -> EmbeddingError:
    message = str(error).casefold()
    if any(word in message for word in ("401", "403", "unauthorized", "authentication", "api key")):
        detail = "authentication failed; check the embedding API key"
    elif is_rate_limit_error(error):
        detail = "rate limit reached; wait and retry with a smaller batch"
    elif "timeout" in message or "timed out" in message:
        detail = "request timed out; retry or check network stability"
    elif any(word in message for word in ("connection", "connecterror", "network is unreachable")):
        detail = "provider could not be reached; check network access"
    else:
        detail = "request failed; verify the model ID and provider account"
    return EmbeddingError(f"{provider.title()} embeddings {detail}.")


def _with_provider_retries(provider: str, call):
    for attempt in range(MAX_EMBEDDING_RETRIES):
        try:
            with RateLimiter(provider=provider):
                return call()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt + 1 >= MAX_EMBEDDING_RETRIES:
                raise _safe_provider_error(provider, exc) from None
            delay = get_retry_wait(exc, base_wait_seconds=4, attempt=attempt)
            log_event(
                logger,
                logging.WARNING,
                "embedding_rate_limited",
                provider=provider,
                attempt=attempt + 1,
                retry_after_s=delay,
            )
            time.sleep(delay)


class LocalSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output = []
        model = _local_model(self.model_name)
        for start in range(0, len(texts), MAX_EMBEDDING_BATCH_SIZE):
            vectors = model.encode(
                texts[start : start + MAX_EMBEDDING_BATCH_SIZE],
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            output.extend(vector.tolist() for vector in vectors)
        return output

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)


class OpenAIEmbeddingAdapter:
    def __init__(self, model_name: str, api_key: str):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            timeout=EMBEDDING_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), MAX_EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + MAX_EMBEDDING_BATCH_SIZE]
            response = _with_provider_retries(
                "openai",
                lambda: self.client.embeddings.create(model=self.model_name, input=batch),
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)


class GeminiEmbeddingAdapter:
    def __init__(self, model_name: str, api_key: str):
        from google import genai
        from google.genai import types

        self.model_name = model_name
        self._types = types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=EMBEDDING_REQUEST_TIMEOUT_SECONDS * 1000),
        )

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), MAX_EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + MAX_EMBEDDING_BATCH_SIZE]
            response = _with_provider_retries(
                "gemini",
                lambda: self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=self._types.EmbedContentConfig(task_type=task_type),
                ),
            )
            vectors.extend(list(item.values) for item in (response.embeddings or []))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_QUERY")


_ADAPTERS = {
    "local": lambda model, _key: LocalSentenceTransformer(model),
    "openai": lambda model, key: OpenAIEmbeddingAdapter(model, key),
    "gemini": lambda model, key: GeminiEmbeddingAdapter(model, key),
}


def _local_model_allowed(model: str) -> bool:
    """Local models are downloaded and executed on the server, so users may only pick vetted ones.

    A free-text name could be a multi-gigabyte repository (disk/RAM exhaustion) or a
    filesystem path. Operators can add models with LOCAL_EMBEDDING_ALLOWLIST=a,b or
    open the field fully with ALLOW_CUSTOM_LOCAL_MODELS=1 on a private machine.
    """
    if os.getenv("ALLOW_CUSTOM_LOCAL_MODELS", "").strip().lower() in {"1", "true", "yes"}:
        return True
    extra = {item.strip() for item in os.getenv("LOCAL_EMBEDDING_ALLOWLIST", "").split(",") if item.strip()}
    return model in set(EMBEDDING_MODELS["local"]) | extra


def normalize_embedding_settings(
    provider: str,
    model: str,
    api_key: str = "",
    *,
    require_key: bool = True,
) -> dict:
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if provider not in _ADAPTERS:
        raise ValueError("Choose Local, OpenAI, or Gemini as the embedding provider.")
    if not model or len(model) > 200:
        raise ValueError("Enter a valid embedding model ID (1–200 characters).")
    if provider == "local" and not _local_model_allowed(model):
        raise ValueError(
            "Choose one of the built-in local embedding models: "
            + ", ".join(EMBEDDING_MODELS["local"])
            + ". Arbitrary model names or paths are disabled on shared deployments."
        )
    api_key = (api_key or "").strip()
    if require_key and provider != "local" and not api_key:
        raise ValueError(f"Enter an API key for the {provider.title()} embedding provider.")
    return {
        "provider": provider,
        "model": model,
        "api_key": api_key if provider != "local" else "",
    }


def build_embedding_backend(provider: str, model: str, api_key: str = "") -> EmbeddingBackend:
    settings = normalize_embedding_settings(provider, model, api_key)
    try:
        return _ADAPTERS[settings["provider"]](settings["model"], settings["api_key"])
    except EmbeddingError:
        raise
    except Exception as exc:
        raise _safe_provider_error(settings["provider"], exc) from None


def close_embedding_backend(backend) -> None:
    client = getattr(backend, "client", None)
    if client is not None and hasattr(client, "close"):
        try:
            client.close()
        except Exception:
            logger.debug("Embedding client cleanup failed (%s).", type(client).__name__)


def validate_embeddings(vectors: list[list[float]], expected_count: int) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise EmbeddingError("The embedding provider returned an incomplete batch. Check the model and retry.")
    checked = []
    dimension = None
    for vector in vectors:
        values = [float(value) for value in vector]
        if not values or any(not math.isfinite(value) for value in values):
            raise EmbeddingError("The embedding provider returned invalid vectors. Check the model and retry.")
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            raise EmbeddingError("The embedding provider returned inconsistent vector dimensions.")
        checked.append(values)
    return checked


def collection_name_for_embedding(provider: str, model: str, base_name: str) -> str:
    """Use independent vector collections for different embedding spaces."""
    provider = (provider or DEFAULT_EMBEDDING_PROVIDER).strip().lower()
    model = (model or DEFAULT_LOCAL_EMBEDDING_MODEL).strip()
    if provider == "local" and model == DEFAULT_LOCAL_EMBEDDING_MODEL:
        return base_name
    digest = hashlib.sha256(f"{provider}:{model}".encode("utf-8")).hexdigest()[:12]
    return f"{base_name}_{digest}"


def lexical_index_name(
    provider: str,
    model: str,
    base_name: str = "hybrid_search",
    collection_base_name: str = "research_papers",
) -> str:
    collection_name = collection_name_for_embedding(provider, model, collection_base_name)
    if collection_name == collection_base_name:
        return f"{base_name}.sqlite3"
    digest = collection_name.rsplit("_", 1)[-1]
    return f"{base_name}_{digest}.sqlite3"

