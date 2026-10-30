import os
from urllib.parse import urlparse


# NOTE: these are the base URLs expected by each provider's *SDK client*.
# The Groq SDK (used by langchain-groq) appends "/openai/v1/..." itself, so its
# base URL must be the bare host. Passing "https://api.groq.com/openai/v1" makes
# it request ".../openai/v1/openai/v1/chat/completions", which Groq answers with
# 404 Not Found. The OpenAI-style providers below take the "/v1" form.
DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434",
}

_GROQ_PATH_SUFFIXES = ("/openai/v1", "/openai", "/v1")
_ASYNC_HTTP_TIMEOUT_SECONDS = 90.0


def _normalize_groq_base_url(value: str) -> str:
    """Accept both ``https://api.groq.com`` and the OpenAI-compatible form."""
    cleaned = value.rstrip("/")
    changed = True
    while changed:
        changed = False
        for suffix in _GROQ_PATH_SUFFIXES:
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[: -len(suffix)].rstrip("/")
                changed = True
    return cleaned


def provider_base_url(provider: str, *, explicit: str | None = None) -> str | None:
    """Resolve and validate a provider endpoint without exposing credentials."""
    normalized = (provider or "").strip().lower()
    if normalized in {"google", "gemini"}:
        return None
    value = (
        explicit
        or os.getenv(f"{normalized.upper()}_BASE_URL")
        or DEFAULT_BASE_URLS.get(normalized)
    )
    if not value:
        return None
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Invalid {normalized.title()} API base URL. "
            "Use a complete URL such as https://api.example.com/v1."
        )
    if normalized == "groq":
        value = _normalize_groq_base_url(value)
    return value


def get_model_provider() -> str:
    provider = (os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "groq").strip().lower()
    aliases = {
        "grok": "groq",
        "xai": "groq",
        "groq": "groq",
        "openai": "openai",
        "gemini": "gemini",
        "google": "gemini",
        "anthropic": "anthropic",
        "openrouter": "openrouter",
        "ollama": "ollama",
    }
    return aliases.get(provider, provider)


def groq_reasoning_options(model_name: str | None) -> dict:
    """Reasoning parameters that Groq documents for the selected model family.

    * ``openai/gpt-oss-*``: ``reasoning_format`` is NOT supported; effort is
      ``low|medium|high`` and ``include_reasoning`` controls the response field.
    * Qwen / DeepSeek-R1 style models: ``reasoning_format=hidden`` keeps the
      reasoning out of ``content`` so JSON and answers parse cleanly.
    * Everything else (for example Llama): no reasoning parameters at all.
    """
    name = (model_name or "").strip().lower()
    if "gpt-oss" in name:
        return {"reasoning_effort": "low", "model_kwargs": {"include_reasoning": False}}
    if "qwen" in name or "deepseek-r1" in name:
        return {"reasoning_format": "hidden"}
    return {}


def _async_http_client(timeout=None):
    """Async HTTP client that never keeps idle connections.

    Evaluation (DeepEval) calls the async API. Pooled keep-alive sockets
    would otherwise stay registered on an event loop that may already be
    closed or idle; the provider later resets them and asyncio raises
    ``ConnectionResetError`` (WinError 10054 on Windows). With no keep-alive
    pool, every connection is opened and closed inside the request that used
    it. Live chat uses the synchronous client and keeps its pooling.
    """
    import httpx

    seconds = float(timeout) if isinstance(timeout, (int, float)) and timeout > 0 else _ASYNC_HTTP_TIMEOUT_SECONDS
    return httpx.AsyncClient(
        timeout=httpx.Timeout(seconds),
        limits=httpx.Limits(max_keepalive_connections=0),
    )


def build_chat_model(
    model_name: str | None = None,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    **kwargs,
):
    """Build the appropriate chat model based on the selected provider.

    Supported providers:
      - groq (default)
      - openai
      - gemini
      - anthropic
      - openrouter
      - ollama

    Provider credentials must be passed explicitly by the active session. This
    avoids silently sharing a server-level key between Streamlit sessions.
    """
    selected_provider = (provider or get_model_provider()).strip().lower()
    selected_provider = {
        "google": "gemini",
        "grok": "groq",
        "xai": "groq",
    }.get(selected_provider, selected_provider)
    model_name = model_name or os.getenv("GENERATOR_MODEL") or os.getenv("JUDGE_MODEL") or "openai/gpt-oss-20b"

    if selected_provider != "ollama" and not (api_key or "").strip():
        raise ValueError(
            f"An API key must be entered for the {selected_provider.title()} provider in this session."
        )

    if selected_provider == "groq":
        from langchain_groq import ChatGroq

        groq_kwargs = dict(kwargs)
        groq_kwargs.setdefault("model", model_name)
        groq_kwargs.setdefault("groq_api_key", api_key)
        groq_kwargs.setdefault("base_url", provider_base_url("groq"))
        groq_kwargs.setdefault("max_retries", 0)
        groq_kwargs.setdefault("temperature", 0)
        groq_kwargs.setdefault("http_async_client", _async_http_client(groq_kwargs.get("timeout")))
        for name, value in groq_reasoning_options(groq_kwargs["model"]).items():
            if name == "model_kwargs":
                groq_kwargs["model_kwargs"] = {**value, **(groq_kwargs.get("model_kwargs") or {})}
            else:
                groq_kwargs.setdefault(name, value)
        return ChatGroq(**groq_kwargs)

    if selected_provider == "openai":
        from langchain_openai import ChatOpenAI

        openai_kwargs = dict(kwargs)
        openai_kwargs.setdefault("model", model_name)
        openai_kwargs.setdefault("api_key", api_key)
        openai_kwargs.setdefault("base_url", provider_base_url("openai"))
        openai_kwargs.setdefault("max_retries", 0)
        openai_kwargs.setdefault("http_async_client", _async_http_client(openai_kwargs.get("timeout")))
        return ChatOpenAI(**openai_kwargs)

    if selected_provider in {"gemini", "google"}:
        from core.google_chat import SessionGoogleChatModel

        gemini_kwargs = dict(kwargs)
        return SessionGoogleChatModel(
            model_name=model_name,
            api_key=api_key,
            temperature=gemini_kwargs.pop("temperature", 0.0),
            max_output_tokens=gemini_kwargs.pop("max_tokens", None),
            timeout=gemini_kwargs.pop("timeout", 90.0),
            streaming=gemini_kwargs.pop("streaming", True),
        )

    if selected_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        anthropic_kwargs = dict(kwargs)
        anthropic_kwargs.setdefault("model", model_name)
        anthropic_kwargs.setdefault("anthropic_api_key", api_key)
        anthropic_kwargs.setdefault("base_url", provider_base_url("anthropic"))
        anthropic_kwargs.setdefault("max_retries", 0)
        return ChatAnthropic(**anthropic_kwargs)

    if selected_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        openrouter_kwargs = dict(kwargs)
        openrouter_kwargs.setdefault("model", model_name)
        openrouter_kwargs.setdefault("api_key", api_key)
        openrouter_kwargs.setdefault("base_url", provider_base_url("openrouter"))
        openrouter_kwargs.setdefault("max_retries", 0)
        openrouter_kwargs.setdefault(
            "http_async_client", _async_http_client(openrouter_kwargs.get("timeout"))
        )
        return ChatOpenAI(**openrouter_kwargs)

    if selected_provider == "ollama":
        from langchain_ollama import ChatOllama

        ollama_kwargs = dict(kwargs)
        ollama_kwargs.setdefault("model", model_name)
        ollama_kwargs.setdefault("base_url", provider_base_url("ollama"))
        ollama_kwargs.setdefault("num_predict", kwargs.get("max_tokens"))
        return ChatOllama(**ollama_kwargs)

    raise ValueError(
        f"Unsupported LLM provider: {selected_provider}. "
        "Use one of: groq, openai, gemini, anthropic, openrouter, ollama."
    )


def close_chat_model(model) -> None:
    """Close provider HTTP clients when a request or evaluation finishes."""
    if model is None:
        return
    close_method = getattr(model, "close", None)
    if callable(close_method):
        try:
            close_method()
            return
        except Exception:
            return
    client = getattr(model, "client", None)
    close_method = getattr(client, "close", None)
    if callable(close_method):
        try:
            close_method()
        except Exception:
            pass


async def aclose_chat_model(model) -> None:
    """Close the async HTTP client of a model on the loop that used it."""
    if model is None:
        return
    async_client = getattr(model, "http_async_client", None)
    close_method = getattr(async_client, "aclose", None)
    if callable(close_method):
        try:
            await close_method()
        except Exception:
            pass
