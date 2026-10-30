"""Per-Streamlit-session model settings; credentials are never written to disk."""

import re

import streamlit as st

from core.embeddings import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
)
from core.config import DATA_DIR, RERANKER_MODEL
from core.cache import clear_cache

PROVIDERS = {
    "OpenAI": ("openai", "gpt-4o-mini"),
    "Gemini": ("gemini", "gemini-3.8-flash"),
    "Anthropic": ("anthropic", "claude-haiku-4-5"),
    "Groq": ("groq", "openai/gpt-oss-20b"),
    "OpenRouter": ("openrouter", "openai/gpt-4o-mini"),
    "Ollama (local)": ("ollama", "qwen2.5:7b"),
}
MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
    "gemini": ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "anthropic": ["claude-haiku-4-5", "claude-sonnet-4-6"],
    "groq": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    "openrouter": [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "anthropic/claude-3.5-sonnet",
    ],
    "ollama": ["qwen2.5:7b", "llama3.1:8b"],
}
CUSTOM_MODEL = "Enter a custom model ID"

ROLE_LABELS = {
    "answer": "Answer generation",
    "golden": "Golden dataset generation",
    "judge": "LLM judge",
}

EMBEDDING_PROVIDERS = {
    "Local Sentence Transformers": ("local", DEFAULT_LOCAL_EMBEDDING_MODEL),
    "OpenAI": ("openai", "text-embedding-3-small"),
    "Gemini": ("gemini", "gemini-embedding-001"),
}
RETRIEVAL_METHODS = {
    "Dense Vector Search": "dense",
    "BM25 Keyword Search": "bm25",
    "Hybrid Search": "hybrid",
}


def _retrieval_method_choices() -> dict[str, str]:
    from core.retriever import RETRIEVER_REGISTRY

    choices = dict(RETRIEVAL_METHODS)
    known = set(choices.values())
    for method in RETRIEVER_REGISTRY:
        if method not in known:
            choices[f"Custom: {method}"] = method
    return choices


def _role_settings(role: str) -> dict:
    provider_key = f"{role}_provider"
    model_key = f"{role}_model"
    api_key_name = f"{role}_api_key"
    default_provider, default_model = PROVIDERS["Groq"]
    provider = st.session_state.get(provider_key, default_provider)
    provider_label = next(
        (label for label, item in PROVIDERS.items() if item[0] == provider),
        "Groq",
    )
    return {
        "provider": provider,
        "model": st.session_state.get(model_key, default_model),
        "api_key": st.session_state.get(api_key_name, ""),
        "provider_label": provider_label,
    }


def get_model_settings(role: str) -> dict:
    """Return one role's current provider/model/key without persisting it."""
    if role not in ROLE_LABELS:
        raise ValueError(f"Unknown model role: {role}")
    return _role_settings(role)


def _provider_changed(role: str) -> None:
    label = st.session_state[f"{role}_provider_choice"]
    provider, model = PROVIDERS[label]
    st.session_state[f"{role}_provider"] = provider
    st.session_state[f"{role}_model"] = model
    st.session_state[f"{role}_model_choice"] = model
    st.session_state[f"{role}_custom_model"] = ""
    st.session_state[f"{role}_api_key"] = ""
    if role == "answer":
        _clear_active_answer_cache()


def _clear_active_answer_cache() -> None:
    workspace_id = st.session_state.get("workspace_id", "")
    if re.fullmatch(r"[0-9a-f]{32}", workspace_id or ""):
        clear_cache(DATA_DIR / "workspaces" / workspace_id / "answer_cache.json")
        st.session_state.pop(f"retrieval_comparison_{workspace_id}", None)


def _role_api_key_changed(role: str) -> None:
    if role == "answer":
        _clear_active_answer_cache()


def render_model_settings(container=None) -> None:
    """Render model configuration. Password fields live only in session state."""
    target = container or st
    target.subheader("Model configuration")
    target.caption("Credentials stay in this Streamlit session and are not saved to files.")
    for role, title in ROLE_LABELS.items():
        with target.expander(title, expanded=(role == "answer")):
            current = _role_settings(role)
            provider_labels = list(PROVIDERS)
            selected_index = provider_labels.index(current["provider_label"])
            selected_label = st.selectbox(
                "Provider",
                provider_labels,
                index=selected_index,
                key=f"{role}_provider_choice",
                on_change=_provider_changed,
                args=(role,),
            )
            provider, recommended_model = PROVIDERS[selected_label]
            st.session_state[f"{role}_provider"] = provider
            model_options = [*MODELS[provider], CUSTOM_MODEL]
            model_choice_key = f"{role}_model_choice"
            if model_choice_key not in st.session_state:
                current_model = st.session_state.get(f"{role}_model", recommended_model)
                st.session_state[model_choice_key] = (
                    current_model if current_model in model_options else CUSTOM_MODEL
                )
            selected_model = st.selectbox(
                "Model",
                model_options,
                key=model_choice_key,
                help="Preset options include budget-friendly choices. You can enter any model ID supported by the provider.",
            )
            if selected_model == CUSTOM_MODEL:
                custom_model = st.text_input(
                    "Custom model ID",
                    key=f"{role}_custom_model",
                    placeholder="Enter the provider model ID",
                )
                st.session_state[f"{role}_model"] = custom_model.strip()
            else:
                st.session_state[f"{role}_model"] = selected_model
            if provider == "ollama":
                st.caption("Ollama runs locally; no API key is required.")
                st.session_state[f"{role}_api_key"] = ""
            else:
                st.text_input(
                    f"{selected_label} API key",
                    type="password",
                    key=f"{role}_api_key",
                    on_change=_role_api_key_changed,
                    args=(role,),
                    help=f"Required for this role. The app does not persist it. Server environment fallback is not used.",
                    placeholder=f"Enter your {selected_label} key",
                )
    render_retrieval_settings(target)


def require_api_key(role: str) -> dict:
    settings = get_model_settings(role)
    settings["api_key"] = settings["api_key"].strip()
    if not settings["model"].strip():
        raise ValueError(f"Enter a model ID for {settings['provider_label']} in Model configuration.")
    if settings["provider"] != "ollama" and not settings["api_key"].strip():
        raise ValueError(
            f"Enter an API key for the {settings['provider_label']} provider in Model configuration."
        )
    return settings


def get_retrieval_settings() -> dict:
    provider = st.session_state.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
    if provider not in EMBEDDING_MODELS:
        provider = DEFAULT_EMBEDDING_PROVIDER
    method = st.session_state.get("retrieval_method", "hybrid")
    if method not in _retrieval_method_choices().values():
        method = "hybrid"
    provider_label = next(
        (label for label, item in EMBEDDING_PROVIDERS.items() if item[0] == provider),
        "Local Sentence Transformers",
    )
    top_k = max(1, min(100, int(st.session_state.get("retrieval_top_k", 4))))
    candidate_k = max(
        top_k,
        min(500, max(1, int(st.session_state.get("retrieval_candidate_k", 32)))),
    )
    return {
        "method": method,
        "embedding_provider": provider,
        "embedding_model": str(st.session_state.get(
            "embedding_model", EMBEDDING_PROVIDERS[provider_label][1]
        ) or "").strip()[:200],
        "embedding_api_key": st.session_state.get("embedding_api_key", ""),
        "embedding_provider_label": provider_label,
        "use_reranker": bool(st.session_state.get("retrieval_use_reranker", True)),
        "reranker_model": RERANKER_MODEL,
        "top_k": top_k,
        "candidate_k": candidate_k,
    }


def _embedding_provider_changed() -> None:
    label = st.session_state["embedding_provider_choice"]
    provider, model = EMBEDDING_PROVIDERS[label]
    st.session_state["embedding_provider"] = provider
    st.session_state["embedding_model"] = model
    st.session_state["embedding_model_choice"] = model
    st.session_state["embedding_custom_model"] = ""
    st.session_state["embedding_api_key"] = ""


def render_retrieval_settings(container=None) -> None:
    """Shared, session-only retrieval and embedding configuration."""
    target = container or st
    target.subheader("Retrieval configuration")
    with target.expander("Search and embeddings", expanded=False):
        settings = get_retrieval_settings()
        method_choices = _retrieval_method_choices()
        method_label = next(
            label for label, value in method_choices.items()
            if value == settings["method"]
        )
        selected_method = st.selectbox(
            "Retriever",
            list(method_choices),
            index=list(method_choices).index(method_label),
            key="retrieval_method_choice",
        )
        st.session_state["retrieval_method"] = method_choices[selected_method]

        provider_labels = list(EMBEDDING_PROVIDERS)
        selected_provider_label = st.selectbox(
            "Embedding provider",
            provider_labels,
            index=provider_labels.index(settings["embedding_provider_label"]),
            key="embedding_provider_choice",
            on_change=_embedding_provider_changed,
        )
        provider, recommended_model = EMBEDDING_PROVIDERS[selected_provider_label]
        model_options = [*EMBEDDING_MODELS[provider], CUSTOM_MODEL]
        model_choice_key = "embedding_model_choice"
        if model_choice_key not in st.session_state:
            current_model = st.session_state.get("embedding_model", recommended_model)
            st.session_state[model_choice_key] = (
                current_model if current_model in model_options else CUSTOM_MODEL
            )
        selected_model = st.selectbox(
            "Embedding model",
            model_options,
            key=model_choice_key,
            help="Changing the embedding model creates a separate index; re-index documents before using it.",
        )
        if selected_model == CUSTOM_MODEL:
            custom_model = st.text_input(
                "Custom embedding model ID",
                key="embedding_custom_model",
                placeholder="Enter a model ID supported by the provider",
            )
            st.session_state["embedding_model"] = custom_model.strip()
        else:
            st.session_state["embedding_model"] = selected_model

        if provider == "local":
            st.caption("Local embeddings run on this machine and need no API key.")
            st.session_state["embedding_api_key"] = ""
        else:
            st.text_input(
                f"{selected_provider_label} embedding API key",
                type="password",
                key="embedding_api_key",
                on_change=_clear_active_answer_cache,
                help="Required for indexing and dense/hybrid queries. Kept in this Streamlit session only.",
                placeholder=f"Enter your {selected_provider_label} key",
            )

        st.checkbox(
            "Rerank results with the local cross-encoder",
            value=settings["use_reranker"],
            key="retrieval_use_reranker",
            help="Adds a local scoring pass after retrieval. Its model is downloaded on first use.",
        )
        st.slider("Results per query (K)", 1, 20, settings["top_k"], key="retrieval_top_k")
        st.slider(
            "Candidate search depth",
            1,
            100,
            min(100, max(settings["top_k"], settings["candidate_k"])),
            key="retrieval_candidate_k",
            help="Search this many candidates before reranking and returning K results.",
        )


def require_embedding_settings(*, methods=None, for_ingest: bool = False) -> dict:
    """Validate the selected embedding provider only when a vector operation needs it."""
    from core.embeddings import normalize_embedding_settings

    settings = get_retrieval_settings()
    active_methods = methods or [settings["method"]]
    supported_methods = set(_retrieval_method_choices().values())
    if any(method not in supported_methods for method in active_methods):
        raise ValueError("Choose a supported retriever before continuing.")
    needs_embeddings = for_ingest or any(method != "bm25" for method in active_methods)
    normalized = normalize_embedding_settings(
        settings["embedding_provider"],
        settings["embedding_model"],
        settings["embedding_api_key"],
        require_key=needs_embeddings,
    )
    if needs_embeddings and normalized["provider"] != "local" and not normalized["api_key"]:
        raise ValueError(
            f"Enter an API key for the {settings['embedding_provider_label']} embedding provider."
        )
    settings["embedding_provider"] = normalized["provider"]
    settings["embedding_model"] = normalized["model"]
    settings["embedding_api_key"] = normalized["api_key"]
    return settings


def get_session_retriever(chroma_dir, workspace_id: str, *, refresh: bool = False):
    """Reuse a retriever only while its workspace and non-secret settings match."""
    from core.retriever import Retriever

    settings = get_retrieval_settings()
    signature = (
        workspace_id,
        settings["method"],
        settings["embedding_provider"],
        settings["embedding_model"],
        settings["top_k"],
        settings["candidate_k"],
        settings["use_reranker"],
        settings["reranker_model"],
    )
    existing = st.session_state.get("retriever")
    if (
        not refresh
        and existing is not None
        and st.session_state.get("retriever_config_signature") == signature
        and getattr(existing, "_api_key", None) == settings["embedding_api_key"]
    ):
        return existing
    if existing is not None and hasattr(existing, "close"):
        existing.close()
    st.session_state.pop(f"retrieval_comparison_{workspace_id}", None)
    retriever = Retriever(
        chroma_dir=chroma_dir,
        method=settings["method"],
        embedding_provider=settings["embedding_provider"],
        embedding_model=settings["embedding_model"],
        api_key=settings["embedding_api_key"],
        use_reranker=settings["use_reranker"],
        reranker_model=settings["reranker_model"],
        top_k=settings["top_k"],
        candidate_k=settings["candidate_k"],
    )
    st.session_state["retriever"] = retriever
    st.session_state["retriever_config_signature"] = signature
    st.session_state["retriever_workspace_id"] = workspace_id
    return retriever
