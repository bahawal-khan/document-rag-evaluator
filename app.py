"""
Main chat page — Research Paper RAG Assistant.

Live-path guardrails (all kept FAST, within the 2-2.5s latency budget):
  - Exact-match cache        -> repeated questions answered instantly
  - Scope check (retrieval distance) -> off-topic questions never reach the LLM
  - Toxic input pre-filter   -> regex-based, before any LLM call
  - PII output post-check    -> regex-based, scans the response after generation
  - Token-budget trimming    -> estimated input <=1500 tokens, output <=650 tokens
  - Streaming with a hard deadline -> token-by-token, true preemptive timeout
"""

import streamlit as st
from core.rate_limit import bind_rate_limit_scope
from core.safe_markdown import neutralize_markdown, safe_stream
from core.abuse import chat_allowed
from pathlib import Path
import time
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

from core.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
    MAX_OUTPUT_TOKENS,
    RERANKER_SCOPE_THRESHOLD,
    SCOPE_DISTANCE_THRESHOLD,
)
from core.ingest import (
    MAX_DOCUMENT_SIZE_BYTES,
    MAX_TOTAL_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_FILE_COUNT,
    build_vectorstore,
    get_existing_collection,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
)
from core.generator import (
    build_generator,
    build_prompt_within_budget,
    format_llm_error,
    is_out_of_scope,
    stream_answer,
    OUT_OF_SCOPE_MESSAGE,
    PROMPT_FINGERPRINT,
)
from core.cache import (
    clear_cache,
    get_cached_request,
    is_usable_cache_entry,
    set_cached_request,
)
from core.evaluation_cache import clear_evaluation_cache
from core.cost import estimate_cost
from core.safety_fast import check_pii, check_toxicity
from core.tracer import (
    log_trace,
    compute_operational_metrics,
    langsmith_enabled,
    test_langsmith_connection,
    clear_traces,
)
from core.model_settings import (
    get_model_settings,
    get_retrieval_settings,
    get_session_retriever,
    render_model_settings,
    require_api_key,
    require_embedding_settings,
)
from core.workspace import WorkspaceLimitError, clear_workspace, get_workspace_paths
from core.evaluation_jobs import load_evaluation_job, delete_workspace_evaluation_jobs

st.set_page_config(page_title="Document RAG Assistant", page_icon="📄", layout="wide")
try:
    workspace = get_workspace_paths()
except WorkspaceLimitError as limit_error:
    st.error(str(limit_error))
    st.stop()
bind_rate_limit_scope(workspace["root"].name)
active_eval_job_id = st.session_state.get("active_eval_job_id") or st.query_params.get("evaluation_job")
active_eval_job = load_evaluation_job(
    active_eval_job_id,
    workspace_id=workspace["root"].name,
)
evaluation_in_progress = bool(
    active_eval_job and active_eval_job.get("status") in {"queued", "running"}
)
if st.session_state.pop("clear_api_keys_pending", False):
    for role in ("answer", "golden", "judge"):
        st.session_state.pop(f"{role}_api_key", None)
    st.session_state.pop("embedding_api_key", None)
pending_cleanup_id = st.session_state.get("pending_workspace_cleanup")
if pending_cleanup_id:
    try:
        old_retriever = st.session_state.pop("retriever", None)
        if old_retriever is not None and hasattr(old_retriever, "close"):
            old_retriever.close()
        clear_workspace(
            {"root": DATA_DIR / "workspaces" / pending_cleanup_id}
        )
        delete_workspace_evaluation_jobs(pending_cleanup_id)
        st.session_state.pop("pending_workspace_cleanup", None)
        st.session_state.pop("workspace_id", None)
    except OSError as exc:
        st.error(f"Could not clear the previous workspace: {exc}")

from core.ui import apply_responsive_styles, render_hero

apply_responsive_styles()

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
retriever = get_session_retriever(workspace["chroma"], workspace["root"].name)

# ---------------- Sidebar: paper ingestion ----------------
with st.sidebar:
    st.header("Workspace")
    st.caption("Your documents, dataset, cache, and traces are isolated to this browser session.")
    render_model_settings(st.sidebar)

    st.divider()
    st.header("📚 Documents")
    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"documents_{workspace['root'].name}",
        help=(
            f"PDF, DOCX, TXT, and Markdown. Up to {MAX_UPLOAD_FILE_COUNT} files, "
            f"{MAX_DOCUMENT_SIZE_BYTES // (1024 * 1024)}MB each and "
            f"{MAX_TOTAL_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB total."
        ),
    )

    with st.expander("Chunking (how documents are split)"):
        chunk_size = st.number_input(
            "Chunk size (characters)",
            min_value=MIN_CHUNK_SIZE,
            max_value=MAX_CHUNK_SIZE,
            value=int(st.session_state.get("indexed_chunking", (CHUNK_SIZE, CHUNK_OVERLAP))[0]),
            step=50,
            key="chunk_size_input",
            help="Smaller chunks give precise, focused passages. Larger chunks keep more surrounding context.",
        )
        chunk_overlap = st.number_input(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=MAX_CHUNK_SIZE // 2,
            value=int(st.session_state.get("indexed_chunking", (CHUNK_SIZE, CHUNK_OVERLAP))[1]),
            step=10,
            key="chunk_overlap_input",
            help="Characters repeated between neighbouring chunks so a sentence is not cut off between them.",
        )
        st.caption(
            "Applies to the next ingest. Ingesting a file again with new settings replaces its old chunks. "
            "After changing it, generate the golden dataset again, because chunk references change."
        )
    chunking_invalid = chunk_overlap * 2 > chunk_size
    if chunking_invalid:
        st.warning("Chunk overlap must be at most half of the chunk size.")

    upload_count_exceeded = bool(uploaded and len(uploaded) > MAX_UPLOAD_FILE_COUNT)
    upload_size_exceeded = bool(
        uploaded and sum(uploaded_file.size for uploaded_file in uploaded) > MAX_TOTAL_UPLOAD_SIZE_BYTES
    )
    if upload_count_exceeded:
        st.warning(f"Select at most {MAX_UPLOAD_FILE_COUNT} files per upload batch.")
    if upload_size_exceeded:
        st.warning(f"The total upload size must be {MAX_TOTAL_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB or less.")

    if st.button(
        "📥 Ingest Papers",
        width="stretch",
        disabled=not uploaded or upload_count_exceeded or upload_size_exceeded or chunking_invalid,
    ):
        upload_dir = workspace["uploads"]
        upload_dir.mkdir(parents=True, exist_ok=True)
        files = []
        try:
            embedding_settings = require_embedding_settings(for_ingest=True)
            if any(uploaded_file.size > MAX_DOCUMENT_SIZE_BYTES for uploaded_file in uploaded):
                raise ValueError(
                    f"Each document must be {MAX_DOCUMENT_SIZE_BYTES // (1024 * 1024)}MB or smaller."
                )
            for uploaded_file in uploaded:
                suffix = Path(uploaded_file.name).suffix.lower()
                stored_path = upload_dir / f"{uuid4().hex}{suffix}"
                files.append((uploaded_file.name, stored_path))
                uploaded_file.seek(0)
                with stored_path.open("wb") as destination:
                    while True:
                        block = uploaded_file.read(1024 * 1024)
                        if not block:
                            break
                        destination.write(block)

            indexing_status = st.empty()

            def _show_indexing_progress(file_name, chunk_count):
                indexing_status.info(
                    f"Indexed at least {chunk_count:,} chunks from {file_name}..."
                )

            with st.spinner("Extracting and indexing bounded document chunks..."):
                clear_cache(workspace["cache"])
                clear_evaluation_cache(workspace["root"] / "evaluation_results_cache.json")
                collection, n_chunks = build_vectorstore(
                    files,
                    chroma_dir=workspace["chroma"],
                    progress_callback=_show_indexing_progress,
                    embedding_provider=embedding_settings["embedding_provider"],
                    embedding_model=embedding_settings["embedding_model"],
                    embedding_api_key=embedding_settings["embedding_api_key"],
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )
                st.session_state["indexed_chunking"] = (int(chunk_size), int(chunk_overlap))
                retriever = get_session_retriever(
                    workspace["chroma"], workspace["root"].name, refresh=True
                )
                if retriever.use_reranker and not retriever.warm_up_reranker(retriever.reranker_model):
                    st.sidebar.info(
                        "The cross-encoder could not load. Search results will use the selected retriever ranking."
                    )
                indexing_status.empty()
                st.success(f"✅ {len(files)} document(s) → {n_chunks} chunks indexed")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
        finally:
            for _, stored_path in files:
                try:
                    stored_path.unlink(missing_ok=True)
                except OSError:
                    pass

    retrieval_settings = get_retrieval_settings()
    existing = get_existing_collection(
        chroma_dir=workspace["chroma"],
        embedding_provider=retrieval_settings["embedding_provider"],
        embedding_model=retrieval_settings["embedding_model"],
    )
    if existing:
        st.caption(f"📦 Ready — {existing.count():,} chunks indexed")
    else:
        st.warning("No papers ingested yet")
    del existing

    _size, _overlap = st.session_state.get("indexed_chunking", (CHUNK_SIZE, CHUNK_OVERLAP))
    st.caption(f"Chunking: {_size} characters with {_overlap} overlap.")
    confirm_clear = st.checkbox(
        "I want to permanently clear this session's files and results",
        key=f"confirm_clear_{workspace['root'].name}",
    )
    if evaluation_in_progress:
        st.info("Wait for this workspace's evaluation to finish before clearing it.")
    if st.button(
        "Clear this workspace",
        disabled=not confirm_clear or evaluation_in_progress,
        key=f"clear_workspace_{workspace['root'].name}",
    ):
        st.session_state["pending_workspace_cleanup"] = workspace["root"].name
        st.session_state["clear_api_keys_pending"] = True
        for key in (
            "workspace_id",
            "retriever",
            "retriever_workspace_id",
            "messages",
            "retriever_summary",
            "generator_summary",
            "pipeline_summary",
            "quality_summary",
            "safety_content_summary",
            "scope_summary",
            "active_eval_job_id",
            "eval_loaded_job_id",
            "eval_running",
        ):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()

    # ---------------- LangSmith status ----------------
    st.header("🔭 Tracing")
    if langsmith_enabled():
        st.success("LangSmith: env vars set")
        st.warning("Tracing is ON: visitors' questions and document excerpts are sent to the server owner's LangSmith project.")
        if st.button("🔌 Test LangSmith Connection"):
            with st.spinner("Testing connection..."):
                ok, msg = test_langsmith_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    else:
        st.caption("LangSmith: OFF — configure optional tracing values in the server environment or Streamlit secrets")

    st.divider()

    # ---------------- Provider connectivity checks ----------------
    st.header("🧪 API Checks")
    if st.button("Test embedding provider", width="stretch"):
        backend = None
        try:
            embedding_settings = require_embedding_settings(for_ingest=True)
            from core.embeddings import (
                build_embedding_backend,
                close_embedding_backend,
                validate_embeddings,
            )

            backend = build_embedding_backend(
                embedding_settings["embedding_provider"],
                embedding_settings["embedding_model"],
                embedding_settings["embedding_api_key"],
            )
            validate_embeddings(backend.embed_queries(["connection check"]), 1)
            st.success(f"{embedding_settings['embedding_provider_label']} embedding connection successful.")
        except Exception as exc:
            st.error(str(exc))
        finally:
            from core.embeddings import close_embedding_backend

            close_embedding_backend(backend)

    if st.button("Test answer model", width="stretch"):
        settings = get_model_settings("answer")
        test_model = None
        try:
            settings = require_api_key("answer")
            test_model = build_generator(
                streaming=False,
                timeout=20,
                model_name=settings["model"],
                provider=settings["provider"],
                api_key=settings["api_key"],
            )
            from core.rate_limit import RateLimiter

            with RateLimiter(provider=settings["provider"]):
                test_model.invoke("Reply with the word OK.")
            st.success(f"{settings['provider_label']} connection successful.")
        except Exception as e:
            provider = settings.get("provider", "groq")
            key = settings.get("api_key")
            st.error(format_llm_error(e, provider, key))
        finally:
            from core.provider import close_chat_model

            close_chat_model(test_model)

    st.divider()

    # ---------------- Live operational metrics ----------------
    st.header("⚡ Live Operational Metrics")
    om = compute_operational_metrics(workspace["traces"])
    if om:
        c1, c2 = st.columns(2)
        c1.metric("P50 Latency", f"{om['p50_latency_s']:.2f}s" if om['p50_latency_s'] is not None else "—")
        c2.metric("P99 Latency", f"{om['p99_latency_s']:.2f}s" if om['p99_latency_s'] is not None else "—")
        c3, c4 = st.columns(2)
        c3.metric("Cache Hit Rate", f"{om['cache_hit_rate']*100:.0f}%")
        c4.metric(
            "Estimated Total Cost",
            f"${om['total_cost_usd']:.4f}" if om["total_cost_usd"] is not None else "—",
        )
        cache_latency = om["avg_cache_latency_s"]
        cache_latency_text = f"{cache_latency:.3f}s" if cache_latency is not None else "—"
        st.caption(
            f"Reliability: {om['reliability']*100:.1f}% | Model API calls: {om['total_requests']} "
            f"(cache hits: {om['cache_hits']}) "
            f"({om['generation_requests']} generated, avg cache response {cache_latency_text}); "
            f"cost estimates: {om['cost_estimates_count']}, unpriced: {om['cost_unknown_count']}"
        )
        if st.button("🗑️ Clear trace history", help="Deletes old traces — use after a config/code change, so new numbers aren't skewed by stale data"):
            clear_traces(workspace["traces"])
            st.rerun()
    else:
        st.caption("No queries yet")

    st.divider()
    st.page_link("pages/1_Evaluation_Suite.py", label="📊 Full Evaluation Suite", icon="📊")

# ---------------- Main chat ----------------
render_hero(
    "📄 Document RAG Assistant",
    "Ask questions about your uploaded documents — answers are grounded only in the supplied material.",
    ("Hybrid search", "Cited answers", "Session-private keys"),
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(neutralize_markdown(msg["content"]))
        if msg.get("meta"):
            st.caption(neutralize_markdown(msg["meta"]))

question = st.chat_input("Ask something about the papers...")

if question:
    _allowed, _wait = chat_allowed(workspace["root"].name)
    if not _allowed:
        st.warning(f"You are sending questions too quickly. Please wait about {_wait} seconds.")
        st.stop()
    request_started = time.perf_counter()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retriever = get_session_retriever(workspace["chroma"], workspace["root"].name)
    trace_entry = {"retrieval_method": retriever.method}

    with st.chat_message("assistant"):
        try:
            # Validate the question before cache lookup or retrieval so an
            # oversized input cannot bypass the budget guard on a cache hit.
            build_prompt_within_budget(question, [])
        except ValueError as exc:
            st.warning(str(exc))
            st.session_state.messages.append({"role": "assistant", "content": str(exc)})
            st.stop()

        # Toxic input checks run before cache lookup so old cache entries
        # cannot bypass current guardrails.
        if check_toxicity(question):
            refusal = "Sorry, this question wasn't phrased appropriately. Please rephrase it professionally."
            st.markdown(refusal)
            st.session_state.messages.append({"role": "assistant", "content": refusal})
            trace_entry.update({
                "cache_hit": False,
                "toxicity_blocked": True,
                "guardrail_latency_s": time.perf_counter() - request_started,
                "cost_usd": 0.0,
                "error": None,
            })
            log_trace(trace_entry, workspace["traces"])
            st.stop()

        # ---- Request configuration (used by retrieval and the cache key) ----
        current_model = get_model_settings("answer")
        retrieval_settings = get_retrieval_settings()

        if not retriever.is_ready():
            st.warning("Please ingest documents from the sidebar first.")
            st.stop()

        try:
            require_embedding_settings()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        # ---- 3. Retrieve + scope check ----
        try:
            with st.spinner("Searching documents and reranking relevant passages..."):
                chunks = retriever.retrieve(question)
            degraded = retriever.last_method_errors.get(retriever.method)
            if degraded and retriever.method == "hybrid" and chunks:
                st.info(degraded)
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, (ValueError, RuntimeError))
                else "Retrieval failed. Check the index and provider configuration, then retry."
            )
            st.error(message)
            trace_entry.update({
                "cache_hit": False,
                "error": True,
                "generation_status": "retrieval_error",
                "retrieval_method": retriever.method,
                "error_type": type(exc).__name__,
                "total_latency_s": time.perf_counter() - request_started,
            })
            log_trace(trace_entry, workspace["traces"])
            st.stop()
        if is_out_of_scope(chunks, st.session_state.get("scope_threshold")):
            st.markdown(OUT_OF_SCOPE_MESSAGE)
            _top = next((c for c in chunks if isinstance(c, dict)), None)
            if _top and _top.get("rerank_score") is not None:
                st.caption(
                    f"Best passage relevance score: {float(_top['rerank_score']):.2f} "
                    "(below the scope threshold). If your question is about the document, "
                    "try rephrasing it with words from the document."
                )
            st.session_state.messages.append({"role": "assistant", "content": OUT_OF_SCOPE_MESSAGE})
            trace_entry.update({
                "cache_hit": False,
                "scope_blocked": True,
                "guardrail_latency_s": time.perf_counter() - request_started,
                "cost_usd": 0.0,
                "error": None,
            })
            log_trace(trace_entry, workspace["traces"])
            st.stop()

        # ---- 4. Build the token-budgeted prompt ----
        try:
            prompt, used_chunks, input_tokens = build_prompt_within_budget(
                question, chunks, retrieval_settings["max_input_tokens"]
            )
        except ValueError as exc:
            st.warning(str(exc))
            st.session_state.messages.append({"role": "assistant", "content": str(exc)})
            st.stop()

        # ---- 5. Cache the complete grounded request (including context) ----
        # Looking up only after retrieval prevents stale answers when the
        # document index or ranking settings change, and makes reruns cheap.
        cache_parameters = {
            "temperature": 0.1,
            "max_tokens": retrieval_settings["max_output_tokens"],
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            # Never fold the embedding API key into the cache key.
            "retrieval": {
                key: value
                for key, value in retrieval_settings.items()
                if key != "embedding_api_key"
            },
        }
        cached = get_cached_request(
            current_model["provider"],
            current_model["model"],
            prompt,
            parameters=cache_parameters,
            context=used_chunks,
            cache_file=workspace["cache"],
        )
        if is_usable_cache_entry(cached):
            st.markdown(neutralize_markdown(cached["answer"]))
            meta = f"⚡ Cached • {cached.get('sources', '')}"
            st.caption(neutralize_markdown(meta))
            st.session_state.messages.append({"role": "assistant", "content": cached["answer"], "meta": meta})
            trace_entry.update({
                "cache_hit": True,
                "cache_latency_s": time.perf_counter() - request_started,
                "provider": current_model["provider"],
                "model": current_model["model"],
                "api_calls": 0,
                "cost_usd": 0.0,
                "error": None,
            })
            log_trace(trace_entry, workspace["traces"])
            st.stop()

        # ---- 6. Stream the answer ----
        gen_metrics = {}
        try:
            model_settings = require_api_key("answer")
            llm = build_generator(
                model_name=model_settings["model"],
                provider=model_settings["provider"],
                api_key=model_settings["api_key"],
                max_tokens=retrieval_settings["max_output_tokens"],
            )
        except Exception as e:
            st.error(str(e))
            st.stop()
        placeholder_stream = stream_answer(
            llm,
            prompt,
            gen_metrics,
            provider=model_settings["provider"],
            model_name=model_settings["model"],
        )
        try:
            full_answer = st.write_stream(safe_stream(placeholder_stream))
        finally:
            from core.provider import close_chat_model

            close_chat_model(llm)

        if gen_metrics.get("error"):
            friendly_error = format_llm_error(
                Exception(gen_metrics["error"]),
                model_settings["provider"],
                model_settings["api_key"],
            )
            st.error(friendly_error)
            st.session_state.messages.append({"role": "assistant", "content": friendly_error})
            error_cost = estimate_cost(
                input_tokens,
                gen_metrics.get("output_tokens", 0),
                provider=model_settings["provider"],
                model_name=model_settings["model"],
            )
            trace_entry.update({
                "cache_hit": False,
                "provider": model_settings["provider"],
                "model": model_settings["model"],
                "api_calls": 1,
                "input_tokens": input_tokens,
                "output_tokens": gen_metrics.get("output_tokens", 0),
                "cost_usd": error_cost,
                "ttft_s": gen_metrics.get("ttft_s"),
                "total_latency_s": gen_metrics.get("total_latency_s"),
                "pii_flagged": [],
                # Keep traces useful for reliability metrics without persisting
                # raw provider exceptions, which can contain request details.
                "error": True,
                "generation_status": "error",
            })
            log_trace(trace_entry, workspace["traces"])
            st.stop()

        # ---- 6. Post-generation: PII scan on the output ----
        pii_found = check_pii(full_answer)
        cost = estimate_cost(
            input_tokens,
            gen_metrics.get("output_tokens", 0),
            provider=model_settings["provider"],
            model_name=model_settings["model"],
        )

        source_refs = []
        for chunk in used_chunks:
            location = chunk.get("location") or f"p.{chunk.get('page')}"
            source_refs.append(f"{chunk.get('source')} ({location})")
        sources = ", ".join(sorted(set(source_refs)))
        meta_parts = [
            f"⏱ {gen_metrics.get('total_latency_s', 0):.2f}s",
            f"🔤 {input_tokens}→{gen_metrics.get('output_tokens', 0)} tokens",
            f"Estimated cost ${cost:.5f}" if cost is not None else "Cost estimate unavailable for this model",
        ]
        if pii_found:
            meta_parts.append(f"⚠️ PII flagged: {', '.join(pii_found)}")
        if gen_metrics.get("timed_out"):
            meta_parts.append("⏱️ cut off at time limit")
        meta = " | ".join(meta_parts) + (f"\n📎 {sources}" if sources else "")
        st.caption(neutralize_markdown(meta))

        st.session_state.messages.append({"role": "assistant", "content": full_answer, "meta": meta})

        # ---- 7. Cache + trace ----
        if not gen_metrics.get("error") and not gen_metrics.get("timed_out"):
            set_cached_request(
                model_settings["provider"],
                model_settings["model"],
                prompt,
                {"answer": full_answer, "sources": sources},
                parameters=cache_parameters,
                context=used_chunks,
                cache_file=workspace["cache"],
            )

        trace_entry.update({
            "cache_hit": False,
            "provider": model_settings["provider"],
            "model": model_settings["model"],
            "input_tokens": input_tokens,
            "output_tokens": gen_metrics.get("output_tokens", 0),
            "cost_usd": cost,
            "ttft_s": gen_metrics.get("ttft_s"),
            "total_latency_s": gen_metrics.get("total_latency_s"),
            "pii_flagged": pii_found,
            "timed_out": bool(gen_metrics.get("timed_out")),
            "error": bool(gen_metrics.get("timed_out")),
            "generation_status": "timed_out" if gen_metrics.get("timed_out") else "success",
        })
        log_trace(trace_entry, workspace["traces"])
