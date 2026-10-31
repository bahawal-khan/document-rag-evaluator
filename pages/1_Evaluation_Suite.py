"""Streamlit evaluation page for component, pipeline, quality, and safety metrics."""

import streamlit as st
import json
import logging
import csv
import io
import hashlib
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
MAX_GOLDEN_DATASET_UPLOAD_BYTES = 5 * 1024 * 1024

load_dotenv()

from core.ingest import get_existing_collection
from core.retriever import Retriever, RETRIEVER_REGISTRY
from core.generator import format_llm_error
from core.scope_calibration import calibrate_scope_threshold
from core.golden import (
    MAX_GOLDEN_QUESTIONS,
    generate_golden_dataset,
    load_golden_dataset,
    save_golden_dataset,
    difficulty_breakdown,
    DIFFICULTY_LEVELS,
    validate_golden_dataset,
)
from core.eval_plan import estimate_run
from core.safe_markdown import neutralize_markdown
from core.rate_limit import (
    PACING_PROFILE_LABELS,
    PACING_PROFILES,
    bind_rate_limit_scope,
    get_min_interval_seconds,
    rate_limit_scope,
)
from core.tracer import compute_operational_metrics
from core.model_settings import (
    get_model_settings,
    get_retrieval_settings,
    get_session_retriever,
    render_model_settings,
    require_api_key,
    require_embedding_settings,
)
from core.workspace import WorkspaceLimitError, get_workspace_paths
from core.evaluation_jobs import (
    load_evaluation_job,
    new_job_id,
    submit_evaluation_job,
)
from core.provider import provider_base_url

from eval.application_eval import OFF_TOPIC_PROBES
from eval.report_pdf import build_pdf_report
from eval.retrieval_eval import run_retrieval_comparison

st.set_page_config(page_title="Evaluation Suite", page_icon="📊", layout="wide")
from core.ui import apply_responsive_styles, render_hero

apply_responsive_styles()
try:
    workspace = get_workspace_paths()
except WorkspaceLimitError as limit_error:
    st.error(str(limit_error))
    st.stop()
workspace_id = workspace["root"].name
bind_rate_limit_scope(workspace_id)


def clear_saved_results():
    for key in (
        "retriever_summary",
        "generator_summary",
        "pipeline_summary",
        "quality_summary",
        "safety_content_summary",
        "scope_summary",
        "run_notes",
        "eval_result_sample_count",
        f"retrieval_comparison_{workspace_id}",
    ):
        st.session_state.pop(key, None)


requested_job_id = st.query_params.get("evaluation_job")
requested_job = load_evaluation_job(requested_job_id, workspace_id=workspace_id)
if (
    requested_job_id
    and not requested_job
    and st.session_state.get("active_eval_job_id") == requested_job_id
):
    st.session_state["eval_running"] = False
    st.session_state.pop("active_eval_job_id", None)
if requested_job:
    st.session_state["active_eval_job_id"] = requested_job_id
    st.session_state["eval_running"] = requested_job.get("status") in {"queued", "running"}
if requested_job and requested_job.get("status") == "complete":
    loaded_job_id = st.session_state.get("eval_loaded_job_id")
    if loaded_job_id != requested_job_id:
        clear_saved_results()
        st.session_state.update(requested_job.get("result") or {})
        st.session_state["eval_loaded_job_id"] = requested_job_id
    st.session_state["eval_result_sample_count"] = int(requested_job.get("sample_count", 0) or 0)
elif requested_job and requested_job.get("status") == "failed":
    st.session_state["eval_running"] = False


def editor_records(edited_data):
    if hasattr(edited_data, "to_dict"):
        return edited_data.to_dict(orient="records")
    if isinstance(edited_data, list):
        return edited_data
    raise TypeError("The dataset editor returned an unsupported data format.")


def render_metric_group_cards(container):
    """Render compact, accessible metric selectors in the sidebar."""
    container.markdown("### Evaluation metrics")
    container.caption("Choose the groups to score in the next evaluation run.")
    metric_cards = (
        (
            "run_component",
            "Component + Pipeline",
            "5 metrics",
            "Context precision, recall, relevancy, answer relevancy, and faithfulness.",
            "#38bdf8",
        ),
        (
            "run_quality",
            "Application Quality",
            "2 metrics",
            "Correctness and completeness against reviewed reference answers.",
            "#a78bfa",
        ),
        (
            "run_safety",
            "Application Safety",
            "3 checks",
            "Toxicity, PII leakage, and off-topic scope adherence.",
            "#fb7185",
        ),
    )
    selected_groups = {}
    for key, title, kicker, description, accent in metric_cards:
        selected = container.checkbox(
            f"Enable {title}",
            value=True,
            key=f"{key}_{workspace_id}",
            help=description,
        )
        selected_groups[key] = selected
        card_class = "metric-card selected" if selected else "metric-card"
        container.markdown(
            f"""
            <div class="{card_class}" style="--metric-accent:{accent}">
              <div class="metric-kicker">{kicker}</div>
              <div class="metric-title">{title}</div>
              <div class="metric-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return selected_groups


render_hero(
    "📊 RAG Evaluation Suite",
    "Component-level, pipeline-level, application-level (quality + safety), and operational metrics — all in one place.",
    ("Golden dataset", "Retriever comparison", "LLM-judge metrics", "PDF report"),
)

with st.sidebar:
    render_model_settings(st.sidebar)
    st.divider()
    selected_groups = render_metric_group_cards(st.sidebar)

golden_editor_key = f"golden_editor_{workspace_id}"
golden_upload_digest_key = f"golden_upload_digest_{workspace_id}"
golden_upload_widget_key = f"golden_dataset_upload_{workspace_id}"
retrieval_config = get_retrieval_settings()
collection = get_existing_collection(
    chroma_dir=workspace["chroma"],
    embedding_provider=retrieval_config["embedding_provider"],
    embedding_model=retrieval_config["embedding_model"],
)
if not collection:
    st.warning("Upload and index documents from the main page first.")
    st.stop()

retriever = get_session_retriever(workspace["chroma"], workspace_id)

tab_dataset, tab_retrieval, tab_run, tab_results = st.tabs([
    "1️⃣ Golden Dataset",
    "2️⃣ Retriever Comparison",
    "3️⃣ Run Evaluation",
    "4️⃣ Results & Report",
])

# =====================================================================
# TAB 1 — Golden Dataset
# =====================================================================
with tab_dataset:
    golden_data = load_golden_dataset(workspace["golden_dataset"])

    st.subheader("Generate from your papers")
    col1, col2 = st.columns([1, 1])

    with col1:
        n_questions = st.slider("How many questions to generate?", 1, MAX_GOLDEN_QUESTIONS, 25)
        est_batches = -(-n_questions // 5)
        st.caption(
            f"≈ {est_batches} generation calls at five chunks per batch. "
            "Provider latency, retries, duplicate questions, and rate limits can increase runtime."
        )

        if st.button("🎲 Auto-generate Golden Dataset", type="primary"):
            settings = get_model_settings("golden")
            try:
                settings = require_api_key("golden")
                progress_bar = st.progress(0, text="Preparing indexed chunks...")
                status_placeholder = st.empty()

                def _update_sampling(done, total):
                    ratio = done / total if total else 0.0
                    progress_bar.progress(
                        int(ratio * 35),
                        text=f"Scanning indexed chunks: {done:,}/{total:,}",
                    )

                def _update_progress(done, total, succeeded, failed):
                    ratio = done / total if total else 0.0
                    progress_bar.progress(
                        35 + int(ratio * 65),
                        text=f"Generating questions: {done}/{total} chunks processed...",
                    )
                    status_placeholder.caption(f"Batches: {succeeded} succeeded, {failed} failed")

                with st.spinner("Generating document-grounded questions..."):
                    golden_data, stats = generate_golden_dataset(
                        collection,
                        n=n_questions,
                        progress_callback=_update_progress,
                        model_name=settings["model"],
                        model_provider=settings["provider"],
                        api_key=settings["api_key"],
                        dataset_path=workspace["golden_dataset"],
                        sampling_progress_callback=_update_sampling,
                    )
                progress_bar.empty()
                status_placeholder.empty()
                st.session_state["golden_dataset"] = golden_data
                st.session_state.pop(golden_editor_key, None)
                st.session_state.pop(golden_upload_digest_key, None)
                st.session_state.pop(golden_upload_widget_key, None)
                clear_saved_results()
                if stats["produced"] < stats["requested"]:
                    st.warning(
                        f"Generated {stats['produced']} of {stats['requested']} requested questions "
                        f"({stats['duplicate_questions']} empty/duplicate outputs removed; "
                        f"{stats['batches_failed']}/{stats['batches_total']} batches failed). "
                        "The provider may also have returned fewer valid items than requested. "
                        "Retry or review the saved partial dataset."
                    )
                    for error in stats.get("errors", [])[:3]:
                        st.error(f"Generation diagnostic: {error}")
                else:
                    st.success(
                        f"Generated {stats['produced']} unique questions "
                        f"({stats['batches_succeeded']}/{stats['batches_total']} batches)."
                    )
                if stats["duplicate_questions"]:
                    st.caption(f"Removed {stats['duplicate_questions']} empty or duplicate questions.")
            except Exception as exc:
                try:
                    provider_url = provider_base_url(settings["provider"])
                except ValueError:
                    provider_url = None
                st.error(
                    format_llm_error(
                        exc,
                        settings["provider"],
                        settings["api_key"],
                        model_name=settings["model"],
                        base_url=provider_url,
                    )
                )

    with col2:
        st.markdown("**Or upload your own**")
        uploaded_golden = st.file_uploader(
            "golden_dataset.json",
            type=["json"],
            key=golden_upload_widget_key,
        )
        if uploaded_golden:
            try:
                upload_bytes = uploaded_golden.getvalue()
                if len(upload_bytes) > MAX_GOLDEN_DATASET_UPLOAD_BYTES:
                    raise ValueError("Golden dataset uploads must be 5MB or smaller.")
                upload_digest = hashlib.sha256(upload_bytes).hexdigest()
                if st.session_state.get(golden_upload_digest_key) != upload_digest:
                    parsed = validate_golden_dataset(json.loads(upload_bytes))
                    save_golden_dataset(parsed, workspace["golden_dataset"])
                    st.session_state["golden_dataset"] = parsed
                    st.session_state[golden_upload_digest_key] = upload_digest
                    st.session_state.pop(golden_editor_key, None)
                    clear_saved_results()
                    st.success(f"Loaded {len(parsed)} Q&A pairs.")
                    golden_data = parsed
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                st.error(f"Invalid golden dataset: {exc}")
        else:
            st.session_state.pop(golden_upload_digest_key, None)
        st.caption(
            "Provide a JSON array with `input` and `expected_output`; optional fields include "
            "`difficulty`, `source`, `page`, and `source_reference`."
        )

    st.divider()

    if golden_data:
        counts = difficulty_breakdown(golden_data)
        st.subheader(f"Current Dataset — {len(golden_data)} questions")
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Easy", counts.get("easy", 0))
        c2.metric("🟡 Medium", counts.get("medium", 0))
        c3.metric("🔴 Hard", counts.get("hard", 0))

        editor_rows = []
        for item in golden_data:
            difficulty = item.get("difficulty", "medium")
            if not isinstance(difficulty, str) or difficulty not in DIFFICULTY_LEVELS:
                difficulty = "medium"
            editor_rows.append({
                "input": item.get("input", ""),
                "expected_output": item.get("expected_output", ""),
                "difficulty": difficulty,
                "source": str(item.get("source") or ""),
                "page": str(item.get("page") or ""),
                "source_reference": str(item.get("source_reference") or ""),
            })
        edited_data = st.data_editor(
            editor_rows,
            key=golden_editor_key,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "input": st.column_config.TextColumn("Question", required=True),
                "expected_output": st.column_config.TextColumn("Reference answer", required=True),
                "difficulty": st.column_config.SelectboxColumn(
                    "Difficulty",
                    options=["easy", "medium", "hard"],
                ),
            },
        )
        if st.button("Save reviewed dataset", type="primary"):
            try:
                reviewed = validate_golden_dataset(editor_records(edited_data))
                save_golden_dataset(reviewed, workspace["golden_dataset"])
                st.session_state["golden_dataset"] = reviewed
                clear_saved_results()
                st.success("Reviewed dataset saved.")
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
    else:
        st.info("Generate or upload a golden dataset above to continue.")

# =====================================================================
# TAB 2 — Retriever Comparison
# =====================================================================
with tab_retrieval:
    comparison_data = load_golden_dataset(workspace["golden_dataset"])
    st.subheader("Compare retrieval strategies")
    st.caption(
        "Selected retrievers use the same ordered questions, embedding profile, candidate depth, K, and reranker. "
        "This comparison uses document references in the dataset and makes no answer-model or judge calls."
    )
    if not comparison_data:
        st.info("Create a golden dataset in the first tab before comparing retrievers.")
    else:
        method_labels = {
            "Dense Vector Search": "dense",
            "BM25 Keyword Search": "bm25",
            "Hybrid Search": "hybrid",
        }
        for registered_method in RETRIEVER_REGISTRY:
            if registered_method not in method_labels.values():
                method_labels[f"Custom: {registered_method}"] = registered_method
        selected_labels = st.multiselect(
            "Retrievers to compare",
            list(method_labels),
            default=list(method_labels),
            key=f"retrieval_compare_methods_{workspace_id}",
        )
        selected_methods = [method_labels[label] for label in selected_labels]
        retrieval_config = get_retrieval_settings()
        upper = len(comparison_data)
        sample_default = min(25, upper)
        sample_size = (
            st.slider(
                "Questions to compare",
                min_value=1,
                max_value=upper,
                value=sample_default,
                key=f"retrieval_compare_sample_{workspace_id}",
            )
            if upper > 1
            else 1
        )
        selected_k = st.slider(
            "K for retrieval metrics",
            min_value=1,
            max_value=30,
            value=min(10, max(1, retrieval_config["top_k"])),
            key=f"retrieval_compare_k_{workspace_id}",
        )
        st.caption(
            f"Embedding: {retrieval_config['embedding_provider']} / {retrieval_config['embedding_model']} · "
            f"candidate depth: {retrieval_config['candidate_k']} · "
            f"reranker: {'on' if retrieval_config['use_reranker'] else 'off'}"
        )
        if st.button(
            "Run retriever comparison",
            type="primary",
            disabled=not selected_methods,
            key=f"run_retrieval_compare_{workspace_id}",
        ):
            try:
                retrieval_config = require_embedding_settings(methods=selected_methods)
                comparison_retriever = Retriever(
                    chroma_dir=workspace["chroma"],
                    method=retrieval_config["method"],
                    embedding_provider=retrieval_config["embedding_provider"],
                    embedding_model=retrieval_config["embedding_model"],
                    api_key=retrieval_config["embedding_api_key"],
                    use_reranker=retrieval_config["use_reranker"],
                    reranker_model=retrieval_config["reranker_model"],
                    top_k=selected_k,
                    candidate_k=retrieval_config["candidate_k"],
                )
                try:
                    with st.spinner("Retrieving the selected question batches..."):
                        comparison = run_retrieval_comparison(
                            comparison_data[:sample_size],
                            comparison_retriever,
                            methods=selected_methods,
                            k=selected_k,
                            batch_size=32,
                        )
                finally:
                    comparison_retriever.close()
                st.session_state[f"retrieval_comparison_{workspace_id}"] = comparison
            except Exception as exc:
                st.error(str(exc))

        comparison = st.session_state.get(f"retrieval_comparison_{workspace_id}")
        if comparison:
            metric_rows = []
            for method, summary in comparison["methods"].items():
                metric_rows.append({
                    "Retriever": method,
                    **{
                        metric: (
                            f"{summary['metrics'][metric]:.3f}"
                            if summary["metrics"][metric] is not None
                            else "N/A"
                        )
                        for metric in ("Precision@K", "Recall@K", "MRR", "nDCG@K")
                    },
                    "Labeled questions": f"{summary['evaluated_questions']}/{summary['total_questions']}",
                })
                if summary["errors"]:
                    st.warning(
                        f"{method} had retrieval errors: {summary['errors'][0]['message']}"
                    )
            st.dataframe(metric_rows, width="stretch", hide_index=True)
            st.download_button(
                "Download comparison JSON",
                data=json.dumps(comparison, ensure_ascii=False, indent=2),
                file_name="retriever_comparison.json",
                mime="application/json",
                key=f"download_retrieval_compare_{workspace_id}",
            )
            if not any(
                summary["evaluated_questions"]
                for summary in comparison["methods"].values()
            ):
                st.warning(
                    "No dataset document references were available. Add source/page or relevant document IDs "
                    "to score retrieval metrics."
                )

with tab_retrieval:
    st.divider()
    st.subheader("🎯 Calibrate the scope check")
    st.caption(
        "The chat refuses questions whose best passage scores too low. Instead of guessing the cut-off, "
        "measure it on your own document using the golden dataset questions and a few unrelated probes. "
        "The result applies to chat answers in this session."
    )
    _active = st.session_state.get("scope_threshold")
    st.caption(
        f"Current scope threshold: {_active:.2f} (calibrated)" if _active is not None
        else "Current scope threshold: default from settings."
    )
    _calib_data = load_golden_dataset(workspace["golden_dataset"])
    if len(_calib_data) < 3:
        st.info("Create a golden dataset with at least 3 questions first.")
    elif st.button("Calibrate scope check", key=f"calibrate_scope_{workspace_id}"):
        try:
            _cfg = require_embedding_settings(methods=[get_retrieval_settings()["method"]])
            if not _cfg["use_reranker"]:
                raise ValueError("Turn on 'Rerank results with the local cross-encoder' in Search and embeddings first.")
            _retr = Retriever(
                chroma_dir=workspace["chroma"],
                method=_cfg["method"],
                embedding_provider=_cfg["embedding_provider"],
                embedding_model=_cfg["embedding_model"],
                api_key=_cfg["embedding_api_key"],
                use_reranker=True,
                reranker_model=_cfg["reranker_model"],
                top_k=_cfg["top_k"],
                candidate_k=_cfg["candidate_k"],
            )
            try:
                with st.spinner("Scoring your questions and unrelated probes..."):
                    _questions = [item["input"] for item in _calib_data[:100]]
                    _in = _retr.retrieve_many(_questions, methods=[_cfg["method"]])[_cfg["method"]]
                    _off = _retr.retrieve_many(list(OFF_TOPIC_PROBES), methods=[_cfg["method"]])[_cfg["method"]]
            finally:
                _retr.close()

            def _best(result_lists):
                values = []
                for chunks in result_lists:
                    scores = [c.get("rerank_score") for c in chunks if isinstance(c, dict) and c.get("rerank_score") is not None]
                    if scores:
                        values.append(max(scores))
                return values

            _in_scores, _off_scores = _best(_in), _best(_off)
            if not _in_scores:
                raise ValueError("The reranker returned no scores, so the scope check cannot be calibrated.")
            st.session_state[f"scope_calibration_{workspace_id}"] = calibrate_scope_threshold(_in_scores, _off_scores)
        except Exception as exc:
            st.error(str(exc))
    _calibration = st.session_state.get(f"scope_calibration_{workspace_id}")
    if _calibration:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Suggested threshold", f"{_calibration['threshold']:.2f}")
        cc2.metric("Your questions kept", f"{_calibration['in_scope_kept']*100:.0f}%")
        cc3.metric(
            "Unrelated probes blocked",
            "—" if _calibration["off_topic_blocked"] is None else f"{_calibration['off_topic_blocked']*100:.0f}%",
        )
        if _calibration["separable"]:
            st.success("Your questions and the unrelated probes separate cleanly, so this cut-off blocks off-topic questions.")
        else:
            st.warning(
                "Some unrelated probes score as high as real questions, so the cut-off favours answering. "
                "The answer prompt still declines questions the passages do not support."
            )
        if st.button("Use this threshold in chat", key=f"apply_scope_{workspace_id}"):
            st.session_state["scope_threshold"] = _calibration["threshold"]
            st.success("Applied. New chat questions use this threshold until the session ends.")


# =====================================================================
# TAB 3 — Run Evaluation
# =====================================================================
with tab_run:
    golden_data = load_golden_dataset(workspace["golden_dataset"])
    if not golden_data:
        st.info("Create a golden dataset in the first tab before running evaluations.")
    else:
        st.warning(
            "Evaluations use the configured judge model and respect the shared request pacing. "
            "The selected metrics are run together over one generated answer set to avoid duplicate generation. "
            "Large samples can still take time and may trigger provider-specific quota waits.",
            icon="⚠️",
        )

        max_sample = len(golden_data)
        if max_sample <= 1:
            st.caption("The dataset has one question, so it will be evaluated automatically.")
            eval_sample_size = 1
        else:
            eval_sample_size = st.slider(
                "How many questions to evaluate on (out of the full dataset)?",
                1, max_sample, min(10, max_sample),
            )
        eval_golden_data = golden_data[:eval_sample_size]
        if eval_sample_size < 20:
            st.warning(
                f"This run uses {eval_sample_size} question(s). Treat the scores as an early diagnostic; "
                "build and review a larger, document-specific golden set before using results as deployment evidence."
            )

        st.markdown("#### Selected evaluation groups")
        st.caption("Configure the highlighted metric cards in the sidebar. All selected groups reuse one generated answer set.")
        run_component = selected_groups["run_component"]
        run_quality = selected_groups["run_quality"]
        run_safety = selected_groups["run_safety"]

        st.markdown("#### Speed and provider plan")
        speed_col, plan_col = st.columns(2)
        with speed_col:
            fast_mode = st.toggle(
                "Fast mode (skip written explanations)",
                value=False,
                help="Scores stay the same; DeepEval skips the extra 'reason' call per metric, "
                "which removes roughly a quarter of the judge requests. Explanations will be empty.",
            )
        with plan_col:
            pacing_profile = st.selectbox(
                "Provider plan",
                list(PACING_PROFILES),
                format_func=lambda key: PACING_PROFILE_LABELS[key],
                help="Free tiers need slower pacing to avoid 429 errors. Paid keys can run at full speed. "
                "Auto starts fast and slows down only for the provider that rate-limits you.",
            )
        run_options = {
            "run_component": run_component,
            "run_quality": run_quality,
            "run_safety": run_safety,
            "include_reasons": not fast_mode,
            "pacing": pacing_profile,
        }
        retrieval_top_k = int(retrieval_config.get("top_k", 4) or 4)
        plan = estimate_run(eval_sample_size, run_options, top_k=retrieval_top_k)
        answer_provider = get_model_settings("answer")["provider"]
        judge_provider = get_model_settings("judge")["provider"]
        judge_requests = plan["metric_calls"] + plan["scope_judges"]
        answer_requests = plan["answers"] + plan["scope_answers"]
        with rate_limit_scope(f"{workspace_id}-estimate", pacing_profile):
            pacing_seconds = (
                max(0, judge_requests - 1) * get_min_interval_seconds(judge_provider)
                + max(0, answer_requests - 1) * get_min_interval_seconds(answer_provider)
            )
        st.caption(
            f"Expect about {plan['minimum']}–{plan['maximum']} provider requests "
            f"({plan['answers']} answers, {plan['metric_calls']} metric judge calls, "
            f"{plan['scope_judges']} scope checks, up to {plan['scope_answers']} scope answers). "
            f"Pacing alone adds about {pacing_seconds / 60:.1f} min; latency and quota waits add more. "
            "Cached answers reduce the count, and failed attempts that are retried do not move the progress bar."
        )

        no_eval_selected = not (run_component or run_quality or run_safety)
        if no_eval_selected:
            st.info("Select at least one evaluation group to start.")

        if st.button(
            "▶️ Run Selected Evaluations",
            type="primary",
            disabled=st.session_state.get("eval_running", False) or no_eval_selected,
        ):
            try:
                answer_settings = require_api_key("answer")
                judge_settings = require_api_key("judge")
                retrieval_settings = require_embedding_settings()
                eval_golden_data = validate_golden_dataset(eval_golden_data)
                job_id = new_job_id()
                queued = submit_evaluation_job(
                    job_id=job_id,
                    workspace_id=workspace_id,
                    golden_data=eval_golden_data,
                    options=run_options,
                    answer_settings=answer_settings,
                    judge_settings=judge_settings,
                    retrieval_settings=retrieval_settings,
                )
                if not queued:
                    st.warning("The evaluation queue is full. Wait for the active run to finish, then retry.")
                else:
                    clear_saved_results()
                    st.session_state["eval_running"] = True
                    st.session_state["active_eval_job_id"] = job_id
                    st.session_state.pop("eval_loaded_job_id", None)
                    st.query_params["evaluation_job"] = job_id
                    st.rerun()
            except Exception as exc:
                logger.warning("Could not start evaluation (%s).", type(exc).__name__)
                st.error(f"Could not start evaluation ({type(exc).__name__}): {exc}")

        active_job_id = st.session_state.get("active_eval_job_id") or requested_job_id
        active_job = load_evaluation_job(active_job_id, workspace_id=workspace_id)
        if active_job:
            poll_interval = "1s" if active_job.get("status") in {"queued", "running"} else None

            @st.fragment(run_every=poll_interval)
            def render_evaluation_job_status(job_id):
                current_job = load_evaluation_job(job_id, workspace_id=workspace_id)
                if not current_job:
                    st.warning("The saved evaluation status has expired. Start a new run to continue.")
                    return
                status = current_job.get("status")
                if status in {"queued", "running"}:
                    if current_job.get("reranker_ready") is False:
                        st.warning(
                            "The local cross-encoder is unavailable; this run uses the selected retriever's base ranking."
                        )
                    percent = max(0, min(99, int(current_job.get("progress", 0))))
                    completed = int(current_job.get("requests_completed", 0) or 0)
                    expected = int(current_job.get("expected_requests", 0) or 0)
                    active = int(current_job.get("active_requests", 0) or 0)
                    api_calls = int(current_job.get("model_api_calls", 0) or 0)
                    cache_hits = int(current_job.get("cache_hits", 0) or 0)
                    failed_requests = int(current_job.get("failed_requests", 0) or 0)
                    average = current_job.get("average_request_seconds")
                    request_started_text = current_job.get("active_request_started_at")
                    if active and request_started_text and expected:
                        try:
                            request_age = max(
                                0.0,
                                (datetime.now(timezone.utc) - datetime.fromisoformat(request_started_text)).total_seconds(),
                            )
                            request_seconds = float(average or 15.0)
                            in_flight_fraction = min(0.85, request_age / request_seconds)
                            smooth_percent = int(2 + 93 * (completed + in_flight_fraction) / expected)
                            percent = max(percent, min(99, smooth_percent))
                        except (TypeError, ValueError, ZeroDivisionError):
                            pass
                    st.progress(
                        percent,
                        text=f"{percent}% complete - {current_job.get('phase', 'Evaluation in progress')}",
                    )
                    started_text = current_job.get("started_at") or current_job.get("created_at")
                    elapsed = 0.0
                    if started_text:
                        try:
                            elapsed = max(
                                0.0,
                                (datetime.now(timezone.utc) - datetime.fromisoformat(started_text)).total_seconds(),
                            )
                        except (TypeError, ValueError):
                            pass
                    elapsed_label = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                    remaining = max(0, expected - completed)
                    if average and remaining:
                        eta = max(0, int(float(average) * remaining))
                        eta_label = f"about {int(eta // 60):02d}:{int(eta % 60):02d} remaining"
                    elif completed == 0:
                        eta_label = "estimating remaining time from the first model response"
                    else:
                        eta_label = "almost done; finishing evaluation calculations"
                    request_label = (
                        f"{api_calls} actual API calls"
                        f" | {cache_hits} cache hits"
                        f" | {failed_requests} failures"
                        f" | {completed}/{expected} attempts completed"
                    )
                    if active:
                        request_label += " | model response in progress"
                    st.caption(f"{request_label} | elapsed {elapsed_label} | {eta_label}")
                    st.caption(
                        "Progress and remaining time are estimates; provider latency, retries, and quota waits can change them."
                    )
                    st.info(
                        f"{current_job.get('phase', 'Evaluation in progress')} - you can leave this tab open; "
                        "it refreshes automatically."
                    )
                elif status == "complete":
                    st.progress(100, text="100% complete - Evaluation finished")
                    st.caption(
                        f"{int(current_job.get('model_api_calls', 0) or 0)} actual API calls · "
                        f"{int(current_job.get('cache_hits', 0) or 0)} cache hits · "
                        f"{int(current_job.get('failed_requests', 0) or 0)} failures"
                    )
                    if st.session_state.get("eval_loaded_job_id") != job_id:
                        clear_saved_results()
                        st.session_state.update(current_job.get("result") or {})
                        st.session_state["eval_loaded_job_id"] = job_id
                        st.session_state["eval_result_sample_count"] = int(current_job.get("sample_count", 0) or 0)
                        st.session_state["eval_running"] = False
                        st.rerun()
                    run_notes = (current_job.get("result") or {}).get("run_notes") or []
                    if run_notes:
                        st.warning("**Evaluation finished with problems.** Some scores are missing.")
                        for note in run_notes:
                            st.warning(note)
                    else:
                        st.success("Evaluation complete. Results are ready in the **Results & Report** tab.")
                elif status == "failed":
                    st.session_state["eval_running"] = False
                    st.error(
                        f"Evaluation failed ({current_job.get('error_type', 'Error')}): "
                        f"{current_job.get('error_message', 'No additional details were recorded.')}"
                    )
                    st.info("Check provider connectivity, API credentials, quota/reset information, and the server log before retrying.")

            render_evaluation_job_status(active_job_id)
        elif active_job_id:
            st.session_state["eval_running"] = False
            st.session_state.pop("active_eval_job_id", None)
            st.warning(
                "This evaluation's saved status could not be found. The Run button is available again; "
                "start a new evaluation if needed."
            )

# =====================================================================
# TAB 4 — Results & Report
# =====================================================================
with tab_results:
    result_sample_count = int(st.session_state.get("eval_result_sample_count", 0) or 0)
    if 0 < result_sample_count < 20:
        st.warning(
            f"This report is based on {result_sample_count} question(s), so treat its scores as preliminary. "
            "Use a larger reviewed dataset before making a deployment decision."
        )

    for _note in st.session_state.get("run_notes", []) or []:
        st.error(f"⚠️ Incomplete results: {_note}")

    def render_metric_table(summary: dict, title: str):
        st.subheader(title)
        rows = []
        for name, v in summary.items():
            if name.startswith("_"):
                continue
            if v["total"] == 0:
                # Nothing was scored: show "no data", never a misleading 0.00.
                rows.append({"Metric": name, "Avg Score": "—", "Pass Rate": "—",
                             "Passed/Total": f"0/0 ({v.get('errored', 0)} errored)", "Status": "⚠️ no data"})
                continue
            status = "✅" if v["average"] >= 0.75 else "❌"
            rows.append({"Metric": name, "Avg Score": f"{v['average']:.2f}", "Pass Rate": f"{v['pass_rate']*100:.0f}%",
                         "Passed/Total": f"{v['passed']}/{v['total']}", "Status": status})
        st.dataframe(rows, width="stretch", hide_index=True)

        # Show WHY low-scoring metrics failed, using the judge's own reasons
        for name, v in summary.items():
            if name.startswith("_") or v["average"] >= 0.75 or not v.get("failure_reasons"):
                continue
            with st.expander(f"🔍 Why did '{name}' score low? (avg {v['average']:.2f})"):
                for r in v["failure_reasons"][:3]:
                    st.markdown(f"**Q:** {neutralize_markdown(r['input'][:150])}")
                    st.markdown(f"**Score:** {r['score']:.2f} — {neutralize_markdown(str(r['reason']))}")
                    st.divider()

    if "retriever_summary" in st.session_state:
        render_metric_table(st.session_state["retriever_summary"], "🔍 Retriever (Component-Level)")
        render_metric_table(st.session_state["generator_summary"], "✍️ Generator (Component-Level)")

    if "pipeline_summary" in st.session_state:
        ps = st.session_state["pipeline_summary"]
        st.subheader("🔗 Pipeline-Level (RAG Triad)")
        st.metric("Questions passing ALL metrics", f"{ps['fully_passed']}/{ps['total_questions']}",
                   f"{ps['fully_passed_rate']*100:.0f}%")
        st.caption(
            "Strict view: a question counts only if every metric passes together, so one weak metric "
            "can make this 0 even when the other metrics are fine. See partial credit below."
        )
        partial = ps.get("partial")
        if partial and partial["scored_questions"]:
            pc1, pc2 = st.columns(2)
            pc1.metric(
                "Questions passing most metrics (≥75%)",
                f"{partial['mostly_passed']}/{partial['scored_questions']}",
                f"{partial['mostly_passed_rate']*100:.0f}%",
            )
            pc2.metric(
                "Average metrics passed per question",
                f"{partial['avg_metrics_passed']:.1f}/{partial['avg_metrics_scored']:.1f}",
            )
            if partial["incomplete_questions"]:
                st.warning(
                    f"{partial['incomplete_questions']} question(s) had at least one metric that could not be "
                    "scored (judge error or quota). They are counted as not fully passing."
                )

        db = ps.get("difficulty_breakdown")
        if db:
            st.markdown("**Pass rate by difficulty**")
            c1, c2, c3 = st.columns(3)
            for col, level, emoji in zip([c1, c2, c3], ["easy", "medium", "hard"], ["🟢", "🟡", "🔴"]):
                d = db.get(level, {})
                if d.get("total"):
                    col.metric(f"{emoji} {level.capitalize()}", f"{d['passed']}/{d['total']}", f"{d['pass_rate']*100:.0f}%")
                else:
                    col.metric(f"{emoji} {level.capitalize()}", "—")

        render_metric_table(ps["metric_summary"], "Combined Metrics")

    if "quality_summary" in st.session_state:
        render_metric_table(st.session_state["quality_summary"], "✨ Application Quality")

    if "safety_content_summary" in st.session_state:
        render_metric_table(st.session_state["safety_content_summary"], "🛡️ Safety — Toxicity & PII")
    if "scope_summary" in st.session_state:
        render_metric_table(st.session_state["scope_summary"], "🎯 Safety — Scope Adherence")
        fast_rate = st.session_state["scope_summary"].get("_fast_check_catch_rate")
        if fast_rate is not None:
            st.caption(f"The fast heuristic filter caught {fast_rate*100:.0f}% of off-topic probes with zero LLM cost.")

    result_sections = {
        "retriever": st.session_state.get("retriever_summary", {}),
        "generator": st.session_state.get("generator_summary", {}),
        "quality": st.session_state.get("quality_summary", {}),
        "safety": st.session_state.get("safety_content_summary", {}),
        "scope": st.session_state.get("scope_summary", {}),
    }
    metric_rows = []
    for section, summary in result_sections.items():
        for name, values in summary.items():
            if name.startswith("_") or not isinstance(values, dict) or "average" not in values:
                continue
            metric_rows.append(
                {
                    "section": section,
                    "metric": name,
                    "average": values["average"],
                    "pass_rate": values["pass_rate"],
                    "passed": values["passed"],
                    "failed": values["failed"],
                    "total": values["total"],
                    "errored": values.get("errored", 0),
                    "failure_reasons": values.get("failure_reasons", []),
                }
            )
    retrieval_comparison = st.session_state.get(f"retrieval_comparison_{workspace_id}")
    if retrieval_comparison:
        for method, summary in retrieval_comparison["methods"].items():
            for name, average in summary["metrics"].items():
                if average is None:
                    continue
                metric_rows.append({
                    "section": "retrieval_comparison",
                    "metric": f"{method}: {name}",
                    "average": average,
                    "pass_rate": None,
                    "passed": None,
                    "failed": None,
                    "total": summary["total_questions"],
                    "failure_reasons": [],
                    "label_coverage": (
                        f"{summary['evaluated_questions']}/{summary['total_questions']}"
                    ),
                })
    if metric_rows:
        st.subheader("Metric score overview")
        chart_data = pd.DataFrame(
            [
                {"Metric": f"{row['section'].title()}: {row['metric']}", "Average score": row["average"]}
                for row in metric_rows
                if row["total"]  # metrics with no data would draw a misleading empty bar
            ]
        ).set_index("Metric")
        if not chart_data.empty:
            st.bar_chart(chart_data, horizontal=True)

    st.subheader("⚡ Operational Metrics (from live chat usage)")
    om = compute_operational_metrics(workspace["traces"])
    if om:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P50 Latency", f"{om['p50_latency_s']:.2f}s" if om["p50_latency_s"] is not None else "—")
        c2.metric("P99 Latency", f"{om['p99_latency_s']:.2f}s" if om['p99_latency_s'] is not None else "—")
        c3.metric("Cache Hit Rate", f"{om['cache_hit_rate']*100:.0f}%")
        c4.metric("Total Cost", f"${om['total_cost_usd']:.4f}")
    else:
        st.caption("Chat with the assistant on the main page to generate operational data.")

    st.divider()

    st.subheader("📦 Download Evaluation Data")
    if metric_rows:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metric_rows,
            "pipeline": st.session_state.get("pipeline_summary"),
            "operational": om,
            "retrieval_comparison": retrieval_comparison,
        }
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in metric_rows)
        csv_buffer = io.StringIO()
        csv_writer = csv.DictWriter(
            csv_buffer,
            fieldnames=[
                "section", "metric", "average", "pass_rate", "passed", "failed",
                "total", "errored", "label_coverage",
            ],
        )
        csv_writer.writeheader()
        csv_writer.writerows(
            {key: row.get(key) for key in csv_writer.fieldnames}
            for row in metric_rows
        )
        download_columns = st.columns(3)
        download_columns[0].download_button(
            "Download JSON",
            data=report_json,
            file_name="llm_evaluation_report.json",
            mime="application/json",
            width="stretch",
        )
        download_columns[1].download_button(
            "Download JSONL",
            data=jsonl,
            file_name="llm_evaluation_metrics.jsonl",
            mime="application/x-ndjson",
            width="stretch",
        )
        download_columns[2].download_button(
            "Download CSV",
            data=csv_buffer.getvalue(),
            file_name="llm_evaluation_metrics.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("Run at least one evaluation group to download results.")

    st.subheader("📄 Download PDF Report")
    required_keys = [
        "retriever_summary",
        "generator_summary",
        "pipeline_summary",
        "quality_summary",
        "safety_content_summary",
        "scope_summary",
    ]
    required_metrics = {
        "Component retrieval": st.session_state.get("retriever_summary"),
        "Component generation": st.session_state.get("generator_summary"),
        "Pipeline": (st.session_state.get("pipeline_summary") or {}).get("metric_summary"),
        "Application quality": st.session_state.get("quality_summary"),
        "Application safety": st.session_state.get("safety_content_summary"),
        "Scope adherence": st.session_state.get("scope_summary"),
    }
    missing_report_sections = [
        label
        for label, summary in required_metrics.items()
        if not isinstance(summary, dict)
        or not any(not name.startswith("_") for name in summary)
    ]
    has_any_results = len(missing_report_sections) < len(required_metrics)
    if has_any_results:
        try:
            pdf_bytes = build_pdf_report(
                st.session_state.get("retriever_summary"), st.session_state.get("generator_summary"),
                st.session_state.get("pipeline_summary"), st.session_state.get("quality_summary"),
                st.session_state.get("safety_content_summary"), st.session_state.get("scope_summary"),
                om,
                run_notes=st.session_state.get("run_notes"),
            )
        except Exception as exc:  # a report problem must never break the results page
            pdf_bytes = None
            st.error(f"The PDF report could not be built ({type(exc).__name__}). JSON and CSV downloads still work.")
        if pdf_bytes:
            st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="rag_evaluation_report.pdf", mime="application/pdf")
        if missing_report_sections:
            st.caption(
                "Not run yet (shown as 'not run' in the PDF): " + ", ".join(missing_report_sections) + ". "
                "Run those groups too for a complete report."
            )
    else:
        st.info("Run at least one evaluation group, then return here to download the PDF report.")
