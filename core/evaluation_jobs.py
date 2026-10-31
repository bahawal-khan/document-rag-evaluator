"""Small, process-local background evaluation queue with durable job status."""

from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

from core.config import DATA_DIR, EVALUATION_REQUEST_TIMEOUT_SECONDS, RERANKER_MODEL
from core.embeddings import normalize_embedding_settings
from core.golden import validate_golden_dataset
from core.logging_utils import log_event
from core.eval_plan import ProgressTracker, estimate_run
from core.rate_limit import PACING_PROFILES, rate_limit_scope
from core.runtime import evaluation_event_loop

logger = logging.getLogger(__name__)
_JOB_ID_RE = re.compile(r"[0-9a-f]{32}")


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


# Evaluation is CPU/RAM heavy (embeddings + reranker) and Streamlit Community
# Cloud has ~2.7 GB, so the default is conservative. Raise EVAL_MAX_WORKERS on
# a bigger host; every user's run is isolated (own rate-limit scope, own judge).
EVAL_MAX_WORKERS = _env_int("EVAL_MAX_WORKERS", 2, 1, 16)
EVAL_QUEUE_SLOTS = _env_int("EVAL_QUEUE_SLOTS", EVAL_MAX_WORKERS * 4, EVAL_MAX_WORKERS, 256)
_executor = ThreadPoolExecutor(max_workers=EVAL_MAX_WORKERS, thread_name_prefix="rag-evaluation")
_queue_slots = threading.BoundedSemaphore(EVAL_QUEUE_SLOTS)
_workspace_lock = threading.Lock()
_active_workspaces: set[str] = set()
_live_jobs: set[str] = set()  # jobs this process is actually running or has queued


def _release_workspace(workspace_id: str) -> None:
    with _workspace_lock:
        _active_workspaces.discard(workspace_id)


def new_job_id() -> str:
    return uuid4().hex


def _job_path(job_id: str) -> Path:
    if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError("Evaluation job ID is invalid.")
    return DATA_DIR / "evaluation_jobs" / f"{job_id}.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_job(job_id: str, record: dict) -> None:
    path = _job_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(_without_credentials(record), ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _without_credentials(value):
    """Drop credential-named fields before any job status reaches disk."""
    sensitive_names = ("api_key", "apikey", "secret", "authorization", "credential", "access_token")
    if isinstance(value, dict):
        return {
            key: _without_credentials(item)
            for key, item in value.items()
            if not any(name in str(key).casefold() for name in sensitive_names)
        }
    if isinstance(value, list):
        return [_without_credentials(item) for item in value]
    if isinstance(value, tuple):
        return [_without_credentials(item) for item in value]
    return value


def load_evaluation_job(
    job_id: str | None,
    *,
    workspace_id: str | None = None,
) -> dict | None:
    if not job_id:
        return None
    try:
        record = json.loads(_job_path(job_id).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        if workspace_id is not None and record.get("workspace_id") != workspace_id:
            return None
        if record.get("status") in {"queued", "running"} and job_id not in _live_jobs:
            # The record says "running" but this process knows nothing about it: the app
            # restarted (Streamlit Cloud does this when idle) and the job died with it.
            record = {
                **record,
                "status": "failed",
                "phase": "Evaluation interrupted",
                "error_type": "Interrupted",
                "error_message": "The app restarted while this evaluation was running. Start it again.",
            }
        return record
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def delete_workspace_evaluation_jobs(workspace_id: str) -> None:
    """Remove saved result/status records when their workspace is cleared."""
    if not re.fullmatch(r"[0-9a-f]{32}", workspace_id or ""):
        raise ValueError("Workspace ID is invalid.")
    job_dir = DATA_DIR / "evaluation_jobs"
    if not job_dir.exists():
        return
    for path in job_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(record, dict)
                and record.get("workspace_id") == workspace_id
                and record.get("status") not in {"queued", "running"}
            ):
                path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            continue

def _update_job(job_id: str, **updates) -> None:
    record = load_evaluation_job(job_id) or {"job_id": job_id}
    record.update(updates)
    record["updated_at"] = _timestamp()
    _write_job(job_id, record)


def _redact_error(error: BaseException, secrets: list[str]) -> str:
    parts = []
    current = error
    seen = set()
    for _ in range(3):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        detail = str(current).strip()
        if detail:
            parts.append(f"{type(current).__name__}: {detail}")
        current = current.__cause__ or current.__context__
    message = " | caused by: ".join(parts) or type(error).__name__
    for secret in secrets:
        if secret:
            message = re.sub(re.escape(secret), "[redacted]", message, flags=re.IGNORECASE)
    return message[:2000]


def _expected_request_count(golden_data: list[dict], options: dict, top_k: int = 4) -> int:
    return estimate_run(len(golden_data), options, top_k=top_k)["maximum"]


def estimate_evaluation_requests(golden_data: list[dict], options: dict, top_k: int = 4) -> int:
    """Return the logical provider-request estimate shown before a run."""
    return _expected_request_count(validate_golden_dataset(golden_data), options, top_k)


def _run_job(
    job_id: str,
    workspace_id: str,
    golden_data: list[dict],
    options: dict,
    answer_settings: dict,
    judge_settings: dict,
    retrieval_settings: dict,
) -> None:
    expected_requests = _expected_request_count(golden_data, options, int(retrieval_settings.get("top_k", 4)))
    progress_lock = threading.Lock()
    tracker = ProgressTracker(expected_requests)

    def report_progress(event: str, **details) -> None:
        try:
            with progress_lock:
                fields = tracker.record(event, **details)
                _update_job(
                    job_id,
                    active_request_started_at=_timestamp() if event == "start" else None,
                    **fields,
                )
        except Exception:
            logger.debug("Could not save progress for evaluation job %s.", job_id, exc_info=True)

    retriever = None
    llm = None
    judge = None
    try:
        _update_job(
            job_id,
            status="running",
            phase="Preparing evaluation models",
            progress=2,
            requests_completed=0,
            expected_requests=expected_requests,
            active_requests=0,
            model_api_calls=0,
            cache_hits=0,
            failed_requests=0,
            average_request_seconds=None,
            started_at=_timestamp(),
        )

        from core.generator import build_generator
        from core.judge import LLMJudge
        from core.retriever import Retriever
        from eval.application_eval import run_scope_adherence_eval
        from eval.suite import run_selected_evaluations
        from core.generator import format_llm_error
        from core.provider import provider_base_url
        from core.evaluation_cache import (
            evaluation_key,
            get_cached_evaluation,
            set_cached_evaluation,
        )

        workspace_root = DATA_DIR / "workspaces" / workspace_id
        evaluation_cache_file = workspace_root / "evaluation_results_cache.json"
        cache_key = evaluation_key(
            golden_data,
            options,
            answer_settings,
            judge_settings,
            retrieval_settings,
        )
        cached_result = get_cached_evaluation(evaluation_cache_file, cache_key)
        if cached_result is not None:
            _update_job(
                job_id,
                status="complete",
                phase="Reused cached evaluation result",
                progress=100,
                active_requests=0,
                model_api_calls=0,
                cache_hits=expected_requests,
                failed_requests=0,
                result=cached_result,
                finished_at=_timestamp(),
            )
            return

        retriever = Retriever(
            chroma_dir=workspace_root / "chroma",
            method=retrieval_settings["method"],
            embedding_provider=retrieval_settings["embedding_provider"],
            embedding_model=retrieval_settings["embedding_model"],
            api_key=retrieval_settings.get("embedding_api_key", ""),
            use_reranker=retrieval_settings["use_reranker"],
            top_k=retrieval_settings["top_k"],
            candidate_k=retrieval_settings["candidate_k"],
            reranker_model=retrieval_settings["reranker_model"],
        )
        _update_job(job_id, phase="Preparing configured retrieval")
        reranker_ready = (
            retriever.warm_up_reranker(retriever.reranker_model)
            if retriever.use_reranker
            else None
        )
        _update_job(
            job_id,
            phase=(
                "Configured retrieval and reranker ready"
                if reranker_ready
                else "Configured retrieval ready; reranker disabled or unavailable"
            ),
            reranker_ready=reranker_ready,
        )
        llm = build_generator(
            streaming=False,
            timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS,
            model_name=answer_settings["model"],
            provider=answer_settings["provider"],
            api_key=answer_settings["api_key"],
            max_tokens=retrieval_settings.get("max_output_tokens"),
        )
        judge = LLMJudge(
            model_name=judge_settings["model"],
            provider=judge_settings["provider"],
            api_key=judge_settings["api_key"],
            progress_callback=report_progress,
            cache_file=workspace_root / "answer_cache.json",
        )

        _update_job(job_id, phase="Generating answers and scoring selected metrics")
        # Defence in depth: evaluation itself is synchronous, but if any library
        # still opens an event loop it is owned (and closed) by this job.
        with evaluation_event_loop():
            results = run_selected_evaluations(
                golden_data,
                retriever,
                llm,
                judge,
                run_component=options["run_component"],
                run_quality=options["run_quality"],
                run_safety=options["run_safety"],
                answer_provider=answer_settings["provider"],
                answer_model=answer_settings["model"],
                cache_file=workspace_root / "answer_cache.json",
                progress_callback=report_progress,
                include_reasons=options.get("include_reasons", True),
                max_input_tokens=retrieval_settings.get("max_input_tokens"),
            )
            results.pop("raw_result", None)

            if options["run_safety"] and getattr(judge, "quota_error", None) is not None:
                # The judge quota is already exhausted; the scope probes would only spend answer calls.
                results.setdefault("run_notes", []).append(
                    "Off-topic scope probes were skipped because the judge quota is exhausted."
                )
            elif options["run_safety"]:
                _update_job(job_id, phase="Checking off-topic scope probes")
                results["scope_summary"] = run_scope_adherence_eval(
                    retriever,
                    llm,
                    judge,
                    provider=answer_settings["provider"],
                    model_name=answer_settings["model"],
                    cache_file=workspace_root / "answer_cache.json",
                    progress_callback=report_progress,
                    max_input_tokens=retrieval_settings.get("max_input_tokens"),
                )

        if not results.get("run_notes"):
            # An incomplete run (quota hit, errored metrics) must never be replayed from cache.
            set_cached_evaluation(evaluation_cache_file, cache_key, results)
        _update_job(
            job_id,
            status="complete",
            phase="Evaluation complete",
            progress=100,
            active_requests=0,
            model_api_calls=tracker.api_calls,
            cache_hits=tracker.cache_hits,
            failed_requests=tracker.failed,
            result=results,
            retrieval_profile={
                "method": retrieval_settings["method"],
                "embedding_provider": retrieval_settings["embedding_provider"],
                "embedding_model": retrieval_settings["embedding_model"],
                "use_reranker": bool(retrieval_settings["use_reranker"]),
                "reranker_model": retrieval_settings["reranker_model"],
                "top_k": int(retrieval_settings["top_k"]),
                "candidate_k": int(retrieval_settings["candidate_k"]),
            },
            finished_at=_timestamp(),
        )
    except Exception as exc:
        log_event(
            logger,
            logging.WARNING,
            "evaluation_job_failed",
            error_type=type(exc).__name__,
            status="failed",
        )
        safe_message = _redact_error(
            exc,
            [
                answer_settings.get("api_key", ""),
                judge_settings.get("api_key", ""),
                retrieval_settings.get("embedding_api_key", ""),
            ],
        )
        try:
            provider_url = provider_base_url(
                answer_settings.get("provider", ""),
            )
        except ValueError:
            provider_url = None
        safe_message = format_llm_error(
            exc,
            answer_settings.get("provider", "groq"),
            model_name=answer_settings.get("model"),
            base_url=provider_url,
        ) if "connection" in safe_message.lower() or "connecterror" in safe_message.lower() else safe_message
        _update_job(
            job_id,
            status="failed",
            phase="Evaluation failed",
            error_type=type(exc).__name__,
            error_message=safe_message,
            model_api_calls=tracker.api_calls,
            cache_hits=tracker.cache_hits,
            failed_requests=max(1, tracker.failed),
            finished_at=_timestamp(),
        )
    finally:
        if retriever is not None:
            retriever.close()
        from core.provider import close_chat_model

        close_chat_model(llm)
        close_chat_model(getattr(judge, "model", None))


def _execute_job(
    job_id: str,
    workspace_id: str,
    golden_data: list[dict],
    options: dict,
    answer_settings: dict,
    judge_settings: dict,
    retrieval_settings: dict,
) -> None:
    """Run one job inside its owner's rate-limit scope so users never throttle each other."""
    try:
        with rate_limit_scope(workspace_id, options.get("pacing", "auto")):
            _run_job(
                job_id, workspace_id, golden_data, options,
                answer_settings, judge_settings, retrieval_settings,
            )
    except BaseException as exc:
        # _run_job records its own failures; this catches a crash outside its try block.
        logger.warning("Evaluation job %s crashed (%s).", job_id, type(exc).__name__)
        try:
            _update_job(
                job_id,
                status="failed",
                phase="Evaluation failed",
                error_type=type(exc).__name__,
                error_message="The evaluation stopped unexpectedly. Please start it again.",
                finished_at=_timestamp(),
            )
        except Exception:
            pass
    finally:
        # Always give the worker slot and this user's one-run lock back, whatever happened.
        _queue_slots.release()
        _release_workspace(workspace_id)
        with _workspace_lock:
            _live_jobs.discard(job_id)


def submit_evaluation_job(
    *,
    job_id: str,
    workspace_id: str,
    golden_data: list[dict],
    options: dict,
    answer_settings: dict,
    judge_settings: dict,
    retrieval_settings: dict,
) -> bool:
    """Queue a run; keep at most one active and one waiting across sessions."""
    if not _JOB_ID_RE.fullmatch(job_id or ""):
        raise ValueError("Evaluation job ID is invalid.")
    if not _JOB_ID_RE.fullmatch(workspace_id or ""):
        raise ValueError("Workspace ID is invalid.")
    golden_data = validate_golden_dataset(golden_data)
    group_options = {"run_component", "run_quality", "run_safety"}
    allowed_options = group_options | {"include_reasons", "pacing"}
    if not isinstance(options, dict) or set(options) - allowed_options:
        raise ValueError("Evaluation options are invalid.")
    for key in group_options | {"include_reasons"}:
        if key in options and not isinstance(options[key], bool):
            raise ValueError("Evaluation group selections must be boolean values.")
    pacing = options.get("pacing", "auto")
    if pacing not in PACING_PROFILES:
        raise ValueError("Choose a supported provider plan.")
    if not any(bool(options.get(key)) for key in group_options):
        raise ValueError("Select at least one supported evaluation group.")
    include_reasons = bool(options.get("include_reasons", True))
    options = {key: bool(options.get(key)) for key in group_options}
    options["include_reasons"] = include_reasons
    options["pacing"] = pacing
    supported_providers = {"openai", "gemini", "anthropic", "groq", "openrouter", "ollama"}
    clean_settings = []
    for settings in (answer_settings, judge_settings):
        provider = str(settings.get("provider", "")).strip().lower()
        model = str(settings.get("model", "")).strip()
        api_key = settings.get("api_key", "")
        if provider not in supported_providers:
            raise ValueError("Choose a supported answer and judge provider.")
        if not model or len(model) > 200:
            raise ValueError("Enter a valid model ID (1–200 characters).")
        if provider != "ollama" and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError(f"Enter an API key for the {provider.title()} provider.")
        clean_settings.append({
            "provider": provider,
            "model": model,
            "api_key": api_key.strip() if isinstance(api_key, str) else "",
        })
    if not isinstance(retrieval_settings, dict):
        raise ValueError("Retrieval settings are invalid.")
    method = str(retrieval_settings.get("method", "")).strip().lower()
    if method not in {"dense", "bm25", "hybrid"}:
        raise ValueError("Choose Dense Vector Search, BM25, or Hybrid Search.")
    embedding = normalize_embedding_settings(
        retrieval_settings.get("embedding_provider", ""),
        retrieval_settings.get("embedding_model", ""),
        retrieval_settings.get("embedding_api_key", ""),
        require_key=method in {"dense", "hybrid"},
    )
    try:
        top_k = int(retrieval_settings.get("top_k", 4))
        candidate_k = int(retrieval_settings.get("candidate_k", 32))
    except (TypeError, ValueError) as exc:
        raise ValueError("Retrieval result limits must be whole numbers.") from exc
    if not 1 <= top_k <= 100 or not top_k <= candidate_k <= 500:
        raise ValueError("Retrieval result limits are outside the supported range.")
    try:
        max_input_tokens = int(retrieval_settings.get("max_input_tokens") or 0) or None
        max_output_tokens = int(retrieval_settings.get("max_output_tokens") or 0) or None
    except (TypeError, ValueError) as exc:
        raise ValueError("Answer length limits must be whole numbers.") from exc
    if max_input_tokens is not None and not 500 <= max_input_tokens <= 200000:
        raise ValueError("The context budget is outside the supported range.")
    if max_output_tokens is not None and not 50 <= max_output_tokens <= 32000:
        raise ValueError("The answer length limit is outside the supported range.")
    reranker_model = str(
        retrieval_settings.get("reranker_model", RERANKER_MODEL) or RERANKER_MODEL
    ).strip()
    if not reranker_model or len(reranker_model) > 200:
        raise ValueError("Enter a valid reranker model ID (1–200 characters).")
    retrieval_settings = {
        "method": method,
        "embedding_provider": embedding["provider"],
        "embedding_model": embedding["model"],
        "embedding_api_key": embedding["api_key"],
        "use_reranker": bool(retrieval_settings.get("use_reranker", False)),
        "reranker_model": reranker_model,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
    }
    answer_settings, judge_settings = clean_settings
    with _workspace_lock:
        if workspace_id in _active_workspaces:
            raise ValueError("You already have an evaluation running or queued. Wait for it to finish.")
        if not _queue_slots.acquire(blocking=False):
            return False
        _active_workspaces.add(workspace_id)
        _live_jobs.add(job_id)

    now = _timestamp()
    try:
        _write_job(job_id, {
            "job_id": job_id,
            "workspace_id": workspace_id,
            "status": "queued",
            "phase": "Waiting for a free evaluation worker",
            "created_at": now,
            "updated_at": now,
            "options": {key: value for key, value in options.items()},
            "sample_count": len(golden_data),
            "progress": 0,
            "requests_completed": 0,
            "expected_requests": _expected_request_count(golden_data, options, top_k),
            "active_requests": 0,
            "failed_requests": 0,
        })
        _executor.submit(
            _execute_job,
            job_id,
            workspace_id,
            golden_data,
            options,
            answer_settings,
            judge_settings,
            retrieval_settings,
        )
    except Exception:
        _queue_slots.release()
        _release_workspace(workspace_id)
        with _workspace_lock:
            _live_jobs.discard(job_id)
        try:
            _job_path(job_id).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return True
