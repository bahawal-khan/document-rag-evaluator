"""
Shared helpers used by the component/pipeline/application eval scripts:
building LLMTestCases from the golden dataset, and summarizing DeepEval's
result object into a simple dict (for both Streamlit display and the PDF
report).
"""

import logging
import os
import threading
import time
from pathlib import Path

from core.config import EVALUATION_TASK_TIMEOUT_SECONDS
from core.cache import get_cached_request, set_cached_request

os.environ.setdefault(
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
    str(EVALUATION_TASK_TIMEOUT_SECONDS),
)

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig

from core.rate_limit import (
    RATE_LIMIT_MAX_RETRIES,
    QuotaExhaustedError,
    RateLimiter,
    is_rate_limit_error,
    plan_rate_limit_retry,
)
from core.generator import format_llm_error
from core.provider import provider_base_url

logger = logging.getLogger(__name__)


MAX_GENERATION_RETRIES = 3  # attempts when the answer model returns an empty response


# Evaluation needs a direct, NON-streaming answer (streaming was only needed
# for the live-chat UX) — so we use .invoke(), not .stream()
def generate_answer_sync(
    llm,
    prompt: str,
    provider: str = "groq",
    progress_callback=None,
    *,
    model_name: str = "",
    cache_file: str | Path | None = None,
    request_parameters: dict | None = None,
) -> str:
    cached = get_cached_request(
        provider,
        model_name,
        prompt,
        parameters=request_parameters or {"role": "evaluation-answer"},
        cache_file=cache_file,
    ) if cache_file else None
    if cached and isinstance(cached.get("answer"), str):
        if progress_callback:
            progress_callback("cache_hit", label="Reused cached evaluation answer")
        return cached["answer"]

    rate_attempts = 0
    empty_attempts = 0
    while True:
        started = None
        try:
            if progress_callback:
                progress_callback("waiting", label=f"Waiting for {provider} answer model")
            with RateLimiter(provider=provider):
                if progress_callback:
                    progress_callback("start", label=f"Generating an answer with {provider}")
                started = time.monotonic()
                response = llm.invoke(prompt).content.strip()
                if not response:
                    raise ValueError("The answer model returned an empty response")
            if progress_callback:
                progress_callback(
                    "complete",
                    duration=time.monotonic() - started,
                    label="Answer generated; scoring selected metrics",
                )
            if cache_file and response:
                set_cached_request(
                    provider,
                    model_name,
                    prompt,
                    {"answer": response},
                    parameters=request_parameters or {"role": "evaluation-answer"},
                    cache_file=cache_file,
                )
            return response
        except Exception as exc:
            if progress_callback and started is not None:
                progress_callback(
                    "failed",
                    duration=time.monotonic() - started,
                    label="Answer request failed; checking retry",
                )
            if isinstance(exc, QuotaExhaustedError):
                raise
            if isinstance(exc, ValueError) and "empty response" in str(exc):
                empty_attempts += 1
                if empty_attempts < MAX_GENERATION_RETRIES:
                    time.sleep(1)
                    continue
            if not is_rate_limit_error(exc):
                raise RuntimeError(
                    format_llm_error(
                        exc,
                        provider,
                        model_name=model_name,
                        base_url=provider_base_url(provider),
                    )
                ) from exc
            wait_seconds = plan_rate_limit_retry(
                exc, rate_attempts, RATE_LIMIT_MAX_RETRIES, provider=provider, base_wait_seconds=12
            )
            rate_attempts += 1
            if wait_seconds is None:
                raise RuntimeError(
                    f"{provider.title()} answer model was still rate limited after "
                    f"{RATE_LIMIT_MAX_RETRIES} attempts."
                ) from exc
            logger.warning(
                "Generator was rate limited; retrying in %.1f seconds (attempt %s/%s).",
                wait_seconds,
                rate_attempts,
                RATE_LIMIT_MAX_RETRIES,
            )
            if progress_callback:
                progress_callback(
                    "waiting",
                    label=f"{provider} rate limit; retrying in {wait_seconds:.0f}s",
                )
            time.sleep(wait_seconds)


def _retrieve_all(retriever, questions: list[str]) -> list[list[dict]]:
    """Retrieve for every question, batching embedding calls when supported.

    ``retrieve_many`` embeds up to 32 questions per provider request instead of
    one request per question, which matters for paid embedding APIs.
    """
    retrieve_many = getattr(retriever, "retrieve_many", None)
    method = getattr(retriever, "method", None)
    if callable(retrieve_many) and method:
        results: list[list[dict]] = []
        for start in range(0, len(questions), 32):
            batch = questions[start : start + 32]
            results.extend(retrieve_many(batch, methods=[method])[method])
        errors = getattr(retriever, "last_method_errors", {}) or {}
        # On any retrieval error fall back to retrieve(), which raises the
        # same safe, user-facing error the single-question path always did.
        if len(results) == len(questions) and not errors.get(method):
            return results
    return [retriever.retrieve(question) for question in questions]


def build_test_cases(
    golden_data,
    retriever,
    llm,
    prompt_builder,
    provider: str = "groq",
    progress_callback=None,
    *,
    model_name: str = "",
    cache_file: str | Path | None = None,
    request_parameters: dict | None = None,
):
    """
    prompt_builder: a function like core.generator.build_prompt_within_budget
    (question, chunks) -> (prompt, used_chunks, input_tokens)
    """
    test_cases = []
    all_chunks = _retrieve_all(retriever, [item["input"] for item in golden_data])
    for item, chunks in zip(golden_data, all_chunks):
        prompt, used_chunks, _ = prompt_builder(item["input"], chunks)
        actual_output = generate_answer_sync(
            llm,
            prompt,
            provider,
            progress_callback,
            model_name=model_name,
            cache_file=cache_file,
            request_parameters=request_parameters,
        )

        test_cases.append(LLMTestCase(
            input=item["input"],
            actual_output=actual_output,
            expected_output=item.get("expected_output"),
            retrieval_context=[c["text"] for c in used_chunks],
            metadata={"difficulty": item.get("difficulty", "medium")},
        ))
    return test_cases


def eval_async_config(throttle=0):
    # Concurrency is 1 anyway (provider quotas), so async buys nothing. Running
    # DeepEval synchronously means no asyncio event loop, no httpx.AsyncClient
    # and no Proactor transports, which removes the source of
    # ``ConnectionResetError: [WinError 10054]`` on Windows.
    return AsyncConfig(run_async=False, max_concurrent=1, throttle_value=throttle)


def eval_cache_config():
    return CacheConfig(write_cache=False)


def eval_error_config():
    # One failed judge call must not discard the rest of a long run: the errored
    # metric is recorded on its test case and excluded from the averages.
    return ErrorConfig(ignore_errors=True)


def eval_display_config():
    return DisplayConfig(show_indicator=False, print_results=False)


# DeepEval keeps ONE process-wide test-run manager (reset at the start of every
# evaluate() and persisted through a shared temp file). Two users scoring at the
# same moment would corrupt each other's results, so only this call is serialised;
# retrieval and answer generation for different users still run in parallel.
_deepeval_lock = threading.Lock()


def evaluate_test_cases(test_cases, metrics):
    with _deepeval_lock:
        return evaluate(
            test_cases,
            metrics=metrics,
            async_config=eval_async_config(),
            error_config=eval_error_config(),
            cache_config=eval_cache_config(),
            display_config=eval_display_config(),
        )


def summarize_results(evaluation_result, metric_names=None) -> dict:
    """EvaluationResult -> {metric_name: {average, pass_rate, passed, failed,
    total, failure_reasons}}. failure_reasons captures a few example
    judge explanations for failed cases, so the UI/report can show WHY a
    metric scored low, not just the number."""
    summary = {}
    for test_result in evaluation_result.test_results:
        if not test_result.metrics_data:
            continue
        for md in test_result.metrics_data:
            if metric_names is not None and md.name not in metric_names:
                continue
            bucket = summary.setdefault(
                md.name, {"scores": [], "passed": 0, "failed": 0, "errored": 0, "reasons": []}
            )
            if getattr(md, "error", None):
                bucket["errored"] += 1
                continue
            score = md.score if md.score is not None else 0
            bucket["scores"].append(score)
            if md.success:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
                if md.reason:
                    bucket["reasons"].append({"input": test_result.input, "reason": md.reason, "score": score})

    final = {}
    for name, bucket in summary.items():
        total = bucket["passed"] + bucket["failed"]
        final[name] = {
            "average": sum(bucket["scores"]) / len(bucket["scores"]) if bucket["scores"] else 0,
            "passed": bucket["passed"],
            "failed": bucket["failed"],
            "errored": bucket["errored"],
            "total": total,
            "pass_rate": bucket["passed"] / total if total else 0,
            "failure_reasons": bucket["reasons"][:5],  # 5 examples is enough for a report
        }
    return final
