"""Orchestrate the selected evaluation metrics over one generated answer set."""

from functools import partial

from core.generator import build_prompt_within_budget
from eval.application_eval import build_quality_metrics, build_safety_content_metrics
from eval.common import build_test_cases, evaluate_test_cases, summarize_results
from eval.component_eval import (
    GENERATOR_METRICS,
    RETRIEVER_METRICS,
    build_component_metrics,
)
from eval.pipeline_eval import summarize_pipeline


def run_selected_evaluations(
    golden_data,
    retriever,
    llm,
    judge,
    *,
    run_component: bool,
    run_quality: bool,
    run_safety: bool,
    answer_provider: str | None = None,
    answer_model: str | None = None,
    cache_file=None,
    progress_callback=None,
    include_reasons: bool = True,
    max_input_tokens: int | None = None,
):
    metrics = []
    if run_component:
        metrics.extend(build_component_metrics(judge, include_reasons))
    if run_quality:
        metrics.extend(build_quality_metrics(judge))
    if run_safety:
        metrics.extend(build_safety_content_metrics(judge, include_reasons))

    if not metrics:
        return {}

    prompt_builder = (
        partial(build_prompt_within_budget, max_input_tokens=max_input_tokens)
        if max_input_tokens
        else build_prompt_within_budget
    )
    if answer_provider:
        test_cases = build_test_cases(
            golden_data,
            retriever,
            llm,
            prompt_builder,
            provider=answer_provider,
            model_name=answer_model or "",
            cache_file=cache_file,
            request_parameters={"role": "evaluation-answer", "temperature": 0.1},
            progress_callback=progress_callback,
        )
    else:
        test_cases = build_test_cases(
            golden_data,
            retriever,
            llm,
            prompt_builder,
            **(
                {
                    "progress_callback": progress_callback,
                    "model_name": answer_model or "",
                    "cache_file": cache_file,
                    "request_parameters": {"role": "evaluation-answer", "temperature": 0.1},
                }
                if answer_model or cache_file or progress_callback
                else {}
            ),
        )
    result = evaluate_test_cases(test_cases, metrics)
    summary = summarize_results(result)
    outputs = {"raw_result": result}

    quota_error = getattr(judge, "quota_error", None)
    scored = sum(item["total"] for item in summary.values())
    if quota_error is not None and not scored:
        # Nothing could be scored, so there is nothing useful to show.
        raise quota_error
    notes = []
    errored = sum(item.get("errored", 0) for item in summary.values())
    skipped = f" {errored} metric result(s) were skipped and are NOT included in the scores." if errored else ""
    if quota_error is not None:
        notes.append(
            "The run stopped early because the judge model's provider ran out of quota. "
            f"Reason: {quota_error}{skipped} "
            "What to do: choose another judge model or provider, or wait until the limit resets, "
            "then run again. Answers already generated are cached and will not be paid for twice."
        )
    elif errored:
        notes.append(
            f"{errored} metric result(s) failed and are NOT included in the scores. "
            "Common reasons: a temporary rate limit, a judge answer that was not valid JSON, "
            "or a network timeout. Metrics showing 'no data' were not scored. Run again, "
            "or choose a more reliable judge model."
        )
    if notes:
        outputs["run_notes"] = notes

    if run_component:
        outputs["retriever_summary"] = {
            name: data for name, data in summary.items() if name in RETRIEVER_METRICS
        }
        outputs["generator_summary"] = {
            name: data for name, data in summary.items() if name in GENERATOR_METRICS
        }
        outputs["pipeline_summary"] = summarize_pipeline(
            result,
            metric_names=RETRIEVER_METRICS + GENERATOR_METRICS,
        )

    if run_quality:
        outputs["quality_summary"] = {
            name: summary[name] for name in ("Correctness", "Completeness") if name in summary
        }

    if run_safety:
        outputs["safety_content_summary"] = {
            name: summary[name] for name in ("Toxicity", "PII Leakage") if name in summary
        }

    return outputs
