"""
PIPELINE-LEVEL evaluation (RAG Triad, extended: Precision + Recall +
Relevancy + Answer Relevancy + Faithfulness, all together).

Deliberately REUSES component_eval.py's single evaluate() call result — no
extra LLM calls here. The difference is purely the ANGLE:
  component-level -> "how good is each metric individually?" (diagnostic)
  pipeline-level   -> "does each QUESTION pass all its metrics together?"
                       (the real end-to-end user-experience view)
"""

from eval.common import summarize_results


def summarize_pipeline(raw_evaluation_result, metric_names=None) -> dict:
    metric_summary = summarize_results(raw_evaluation_result, metric_names=metric_names)

    test_results = raw_evaluation_result.test_results
    total = len(test_results)
    fully_passed = sum(
        1 for test_result in test_results if _test_result_passed(test_result, metric_names)
    )

    return {
        "metric_summary": metric_summary,          # same 5-metric table, as a combined view
        "total_questions": total,
        "fully_passed": fully_passed,
        "fully_passed_rate": fully_passed / total if total else 0,
        "difficulty_breakdown": _difficulty_pass_rates(test_results, metric_names),
        # Partial credit: the strict "ALL metrics" figure above is all-or-nothing, so one
        # weak metric (often Contextual Relevancy) zeroes every question. These figures show
        # how close each question actually came.
        "partial": _partial_credit(test_results, metric_names),
    }


PARTIAL_PASS_SHARE = 0.75  # a question "mostly passes" when >= 75% of its scored metrics pass


def _scored_metrics(test_result, metric_names=None):
    return [
        metric
        for metric in (test_result.metrics_data or [])
        if (metric_names is None or metric.name in metric_names) and not getattr(metric, "error", None)
    ]


def _partial_credit(test_results, metric_names=None) -> dict:
    mostly = 0
    scored_questions = 0
    incomplete = 0
    passed_sum = 0
    scored_sum = 0
    by_level = {level: [0, 0] for level in ("easy", "medium", "hard")}  # [mostly, total]
    for test_result in test_results:
        all_metrics = [
            metric
            for metric in (test_result.metrics_data or [])
            if metric_names is None or metric.name in metric_names
        ]
        scored = _scored_metrics(test_result, metric_names)
        if len(scored) < len(all_metrics):
            incomplete += 1
        difficulty = (test_result.metadata or {}).get("difficulty", "medium")
        if not isinstance(difficulty, str) or difficulty not in by_level:
            difficulty = "medium"
        by_level[difficulty][1] += 1
        if not scored:
            continue
        scored_questions += 1
        passed = sum(1 for metric in scored if metric.success)
        passed_sum += passed
        scored_sum += len(scored)
        if passed / len(scored) >= PARTIAL_PASS_SHARE:
            mostly += 1
            by_level[difficulty][0] += 1
    return {
        "scored_questions": scored_questions,
        "incomplete_questions": incomplete,
        "mostly_passed": mostly,
        "mostly_passed_rate": mostly / scored_questions if scored_questions else 0,
        "avg_metrics_passed": passed_sum / scored_questions if scored_questions else 0,
        "avg_metrics_scored": scored_sum / scored_questions if scored_questions else 0,
        "by_difficulty": {
            level: {"mostly_passed": m, "total": t} for level, (m, t) in by_level.items()
        },
    }


def _test_result_passed(test_result, metric_names=None):
    if metric_names is None:
        return bool(test_result.success)

    matching_metrics = [
        metric
        for metric in (test_result.metrics_data or [])
        if metric.name in metric_names
    ]
    return bool(matching_metrics) and all(metric.success for metric in matching_metrics)


def _difficulty_pass_rates(test_results, metric_names=None):
    """Segments 'fully passed all metrics' by difficulty level (easy/medium/
    hard), using the metadata attached to each test case in build_test_cases.
    This answers: is the RAG system weaker on harder questions?"""
    buckets = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}  # [passed, total]
    for t in test_results:
        difficulty = (t.metadata or {}).get("difficulty", "medium")
        if not isinstance(difficulty, str) or difficulty not in buckets:
            difficulty = "medium"
        buckets[difficulty][1] += 1
        if _test_result_passed(t, metric_names):
            buckets[difficulty][0] += 1

    return {
        level: {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total else None,
        }
        for level, (passed, total) in buckets.items()
    }
