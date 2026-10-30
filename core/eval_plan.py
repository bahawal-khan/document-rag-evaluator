"""How many provider requests an evaluation run needs: one source of truth.

The per-metric judge-call counts below were measured by running every DeepEval
metric this app uses against a counting stub judge (tests/test_eval_plan.py
re-checks them against the installed DeepEval, so a library upgrade that changes
them fails a test instead of silently making the estimate wrong).

* Contextual Precision / Recall: 1 verdict call (+1 reason call)
* Contextual Relevancy:          1 verdict call per retrieved chunk (+1 reason)
* Answer Relevancy:              statements + verdicts (+1 reason)
* Faithfulness:                  truths + claims + verdicts (+1 reason)
* Correctness / Completeness:    2 calls each (GEval generates steps, then scores)
* Toxicity / PII leakage:        2 calls each (+1 reason)
* Scope probe:                   1 answer call + 1 GEval call
"""

from __future__ import annotations

DEFAULT_SCOPE_PROBES = 5


def component_judge_calls(top_k: int, include_reasons: bool) -> int:
    reason = 1 if include_reasons else 0
    return (1 + reason) + (1 + reason) + (max(1, int(top_k)) + reason) + (2 + reason) + (3 + reason)


def quality_judge_calls() -> int:
    return 4


def safety_judge_calls(include_reasons: bool) -> int:
    return 2 * (2 + (1 if include_reasons else 0))


def estimate_run(
    question_count: int,
    options: dict,
    *,
    top_k: int = 4,
    scope_probes: int = DEFAULT_SCOPE_PROBES,
) -> dict:
    """Return the expected request counts for a run (before caching and retries)."""
    questions = max(0, int(question_count))
    reasons = bool(options.get("include_reasons", True))
    per_question_judge = 0
    if options.get("run_component"):
        per_question_judge += component_judge_calls(top_k, reasons)
    if options.get("run_quality"):
        per_question_judge += quality_judge_calls()
    if options.get("run_safety"):
        per_question_judge += safety_judge_calls(reasons)
    answers = questions if per_question_judge else 0
    metric_calls = questions * per_question_judge
    scope_judges = scope_probes if options.get("run_safety") else 0
    scope_answers = scope_probes if options.get("run_safety") else 0
    maximum = answers + metric_calls + scope_judges + scope_answers
    return {
        "answers": answers,
        "metric_calls": metric_calls,
        "scope_judges": scope_judges,
        "scope_answers": scope_answers,
        "per_question_judge_calls": per_question_judge,
        "minimum": max(1, maximum - scope_answers),
        "maximum": max(1, maximum),
    }


class ProgressTracker:
    """Request accounting for one run, kept separate from the job so it can be unit-tested.

    Only completed requests and cache hits advance the bar. A failed attempt (usually
    a rate-limit error that is about to be retried) is counted separately, so retries
    can no longer move the denominator and pin the bar near the end.
    """

    def __init__(self, expected_requests: int):
        self.expected = max(1, int(expected_requests))
        self.completed = 0
        self.active = 0
        self.api_calls = 0
        self.cache_hits = 0
        self.failed = 0
        self.seconds = 0.0
        self.timed = 0
        self.last_percent = 2

    def record(self, event: str, **details) -> dict:
        phase = details.get("label")
        if event == "waiting":
            phase = phase or "Waiting for provider pacing"
        elif event == "cache_hit":
            self.cache_hits += 1
            self.completed += 1
            phase = phase or "Reused a cached model response"
        elif event == "start":
            self.active += 1
            self.api_calls += 1
            phase = phase or "Calling a model"
        else:
            self.active = max(0, self.active - 1)
            if event == "failed":
                self.failed += 1
            else:
                self.completed += 1
                try:
                    duration = max(0.0, float(details.get("duration", 0.0)))
                except (TypeError, ValueError):
                    duration = 0.0
                if duration:
                    self.seconds += duration
                    self.timed += 1
            phase = phase or "Processing evaluation results"
        exceeded = self.completed >= self.expected
        percent = min(95, max(2, int(2 + 93 * self.completed / self.expected)))
        self.last_percent = max(self.last_percent, percent)
        if exceeded:
            phase = f"{phase} (taking longer than estimated)"
        return {
            "phase": phase,
            "progress": self.last_percent,
            "requests_completed": self.completed,
            "expected_requests": self.expected,
            "estimate_exceeded": exceeded,
            "active_requests": self.active,
            "model_api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "failed_requests": self.failed,
            "average_request_seconds": self.seconds / self.timed if self.timed else None,
        }
