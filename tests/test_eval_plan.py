"""Estimator, progress accounting, quota-aware retries and failure isolation for evaluations."""
import json
from types import SimpleNamespace

import pytest

from core import rate_limit
from core.eval_plan import ProgressTracker, component_judge_calls, estimate_run, quality_judge_calls, safety_judge_calls


# ---------------------------------------------------------------- estimator
def _fake_value(schema, defs):
    if "$ref" in schema:
        return _fake_value(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            if option.get("type") != "null":
                return _fake_value(option, defs)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        return {k: _fake_value(v, defs) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return [_fake_value(schema.get("items", {}), defs)]
    if kind == "boolean":
        return True
    if kind in ("integer", "number"):
        return 1
    return "yes"


def _counting_judge():
    from deepeval.models.base_model import DeepEvalBaseLLM

    class Counting(DeepEvalBaseLLM):
        def __init__(self):
            self.n = 0
            self.model_name = "stub"

        def load_model(self):
            return self

        def get_model_name(self):
            return "stub"

        def generate(self, prompt, schema=None, *a, **k):
            self.n += 1
            if schema is None:
                return json.dumps({"verdict": "yes"})
            js = schema.model_json_schema()
            data = _fake_value(js, js.get("$defs", {}))
            try:
                return schema(**data)
            except Exception:
                return json.dumps(data)

        async def a_generate(self, prompt, schema=None, *a, **k):
            return self.generate(prompt, schema, *a, **k)

    return Counting()


@pytest.mark.parametrize("reasons", [True, False])
def test_plan_constants_match_the_installed_deepeval(reasons):
    """If a DeepEval upgrade changes calls per metric, this fails instead of the estimate drifting."""
    from deepeval.test_case import LLMTestCase

    from eval import application_eval, component_eval
    from eval.common import evaluate_test_cases

    top_k = 4
    case = LLMTestCase(
        input="What is X?",
        actual_output="X is a thing. It has two parts.",
        expected_output="X is a thing with two parts.",
        retrieval_context=["X is a thing.", "It has two parts.", "Unrelated.", "More."][:top_k],
        context=["X is a thing."],
    )

    def total_calls(builder):
        total = 0
        for metric in builder(_counting_judge(), reasons) if builder is not application_eval.build_quality_metrics else builder(_counting_judge()):
            judge = _counting_judge()
            metric.model = judge
            evaluate_test_cases([case], [metric])
            total += judge.n
        return total

    assert total_calls(component_eval.build_component_metrics) == component_judge_calls(top_k, reasons)
    assert total_calls(application_eval.build_safety_content_metrics) == safety_judge_calls(reasons)
    assert total_calls(application_eval.build_quality_metrics) == quality_judge_calls()


def test_estimate_run_counts_all_parts():
    plan = estimate_run(10, {"run_component": True, "run_quality": True, "run_safety": True}, top_k=4, scope_probes=5)
    per_question = component_judge_calls(4, True) + quality_judge_calls() + safety_judge_calls(True)
    assert plan["per_question_judge_calls"] == per_question
    assert plan["maximum"] == 10 + 10 * per_question + 5 + 5
    assert plan["minimum"] == plan["maximum"] - 5


def test_fast_mode_removes_the_reason_calls():
    base = {"run_component": True, "run_quality": False, "run_safety": True}
    with_reasons = estimate_run(10, {**base, "include_reasons": True}, top_k=4)["maximum"]
    fast = estimate_run(10, {**base, "include_reasons": False}, top_k=4)["maximum"]
    assert fast < with_reasons
    assert estimate_run(10, {"run_component": False})["maximum"] == 1


# ---------------------------------------------------------------- progress
def test_failed_attempts_never_advance_or_pin_the_bar():
    tracker = ProgressTracker(100)
    for _ in range(5):
        tracker.record("start")
        tracker.record("complete", duration=1.0)
    before = tracker.record("waiting")["progress"]
    for _ in range(300):  # a long rate-limit storm
        tracker.record("start")
        state = tracker.record("failed", duration=0.1)
    assert state["progress"] == before
    assert state["requests_completed"] == 5
    assert state["failed_requests"] == 300
    assert state["expected_requests"] == 100  # the denominator never chases the numerator


def test_progress_is_capped_and_flags_an_overrun():
    tracker = ProgressTracker(3)
    for _ in range(5):
        tracker.record("start")
        state = tracker.record("complete", duration=1.0)
    assert state["progress"] == 95
    assert state["estimate_exceeded"] is True
    assert "taking longer than estimated" in state["phase"]


def test_cache_hits_count_as_completed_requests():
    tracker = ProgressTracker(4)
    state = tracker.record("cache_hit")
    assert state["requests_completed"] == 1 and state["cache_hits"] == 1 and state["model_api_calls"] == 0


# ---------------------------------------------------------------- quota-aware retries
def _plan(message, attempt=0, provider="groq", maximum=5):
    return rate_limit.plan_rate_limit_retry(Exception(message), attempt, maximum, provider=provider)


def test_per_minute_limit_waits_then_gives_up_without_a_quota_error():
    assert 7 < _plan("Error code: 429 - rate limit (TPM). Please try again in 7.5s") < 12
    assert _plan("Error code: 429 - rate limit (TPM). Please try again in 7.5s", attempt=4) is None


def test_long_or_daily_or_billing_limits_stop_instead_of_waiting():
    with pytest.raises(rate_limit.QuotaExhaustedError):
        _plan("429 Rate limit reached on tokens per day (TPD). Please try again in 19h2m3s")
    with pytest.raises(rate_limit.QuotaExhaustedError):
        _plan("429 Rate limit reached on tokens per day (TPD). Please try again in 6m43s")
    with pytest.raises(rate_limit.QuotaExhaustedError):
        _plan("429 insufficient_quota: You exceeded your current quota", provider="openai")
    with pytest.raises(rate_limit.QuotaExhaustedError):
        _plan("429 requests per day exceeded, try again in 20s", attempt=4)


def test_retry_after_parser_understands_hours_minutes_seconds_and_ms():
    wait = rate_limit.get_retry_wait
    assert wait(Exception("try again in 6m43.2s")) == pytest.approx(404.2)
    assert wait(Exception("try again in 19h2m3s")) == pytest.approx(68524.0)
    assert wait(Exception("retry in 350ms")) == pytest.approx(1.35)


def test_adaptive_pacing_slows_down_after_a_429_and_resets_per_run():
    rate_limit.set_pacing_profile("paid")
    assert rate_limit.get_min_interval_seconds("openai") == 0.0
    rate_limit.note_rate_limited("openai")
    assert rate_limit.get_min_interval_seconds("openai") >= 2.0
    rate_limit.set_pacing_profile("free")
    assert rate_limit.get_min_interval_seconds("groq") >= 2.5
    assert rate_limit.get_min_interval_seconds("openai") == 0.0
    rate_limit.set_pacing_profile("auto")
    assert rate_limit.get_min_interval_seconds("groq") == rate_limit.PROVIDER_MIN_INTERVAL_SECONDS["groq"]


def test_bad_pacing_env_value_does_not_crash_import(monkeypatch):
    monkeypatch.setenv("GROQ_MIN_INTERVAL_SECONDS", "fast")
    assert rate_limit._env_float("GROQ_MIN_INTERVAL_SECONDS", 0.0) == 0.0


# ---------------------------------------------------------------- judge circuit breaker
def test_judge_stops_calling_the_provider_after_a_quota_error(monkeypatch):
    from core import judge as judge_module

    calls = {"n": 0}

    class DeadQuota:
        temperature = 0

        def invoke(self, prompt, config=None):
            calls["n"] += 1
            raise Exception("Error code: 429 - Rate limit reached on tokens per day (TPD). Please try again in 19h2m3s")

    monkeypatch.setattr(judge_module, "build_chat_model", lambda *a, **k: DeadQuota())
    monkeypatch.setattr(judge_module, "wait_for_rate_limit", lambda **k: None)
    judge = judge_module.LLMJudge("m", provider="groq", api_key="k")
    with pytest.raises(rate_limit.QuotaExhaustedError):
        judge.generate("p1")
    with pytest.raises(rate_limit.QuotaExhaustedError):
        judge.generate("p2")
    assert calls["n"] == 1  # the second prompt never reached the provider
    assert judge.quota_error is not None


# ---------------------------------------------------------------- failure isolation
def test_one_failing_metric_does_not_abort_the_run_and_is_reported():
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase

    from eval.common import evaluate_test_cases, summarize_results

    class Broken(DeepEvalBaseLLM):
        def __init__(self):
            self.model_name = "broken"

        def load_model(self):
            return self

        def generate(self, prompt, *a, **k):
            raise RuntimeError("provider exploded")

        async def a_generate(self, prompt, *a, **k):
            raise RuntimeError("provider exploded")

        def get_model_name(self):
            return "broken"

    case = LLMTestCase(input="q", actual_output="a", expected_output="a", retrieval_context=["ctx"])
    result = evaluate_test_cases([case, case], [AnswerRelevancyMetric(model=Broken(), include_reason=False)])
    summary = summarize_results(result)
    assert summary["Answer Relevancy"]["errored"] == 2
    assert summary["Answer Relevancy"]["total"] == 0


def test_run_selected_evaluations_raises_the_quota_error_when_nothing_was_scored(monkeypatch):
    from eval import suite

    quota = rate_limit.QuotaExhaustedError("Groq quota exhausted.")
    judge = SimpleNamespace(quota_error=quota)
    monkeypatch.setattr(suite, "build_test_cases", lambda *a, **k: [])
    monkeypatch.setattr(suite, "build_component_metrics", lambda *a, **k: ["m"])
    monkeypatch.setattr(suite, "evaluate_test_cases", lambda *a, **k: SimpleNamespace(test_results=[]))
    with pytest.raises(rate_limit.QuotaExhaustedError):
        suite.run_selected_evaluations(
            [], None, None, judge, run_component=True, run_quality=False, run_safety=False
        )


def test_job_options_reject_unknown_pacing_profile():
    from core import evaluation_jobs

    golden = [{"input": "q", "expected_output": "a"}]
    with pytest.raises(ValueError):
        evaluation_jobs.submit_evaluation_job(
            job_id="a" * 32,
            workspace_id="b" * 32,
            golden_data=golden,
            options={"run_component": True, "pacing": "turbo"},
            answer_settings={"provider": "groq", "model": "m", "api_key": "k"},
            judge_settings={"provider": "groq", "model": "m", "api_key": "k"},
            retrieval_settings={"method": "bm25", "embedding_provider": "local", "embedding_model": "x", "top_k": 4, "candidate_k": 32},
        )


def test_golden_generation_stops_after_the_first_quota_error(monkeypatch):
    from core import golden

    class Collection:
        def count(self):
            return 40

        def get(self, limit, offset, include):
            n = min(limit, 40 - offset)
            return {"documents": ["text " * 30] * n, "metadatas": [{"source": "a.pdf"}] * n}

    calls = {"n": 0}

    class Llm:
        def invoke(self, prompt, config=None):
            calls["n"] += 1
            raise RuntimeError("Error code: 429 - Rate limit reached on tokens per day (TPD). Please try again in 19h2m3s")

    monkeypatch.setattr(golden, "_build_llm", lambda *a, **k: Llm())
    monkeypatch.setattr(golden, "close_chat_model", lambda m: None)
    monkeypatch.setattr(golden.time, "sleep", lambda s: None)
    _, stats = golden.generate_golden_dataset(Collection(), n=20, api_key="k", model_provider="groq")
    assert stats["produced"] == 0
    assert calls["n"] == 1
    assert "quota" in stats["aborted_reason"].lower()
    assert "authentication" not in stats["aborted_reason"].lower()
