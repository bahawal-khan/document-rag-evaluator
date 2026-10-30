import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading
import time
import pytest
from pathlib import Path
from types import SimpleNamespace

from core.config import MAX_INPUT_TOKENS
from core.cost import count_tokens
from core.generator import SYSTEM_PROMPT, build_prompt_within_budget, format_llm_error
from core.rate_limit import (
    PROVIDER_MIN_INTERVAL_SECONDS,
    RateLimiter,
    get_min_interval_seconds,
    get_retry_wait,
    is_rate_limit_error,
)


def test_retry_wait_uses_backoff_for_429s():
    wait = get_retry_wait("429 Too Many Requests. Please try again in 15s")
    assert wait >= 15


def test_retry_wait_honors_provider_retry_after_header():
    error = Exception("429 Too Many Requests")
    error.response = SimpleNamespace(headers={"retry-after": "2"})

    assert get_retry_wait(error) == 3


def test_retry_wait_handles_case_insensitive_retry_headers():
    error = Exception("429 Too Many Requests")
    error.response = SimpleNamespace(headers={"Retry-After": "2"})

    assert get_retry_wait(error) == 3


def test_retry_wait_uses_exponential_backoff_without_provider_hint():
    assert get_retry_wait("429 rate limit", base_wait_seconds=4, attempt=2) == 16


def test_rate_limit_detection_matches_common_provider_errors():
    assert is_rate_limit_error("Too Many Requests")
    assert is_rate_limit_error("quota exceeded")
    assert is_rate_limit_error("RESOURCE_EXHAUSTED")
    assert not is_rate_limit_error("invalid API key")


def test_rate_limiter_serializes_calls():
    limiter = RateLimiter(min_interval_seconds=0.05, max_parallel_requests=1)
    start = time.perf_counter()

    with limiter:
        time.sleep(0.05)
    with limiter:
        time.sleep(0.05)

    elapsed = time.perf_counter() - start
    assert elapsed >= 0.09


def test_rate_limiter_bounds_parallel_provider_calls():
    limiter = RateLimiter(min_interval_seconds=0, max_parallel_requests=2, provider="test")
    lock = threading.Lock()
    active = 0
    peak = 0

    def make_call(_index):
        nonlocal active, peak
        with limiter:
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(make_call, range(10)))

    assert peak == 2


def test_provider_pacing_is_configured_independently(monkeypatch):
    monkeypatch.setitem(PROVIDER_MIN_INTERVAL_SECONDS, "groq", 2.5)
    monkeypatch.setitem(PROVIDER_MIN_INTERVAL_SECONDS, "openai", 0.0)

    assert get_min_interval_seconds("groq") == 2.5
    assert get_min_interval_seconds("openai") == 0.0


def test_async_rate_limiter_shares_sync_pacing_state():
    limiter = RateLimiter(min_interval_seconds=0.2)
    with limiter:
        pass

    start = time.perf_counter()
    asyncio.run(limiter.async_wait())

    assert time.perf_counter() - start >= 0.15


def test_eval_config_does_not_add_deepeval_scheduling_delay():
    from eval.common import eval_async_config

    config = eval_async_config()
    assert config.max_concurrent == 1
    assert config.throttle_value == 0


def test_evaluation_answer_generation_retries_rate_limits(monkeypatch):
    from eval import common

    class FlakyLLM:
        calls = 0

        def invoke(self, prompt):
            assert prompt == "prompt"
            self.calls += 1
            if self.calls == 1:
                raise Exception("429 Too Many Requests")
            return SimpleNamespace(content=" answer ")

    llm = FlakyLLM()
    monkeypatch.setattr(common, "RateLimiter", lambda **kwargs: nullcontext())

    def no_sleep(seconds):
        assert seconds > 0

    monkeypatch.setattr(common.time, "sleep", no_sleep)

    assert common.generate_answer_sync(llm, "prompt") == "answer"
    assert llm.calls == 2


def test_evaluation_answer_cache_prevents_duplicate_provider_calls(monkeypatch, tmp_path):
    from eval import common

    class CountingLLM:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(content="cached answer")

    monkeypatch.setattr(common, "RateLimiter", lambda **kwargs: nullcontext())
    llm = CountingLLM()
    cache_file = tmp_path / "cache.json"

    assert common.generate_answer_sync(
        llm, "same prompt", "groq", model_name="model-a", cache_file=cache_file
    ) == "cached answer"
    assert common.generate_answer_sync(
        llm, "same prompt", "groq", model_name="model-a", cache_file=cache_file
    ) == "cached answer"
    assert llm.calls == 1


def test_failed_or_empty_evaluation_response_is_not_cached(monkeypatch, tmp_path):
    from eval import common

    class EmptyThenValidLLM:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(content="" if self.calls == 1 else "valid")

    monkeypatch.setattr(common, "RateLimiter", lambda **kwargs: nullcontext())
    monkeypatch.setattr(common, "MAX_GENERATION_RETRIES", 1)
    llm = EmptyThenValidLLM()
    cache_file = tmp_path / "cache.json"

    with pytest.raises(RuntimeError):
        common.generate_answer_sync(
            llm, "prompt", "groq", model_name="model-a", cache_file=cache_file
        )
    assert not cache_file.exists()


def test_generator_prompt_requires_semantic_grounding_and_paraphrase():
    assert "paraphrase" in SYSTEM_PROMPT.lower()
    assert "use only the supplied" in SYSTEM_PROMPT.lower()
    assert "source material" in SYSTEM_PROMPT.lower()
    assert "do not penalize" in SYSTEM_PROMPT.lower()
    assert "untrusted input" in SYSTEM_PROMPT.lower()


def test_prompt_builder_keeps_estimated_input_within_budget():
    prompt, used_chunks, input_tokens = build_prompt_within_budget(
        "What does the paper conclude?",
        [{
            "text": "A grounded finding. " * 500,
            "source": "paper.pdf",
            "location": "Page 1",
            "distance": 0.1,
        }],
    )

    assert used_chunks
    assert input_tokens == count_tokens(prompt)
    assert input_tokens <= MAX_INPUT_TOKENS


def test_prompt_builder_rejects_a_question_that_exceeds_budget():
    oversized_question = "q" * ((MAX_INPUT_TOKENS + 100) * 4)

    try:
        build_prompt_within_budget(oversized_question, [])
    except ValueError as exc:
        assert "too long" in str(exc).lower()
    else:
        raise AssertionError("An oversized question should be rejected.")


def test_character_token_estimate_rounds_up():
    assert count_tokens("12345") == 2


def test_generator_supports_eval_timeout_without_changing_live_defaults(monkeypatch):
    from core import generator

    monkeypatch.setattr(
        generator,
        "build_chat_model",
        lambda model_name, **kwargs: {"model_name": model_name, **kwargs},
    )

    eval_model = generator.build_generator(streaming=False, timeout=90)
    live_model = generator.build_generator()

    assert eval_model["streaming"] is False
    assert eval_model["timeout"] == 90
    assert live_model["streaming"] is True
    assert live_model["timeout"] == generator.HARD_REQUEST_DEADLINE_SECONDS + 2


def test_streaming_empty_model_response_is_marked_as_an_error(monkeypatch):
    from core import generator

    class EmptyLLM:
        def stream(self, prompt, config=None):
            return iter(())

    monkeypatch.setattr(generator, "RateLimiter", lambda **kwargs: nullcontext())
    monkeypatch.setattr(generator, "get_langsmith_callbacks", lambda: [])
    metrics = {}

    output = "".join(generator.stream_answer(EmptyLLM(), "prompt", metrics))

    assert output == ""
    assert "empty response" in metrics["error"].lower()


def test_eval_criteria_are_semantic_not_exact_string_match():
    app_eval_text = Path("eval/application_eval.py").read_text(encoding="utf-8")
    assert "paraphrase" in app_eval_text.lower()
    assert "semantically" in app_eval_text.lower()
    assert "exact wording" not in app_eval_text.lower()


def test_format_llm_error_handles_connection_failures():
    message = format_llm_error(Exception("All connection attempts failed"))
    assert "could not reach the groq api" in message.lower()
    assert "internet connection" in message.lower()

    generic_connection_error = format_llm_error(ConnectionError("Connection error while contacting model"))
    assert "could not reach the groq api" in generic_connection_error.lower()
    assert "internet connection" in generic_connection_error.lower()
