"""Provider-neutral LLM judge with rate-limit retries and JSON response repair."""

import re
import json as _json
import time
import asyncio
import threading
import weakref
import logging
from pathlib import Path

from core.config import EVALUATION_REQUEST_TIMEOUT_SECONDS, JUDGE_MODEL
from core.provider import build_chat_model
from core.provider import provider_base_url
from core.generator import format_llm_error
from core.cache import get_cached_request, set_cached_request
from core.rate_limit import (
    RATE_LIMIT_MAX_RETRIES,
    QuotaExhaustedError,
    async_wait_for_rate_limit,
    is_rate_limit_error,
    plan_rate_limit_retry,
    wait_for_rate_limit,
)
from core.tracer import get_langsmith_callbacks
from deepeval.models.base_model import DeepEvalBaseLLM

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

MAX_RETRIES = 3  # attempts for malformed or empty judge output
BASE_WAIT_SECONDS = 12
logger = logging.getLogger(__name__)

_async_semaphores_lock = threading.Lock()
_async_semaphores = weakref.WeakKeyDictionary()
def _get_async_semaphore():
    loop = asyncio.get_running_loop()
    with _async_semaphores_lock:
        semaphore = _async_semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)
            _async_semaphores[loop] = semaphore
        return semaphore


def _extract_json(text: str) -> str:
    """Grabs just the substring from the first '{' to the last '}', so
    DeepEval's JSON parser doesn't choke on stray text around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _is_valid_json(text: str) -> bool:
    """The model occasionally emits malformed JSON (missing comma, etc.) —
    validating here lets us retry before DeepEval ever sees it."""
    try:
        _json.loads(re.sub(r",\s*([\]}])", r"\1", text))
        return True
    except Exception:
        return False


def _try_repair(text: str):
    if not _HAS_JSON_REPAIR:
        return None
    try:
        fixed = repair_json(text)
        if fixed and fixed.strip() not in ("", "{}", "[]") and _is_valid_json(fixed):
            return fixed
    except Exception:
        pass
    return None


class LLMJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = JUDGE_MODEL,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        progress_callback=None,
        cache_file: str | Path | None = None,
    ):
        self.model_name = model_name
        self.provider = provider
        self.progress_callback = progress_callback
        self.cache_file = cache_file
        # Set once the provider reports an exhausted quota. Every later judge call
        # then fails immediately instead of spending more requests on a dead quota.
        self.quota_error: QuotaExhaustedError | None = None
        # One request at a time per judge (= per evaluation job). It must not be
        # process-wide, or every concurrent user's judge calls would queue behind each other.
        self._sync_semaphore = threading.Semaphore(1)
        self._cache_parameters = {"role": "judge", "temperature": 0, "max_tokens": 4096}
        self.model = build_chat_model(
            model_name,
            provider=provider,
            api_key=api_key,
            temperature=0,
            max_retries=0,
            max_tokens=4096,
            timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS,
        )

    def load_model(self):
        return self.model

    def _report_progress(self, event: str, **details) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback(event, **details)
        except Exception:
            logger.debug("Could not report judge request progress.", exc_info=True)

    def _handle_failure(self, error: Exception, state: dict, chat_model) -> float:
        """Return seconds to wait before retrying, or raise when retrying cannot help."""
        if isinstance(error, QuotaExhaustedError):
            self.quota_error = error
            raise error
        provider_name = (self.provider or "groq").title()
        if is_rate_limit_error(error):
            try:
                wait = plan_rate_limit_retry(
                    error,
                    state["rate"],
                    RATE_LIMIT_MAX_RETRIES,
                    provider=self.provider,
                    base_wait_seconds=BASE_WAIT_SECONDS,
                )
            except QuotaExhaustedError as quota_error:
                self.quota_error = quota_error
                raise
            state["rate"] += 1
            if wait is None:
                raise RuntimeError(
                    f"{provider_name} judge was still rate limited after {RATE_LIMIT_MAX_RETRIES} attempts."
                ) from error
            logger.warning(
                "Judge was rate limited; retrying in %.1f seconds (attempt %s/%s).",
                wait,
                state["rate"],
                RATE_LIMIT_MAX_RETRIES,
            )
            self._report_progress("waiting", label=f"Judge rate limit; retrying in {wait:.0f}s")
            return wait
        text = str(error).lower()
        if "empty response" in text or "malformed json" in text:
            state["invalid"] += 1
            if state["invalid"] >= MAX_RETRIES:
                raise RuntimeError(
                    f"{provider_name} judge failed after {MAX_RETRIES} attempts: {error}"
                ) from error
            chat_model.temperature = min(0.6, state["invalid"] * 0.15)
            logger.warning(
                "Judge response was invalid; retrying (attempt %s/%s): %s", state["invalid"], MAX_RETRIES, error
            )
            self._report_progress("waiting", label="Invalid judge response; retrying in 2s")
            return 2.0
        raise RuntimeError(
            format_llm_error(
                error,
                self.provider or "groq",
                model_name=self.model_name,
                base_url=provider_base_url(self.provider or "groq"),
            )
        ) from error

    def _cached_answer(self, prompt: str):
        if not self.cache_file:
            return None
        cached = get_cached_request(
            self.provider or "groq",
            self.model_name,
            prompt,
            parameters=self._cache_parameters,
            cache_file=self.cache_file,
        )
        if cached and isinstance(cached.get("answer"), str):
            self._report_progress("cache_hit", label="Reused cached judge response")
            return cached["answer"]
        return None

    def _finish(self, content: str, prompt: str):
        """Validate/repair the model text. Returns the JSON string or raises ValueError."""
        result = _extract_json(content)
        if not result.strip():
            raise ValueError("The judge model returned an empty response")
        if not _is_valid_json(result):
            fixed = _try_repair(result)
            if fixed:
                return fixed
            raise ValueError("The judge model returned malformed JSON")
        if self.cache_file:
            set_cached_request(
                self.provider or "groq",
                self.model_name,
                prompt,
                {"answer": result},
                parameters=self._cache_parameters,
                cache_file=self.cache_file,
            )
        return result

    def generate(self, prompt: str) -> str:
        if self.quota_error is not None:
            raise self.quota_error
        chat_model = self.load_model()
        cached = self._cached_answer(prompt)
        if cached is not None:
            return cached
        state = {"rate": 0, "invalid": 0}
        while True:
            request_started = None
            try:
                self._report_progress("waiting", label="Waiting for judge model pacing")
                with self._sync_semaphore:
                    wait_for_rate_limit(provider=self.provider)
                    self._report_progress(
                        "start", label=f"Scoring a metric with {self.provider or 'judge'}"
                    )
                    request_started = time.monotonic()
                    callbacks = get_langsmith_callbacks()
                    try:
                        response = chat_model.invoke(
                            prompt, config={"callbacks": callbacks} if callbacks else None
                        )
                    except Exception:
                        self._report_progress(
                            "failed",
                            duration=time.monotonic() - request_started,
                            label="Judge request failed; checking retry",
                        )
                        request_started = None
                        raise
                    self._report_progress(
                        "complete",
                        duration=time.monotonic() - request_started,
                        label="Judge response received; scoring evaluation",
                    )
                    request_started = None
                return self._finish(response.content, prompt)
            except Exception as error:
                time.sleep(self._handle_failure(error, state, chat_model))

    async def a_generate(self, prompt: str) -> str:
        if self.quota_error is not None:
            raise self.quota_error
        chat_model = self.load_model()
        cached = self._cached_answer(prompt)
        if cached is not None:
            return cached
        semaphore = _get_async_semaphore()
        state = {"rate": 0, "invalid": 0}
        while True:
            request_started = None
            try:
                self._report_progress("waiting", label="Waiting for judge model pacing")
                async with semaphore:
                    await async_wait_for_rate_limit(provider=self.provider)
                    self._report_progress(
                        "start", label=f"Scoring a metric with {self.provider or 'judge'}"
                    )
                    request_started = time.monotonic()
                    callbacks = get_langsmith_callbacks()
                    try:
                        response = await chat_model.ainvoke(
                            prompt, config={"callbacks": callbacks} if callbacks else None
                        )
                    except Exception:
                        self._report_progress(
                            "failed",
                            duration=time.monotonic() - request_started,
                            label="Judge request failed; checking retry",
                        )
                        request_started = None
                        raise
                    self._report_progress(
                        "complete",
                        duration=time.monotonic() - request_started,
                        label="Judge response received; scoring evaluation",
                    )
                    request_started = None
                return self._finish(response.content, prompt)
            except Exception as error:
                await asyncio.sleep(self._handle_failure(error, state, chat_model))

    def get_model_name(self):
        provider_name = (self.provider or "groq").title()
        return f"{provider_name} ({self.model_name})"


GroqJudge = LLMJudge
