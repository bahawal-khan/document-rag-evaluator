import asyncio
import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
import threading
import time
import os
import weakref

# No arbitrary delay is added for paid providers. Operators on a free or
# throttled tier can configure per-provider pacing, for example:
# GROQ_MIN_INTERVAL_SECONDS=2.5
def _env_float(name: str, default: float) -> float:
    """Read a non-negative float from the environment; bad values fall back to the default."""
    try:
        return max(0.0, float(os.getenv(name, "") or default))
    except (TypeError, ValueError):
        return default


_LEGACY_DEFAULT = _env_float("LLM_MIN_INTERVAL_SECONDS", 0.0)
PROVIDER_MIN_INTERVAL_SECONDS = {
    provider: _env_float(f"{provider.upper()}_MIN_INTERVAL_SECONDS", _LEGACY_DEFAULT)
    for provider in ("groq", "openai", "gemini", "anthropic", "openrouter", "ollama")
}
LLM_MIN_INTERVAL_SECONDS = max(PROVIDER_MIN_INTERVAL_SECONDS.values(), default=0.0)

# Pacing profiles let a user say what kind of API key they have. "auto" starts
# unpaced and slows down by itself when a provider answers 429; "free" starts at
# a conservative per-provider spacing; "paid" starts unpaced. All three keep the
# adaptive back-off, so a wrong choice costs time, not a failed run.
PACING_PROFILES = {
    "auto": {},
    "free": {"groq": 2.5, "gemini": 6.0, "openrouter": 3.5, "google": 6.0},
    "paid": {},
}
PACING_PROFILE_LABELS = {
    "auto": "Auto (recommended): adapts to rate limits",
    "free": "Free tier: slower and safer",
    "paid": "Paid tier: fastest",
}
ADAPTIVE_MAX_INTERVAL_SECONDS = 12.0
RATE_LIMIT_MAX_WAIT_SECONDS = _env_float("RATE_LIMIT_MAX_WAIT_SECONDS", 180.0)
RATE_LIMIT_MAX_RETRIES = max(1, int(_env_float("RATE_LIMIT_MAX_RETRIES", 5)))
DEFAULT_RETRY_BASE_SECONDS = 12.0

# ---------------------------------------------------------------------------
# Per-scope state. A "scope" is one user's workspace (a Streamlit session), so
# when many people use the app at once their pacing, adaptive back-off and
# parallelism limits never affect each other. The scope travels in a
# ContextVar: set it once at the top of a Streamlit run, or with
# ``rate_limit_scope`` around a background job. Without a scope everything falls
# back to the single shared "" scope (CLI scripts, tests).
# ---------------------------------------------------------------------------
_SCOPE_IDLE_SECONDS = 3600.0
_SCOPE_PRUNE_EVERY_SECONDS = 300.0
_scope_var: contextvars.ContextVar[str] = contextvars.ContextVar("rate_limit_scope", default="")

_provider_states_lock = threading.Lock()
_provider_states: dict[tuple[str, str], tuple[threading.Lock, list[float]]] = {}
_provider_semaphores: dict[tuple[str, str, int], threading.BoundedSemaphore] = {}
_scope_last_used: dict[str, float] = {}
_last_prune = [0.0]
_async_semaphores = weakref.WeakKeyDictionary()
_async_semaphores_lock = threading.Lock()

_pacing_lock = threading.Lock()
_pacing_profiles: dict[str, str] = {}
_adaptive_intervals: dict[tuple[str, str], float] = {}


def current_scope() -> str:
    return _scope_var.get()


def bind_rate_limit_scope(scope: str | None) -> None:
    """Bind the calling thread (for example a Streamlit script run) to one user's scope."""
    _scope_var.set(str(scope or ""))


def _clear_scope_pacing(scope: str) -> None:
    with _pacing_lock:
        _pacing_profiles.pop(scope, None)
        for key in [key for key in _adaptive_intervals if key[0] == scope]:
            del _adaptive_intervals[key]


@contextmanager
def rate_limit_scope(scope: str | None, profile: str | None = None):
    """Run a block (for example one evaluation job) inside a user's scope and pacing profile."""
    name = str(scope or "")
    token = _scope_var.set(name)
    if profile is not None:
        set_pacing_profile(profile)
    try:
        yield name
    finally:
        _scope_var.reset(token)
        # Back-off learned during a job must not slow that user's later chat requests.
        _clear_scope_pacing(name)


def _prune_idle_scopes_locked(now: float) -> None:
    """Forget scopes that have been idle for an hour (caller holds _provider_states_lock)."""
    if now - _last_prune[0] < _SCOPE_PRUNE_EVERY_SECONDS:
        return
    _last_prune[0] = now
    stale = [
        scope
        for scope, used in _scope_last_used.items()
        if scope and now - used > _SCOPE_IDLE_SECONDS
    ]
    for scope in stale:
        _scope_last_used.pop(scope, None)
        for key in [key for key in _provider_states if key[0] == scope]:
            del _provider_states[key]
        for key in [key for key in _provider_semaphores if key[0] == scope]:
            del _provider_semaphores[key]
        _clear_scope_pacing(scope)


def get_min_interval_seconds(provider: str | None = None) -> float:
    """Pacing for a provider: the larger of env config, the active profile and the adaptive back-off."""
    name = (provider or "groq").lower()
    scope = _scope_var.get()
    with _pacing_lock:
        profile_name = _pacing_profiles.get(scope, "auto")
        profile = PACING_PROFILES.get(profile_name, {}).get(name, 0.0)
        adaptive = _adaptive_intervals.get((scope, name), 0.0)
    return max(PROVIDER_MIN_INTERVAL_SECONDS.get(name, 0.0), profile, adaptive)


def set_pacing_profile(profile: str | None) -> None:
    """Select the pacing profile for the current scope and reset its adaptive back-off."""
    scope = _scope_var.get()
    with _pacing_lock:
        _pacing_profiles[scope] = profile if profile in PACING_PROFILES else "auto"
        for key in [key for key in _adaptive_intervals if key[0] == scope]:
            del _adaptive_intervals[key]


def note_rate_limited(provider: str | None) -> float:
    """Slow a provider down (for the current scope only) after a 429. Returns the new spacing."""
    name = (provider or "groq").lower()
    scope = _scope_var.get()
    with _pacing_lock:
        profile_name = _pacing_profiles.get(scope, "auto")
        current = max(
            _adaptive_intervals.get((scope, name), 0.0),
            PROVIDER_MIN_INTERVAL_SECONDS.get(name, 0.0),
            PACING_PROFILES.get(profile_name, {}).get(name, 0.0),
        )
        value = min(ADAPTIVE_MAX_INTERVAL_SECONDS, max(2.0, current * 1.6))
        _adaptive_intervals[(scope, name)] = value
        return value


class RateLimiter:
    """Per-user, per-provider request gate shared by generator, judge, golden and embedding calls."""

    def __init__(
        self,
        min_interval_seconds: float | None = None,
        max_parallel_requests: int = 4,
        provider: str | None = None,
    ):
        self.provider = (provider or "groq").lower()
        self.min_interval_seconds = (
            get_min_interval_seconds(self.provider)
            if min_interval_seconds is None
            else max(0.0, min_interval_seconds)
        )
        self.max_parallel_requests = max(1, int(max_parallel_requests))
        self.scope = _scope_var.get()
        now = time.monotonic()
        with _provider_states_lock:
            _scope_last_used[self.scope] = now
            _prune_idle_scopes_locked(now)
            self._lock, self._last_call = _provider_states.setdefault(
                (self.scope, self.provider), (threading.Lock(), [0.0])
            )
            self._parallel = _provider_semaphores.setdefault(
                (self.scope, self.provider, self.max_parallel_requests),
                threading.BoundedSemaphore(self.max_parallel_requests),
            )

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait_needed = self.min_interval_seconds - (now - self._last_call[0])
            if wait_needed > 0:
                time.sleep(wait_needed)
            self._last_call[0] = time.monotonic()

    async def async_wait(self):
        await asyncio.get_running_loop().run_in_executor(None, self.wait)

    def __enter__(self):
        self._parallel.acquire()
        try:
            self.wait()
        except BaseException:
            self._parallel.release()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self._parallel.release()
        return False

    def _async_parallel(self):
        loop = asyncio.get_running_loop()
        with _async_semaphores_lock:
            loop_semaphores = _async_semaphores.setdefault(loop, {})
            return loop_semaphores.setdefault(
                (self.scope, self.provider, self.max_parallel_requests),
                asyncio.Semaphore(self.max_parallel_requests),
            )

    async def __aenter__(self):
        semaphore = self._async_parallel()
        await semaphore.acquire()
        try:
            await self.async_wait()
        except BaseException:
            semaphore.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._async_parallel().release()
        return False


shared_rate_limiter = RateLimiter()  # default (unscoped) limiter kept for backward compatibility


def wait_for_rate_limit(
    min_interval_seconds: float | None = None,
    *,
    provider: str | None = None,
):
    """Space request starts according to the selected provider's tier."""
    limiter = RateLimiter(min_interval_seconds=min_interval_seconds, provider=provider)
    limiter.wait()


async def async_wait_for_rate_limit(
    min_interval_seconds: float | None = None,
    *,
    provider: str | None = None,
):
    limiter = RateLimiter(min_interval_seconds=min_interval_seconds, provider=provider)
    await limiter.async_wait()


_DURATION_RE = re.compile(
    r"\s*(?:(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?)?\s*(?:(\d+(?:\.\d+)?)\s*m(?:in(?:ute)?s?)?(?![a-z]))?"
    r"\s*(?:(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?(?![a-z]))?",
    re.IGNORECASE,
)


def _duration_after(text: str, lead_pattern: str) -> float | None:
    """Parse '6m43.2s', '19h2m3s', '7.5s' or '350ms' following a phrase such as 'try again in'."""
    for match in re.finditer(lead_pattern, text, flags=re.IGNORECASE):
        tail = text[match.end():]
        ms = re.match(r"\s*(\d+(?:\.\d+)?)\s*ms\b", tail, re.IGNORECASE)
        if ms:
            return float(ms.group(1)) / 1000
        duration = _DURATION_RE.match(tail)
        if duration and any(duration.groups()):
            hours, minutes, seconds = (float(part) if part else 0.0 for part in duration.groups())
            return hours * 3600 + minutes * 60 + seconds
    return None


def get_retry_wait(
    error_text: str | BaseException | None,
    base_wait_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    attempt: int = 0,
) -> float:
    return min(_raw_retry_wait(error_text, base_wait_seconds, attempt), 7 * 24 * 3600.0)


def _raw_retry_wait(error_text, base_wait_seconds, attempt) -> float:
    error_message = str(error_text or "")
    header_waits = _retry_waits_from_headers(error_text)
    if header_waits:
        return max(header_waits) + 1.0

    parsed = _duration_after(error_message, r"(?:try again|retry|retrying|please retry)\s*(?:in|after)")
    if parsed is not None:
        return parsed + 1.0

    retry_after = re.search(r"retry[- ]after\s*[:=]?\s*(\d+(?:\.\d+)?)", error_message, re.IGNORECASE)
    if retry_after:
        return float(retry_after.group(1)) + 1.0

    if not error_message or is_rate_limit_error(error_text):
        return min(base_wait_seconds * (2 ** max(attempt, 0)), 120.0)

    return 0.0


def is_rate_limit_error(error: object) -> bool:
    message = str(error or "")
    return bool(
        re.search(
            r"\b429\b|rate[_ -]?limit|too many requests|quota exceeded|resource[_ -]?exhausted",
            message,
            flags=re.IGNORECASE,
        )
    )


def _header_value(headers, header_name: str):
    if headers is None:
        return None
    if hasattr(headers, "items"):
        items = headers.items()
    elif isinstance(headers, (list, tuple)):
        items = headers
    else:
        return None

    target = header_name.lower()
    for key, value in items:
        if str(key).lower() == target:
            return value
    return None


def _retry_waits_from_headers(error: object) -> list[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return []

    waits = []
    for header_name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        value = _header_value(headers, header_name)
        if value is None:
            continue

        value = str(value).strip()
        seconds_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*s", value, re.IGNORECASE)
        milliseconds_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*ms", value, re.IGNORECASE)
        if seconds_match:
            waits.append(float(seconds_match.group(1)))
        elif milliseconds_match:
            waits.append(float(milliseconds_match.group(1)) / 1000)
        else:
            try:
                waits.append(float(value))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    waits.append(max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds()))
                except (TypeError, ValueError, OverflowError):
                    continue
    return waits


class QuotaExhaustedError(RuntimeError):
    """A provider quota (daily, billing or very long window) that retrying cannot fix soon."""


_BILLING_PATTERN = re.compile(
    r"insufficient[_ ]quota|exceeded your current quota|credit balance is too low|"
    r"payment required|\b402\b|out of credits|free[- ]models?[- ]per[- ]day",
    re.IGNORECASE,
)
_DAILY_PATTERN = re.compile(
    r"per[ _-]?day|\bTPD\b|\bRPD\b|daily|PerDay|per[ _-]?month",
    re.IGNORECASE,
)


def classify_rate_limit(error: object) -> tuple[str, float | None]:
    """Return (kind, wait_seconds). kind is 'billing', 'daily' or 'minute'; wait may be None."""
    message = str(error or "")
    waits = _retry_waits_from_headers(error)
    wait = get_retry_wait(error, DEFAULT_RETRY_BASE_SECONDS, 0)
    explicit = max(waits) + 1.0 if waits else None
    if explicit is None and re.search(r"try again in|retry[- ]after|retry in", message, re.IGNORECASE):
        explicit = wait
    if _BILLING_PATTERN.search(message):
        return "billing", explicit
    if _DAILY_PATTERN.search(message):
        return "daily", explicit
    return "minute", explicit


def _reset_hint(wait: float | None) -> str:
    if not wait or wait <= 0:
        return ""
    if wait < 90:
        return f" The provider suggests retrying in about {int(wait)}s."
    if wait < 5400:
        return f" The provider suggests retrying in about {int(round(wait / 60))} min."
    return f" The provider suggests retrying in about {wait / 3600:.1f} h."


def plan_rate_limit_retry(
    error: object,
    attempt: int,
    max_attempts: int,
    *,
    provider: str | None = None,
    base_wait_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
) -> float | None:
    """Decide what to do after a rate-limit error.

    Returns seconds to wait before the next attempt, or None when a transient
    per-minute limit is still failing after ``max_attempts``. Raises
    ``QuotaExhaustedError`` when waiting cannot help: a billing or daily quota,
    or a required wait longer than ``RATE_LIMIT_MAX_WAIT_SECONDS``.
    """
    name = (provider or "groq").title()
    kind, explicit = classify_rate_limit(error)
    if kind == "billing":
        raise QuotaExhaustedError(
            f"{name} reports that the account has no usable quota or credit left."
            f"{_reset_hint(explicit)} Add credit or billing, or choose another provider or model."
        ) from (error if isinstance(error, BaseException) else None)
    wait = get_retry_wait(error, base_wait_seconds, attempt)
    if explicit is not None and explicit > RATE_LIMIT_MAX_WAIT_SECONDS:
        raise QuotaExhaustedError(
            f"{name} rate limit needs a wait of more than {int(RATE_LIMIT_MAX_WAIT_SECONDS)}s, so the run was stopped "
            f"instead of waiting.{_reset_hint(explicit)} Daily limits reset on a rolling window; "
            "choose another model or provider, or retry later."
        ) from (error if isinstance(error, BaseException) else None)
    if attempt + 1 >= max_attempts:
        if kind == "daily":
            raise QuotaExhaustedError(
                f"{name} daily quota appears to be exhausted.{_reset_hint(explicit)} "
                "Choose another model or provider, or retry after the limit resets."
            ) from (error if isinstance(error, BaseException) else None)
        return None
    note_rate_limited(provider)
    return min(max(wait, 1.0), RATE_LIMIT_MAX_WAIT_SECONDS)
