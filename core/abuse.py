"""Per-session request throttle: one visitor cannot burn the whole server's capacity."""

from collections import deque
import os
import threading
import time

MAX_CHAT_PER_MINUTE = max(1, int(os.getenv("MAX_CHAT_PER_MINUTE", "40") or 40))
_WINDOW_SECONDS = 60.0
_IDLE_FORGET_SECONDS = 600.0
_lock = threading.Lock()
_recent: dict[str, deque] = {}
_last_cleanup = [0.0]


def chat_allowed(scope: str, *, now: float | None = None) -> tuple[bool, int]:
    """Return (allowed, seconds_to_wait). Counts the request when it is allowed."""
    current = time.monotonic() if now is None else now
    with _lock:
        if current - _last_cleanup[0] > 60:
            _last_cleanup[0] = current
            for key in [k for k, q in _recent.items() if not q or current - q[-1] > _IDLE_FORGET_SECONDS]:
                del _recent[key]
        window = _recent.setdefault(scope, deque())
        while window and current - window[0] > _WINDOW_SECONDS:
            window.popleft()
        if len(window) >= MAX_CHAT_PER_MINUTE:
            return False, max(1, int(_WINDOW_SECONDS - (current - window[0])) + 1)
        window.append(current)
        return True, 0
