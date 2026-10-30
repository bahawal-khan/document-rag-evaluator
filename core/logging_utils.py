"""Small JSON event logger that accepts only non-sensitive diagnostic fields."""

import json
import logging

_SAFE_FIELDS = {
    "provider",
    "method",
    "error_type",
    "attempt",
    "retry_after_s",
    "duration_s",
    "status",
    "count",
}


def log_event(logger: logging.Logger, level: int, event: str, **fields) -> None:
    payload = {"event": str(event)[:80]}
    for name, value in fields.items():
        if name not in _SAFE_FIELDS or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            payload[name] = value if not isinstance(value, str) else value[:80]
    logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))

