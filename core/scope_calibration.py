"""Pick the scope-check threshold from the user's own document instead of guessing.

The chat refuses a question when the best retrieved passage scores below a
threshold. Cross-encoder scores differ by model and by document, so a fixed
number is either too strict (every question refused) or too loose. Given the
best-passage scores of questions that ARE about the document (from the golden
dataset) and of unrelated probes, this picks a threshold that keeps almost all
real questions and, when the two groups separate, blocks the unrelated ones.
"""

import math

MIN_IN_SCOPE_SAMPLES = 3
THRESHOLD_LIMIT = 20.0


def _percentile(sorted_values: list[float], percent: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _clean(scores) -> list[float]:
    cleaned = []
    for score in scores or []:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            cleaned.append(value)
    return sorted(cleaned)


def calibrate_scope_threshold(in_scope_scores, off_topic_scores=None) -> dict:
    """Return {"threshold", "separable", "in_scope_kept", "off_topic_blocked", ...}."""
    inside = _clean(in_scope_scores)
    outside = _clean(off_topic_scores)
    if len(inside) < MIN_IN_SCOPE_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_IN_SCOPE_SAMPLES} scored questions about the document to calibrate; "
            f"got {len(inside)}."
        )
    low_in = _percentile(inside, 5)
    off_max = outside[-1] if outside else None
    if off_max is not None and off_max < low_in:
        threshold = (off_max + low_in) / 2.0
        separable = True
    else:
        # Overlap (or no probes): favour answering. The grounded prompt still declines
        # questions the passages do not support, so a loose gate costs one LLM call.
        threshold = low_in - 0.5
        separable = off_max is None
    threshold = max(-THRESHOLD_LIMIT, min(THRESHOLD_LIMIT, threshold))
    return {
        "threshold": round(threshold, 2),
        "separable": separable,
        "in_scope_samples": len(inside),
        "off_topic_samples": len(outside),
        "in_scope_p05": round(low_in, 2),
        "in_scope_median": round(_percentile(inside, 50), 2),
        "off_topic_max": None if off_max is None else round(off_max, 2),
        "in_scope_kept": sum(1 for v in inside if v >= threshold) / len(inside),
        "off_topic_blocked": (sum(1 for v in outside if v < threshold) / len(outside)) if outside else None,
    }
