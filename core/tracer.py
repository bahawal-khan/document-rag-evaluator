"""
Logs every live query to a trace file (JSONL — one line per call). This is
a lightweight, "LangSmith-style" local trace log.

Want REAL LangSmith too? Set these optional environment values or Streamlit secrets:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=rag-research-assistant
Every LLM call (generator + judge) explicitly attaches a LangChainTracer
callback when these are set.
"""

import json
import os
import time
from pathlib import Path
import numpy as np
from core.config import TRACE_LOG_PATH

_TRACE_FIELDS = {
    "cache_hit",
    "cache_latency_s",
    "provider",
    "cost_usd",
    "error",
    "generation_status",
    "toxicity_blocked",
    "guardrail_latency_s",
    "scope_blocked",
    "input_tokens",
    "output_tokens",
    "ttft_s",
    "total_latency_s",
    "pii_flagged",
    "timed_out",
    "retrieval_method",
    "error_type",
    "api_calls",
    "retry_count",
}


def langsmith_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() == "true" and bool(os.getenv("LANGCHAIN_API_KEY"))


def get_langsmith_callbacks():
    """Rather than relying purely on ambient env-var detection, this builds
    an explicit LangChainTracer callback and passes it into every LLM call —
    more reliable (avoids timing/env-var-caching issues)."""
    if not langsmith_enabled():
        return []
    try:
        from langchain_core.tracers.langchain import LangChainTracer
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        return [LangChainTracer(project_name=project)]
    except Exception:
        return []


def test_langsmith_connection():
    """Diagnostic: actually pings the LangSmith API to confirm connectivity/auth,
    rather than just checking 'is the env var set'. Returns (success: bool, message: str)."""
    if not langsmith_enabled():
        return False, "LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are not both set."
    try:
        from langsmith import Client
        client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
        list(client.list_projects(limit=1))  # lightweight call just to confirm auth/connectivity
        return True, f"LangSmith connection OK. Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}"
    except ImportError:
        return False, "The 'langsmith' package isn't installed — run: pip install langsmith"
    except Exception as e:
        message = str(e)
        key = os.getenv("LANGCHAIN_API_KEY")
        if key:
            message = message.replace(key, "[redacted]")
        return False, f"LangSmith connection FAILED: {message}"


def log_trace(entry: dict, trace_path: str | Path | None = None):
    # Trace files are operational telemetry only; never persist arbitrary
    # request fields, exception text, custom model IDs, or credential fields.
    safe_entry = {}
    for key, value in entry.items():
        if key not in _TRACE_FIELDS:
            continue
        if key == "pii_flagged" and isinstance(value, list):
            safe_entry[key] = [
                item for item in value
                if item in {"email", "credit_card_like", "ssn_like", "phone_number"}
            ]
        elif value is None or isinstance(value, (str, int, float, bool)):
            safe_entry[key] = value
    safe_entry["timestamp"] = time.time()
    path = Path(trace_path) if trace_path is not None else TRACE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")


def clear_traces(trace_path: str | Path | None = None):
    """Deletes old traces — useful after a config/code change, so fresh
    operational metrics (latency/cost) aren't skewed by stale entries."""
    path = Path(trace_path) if trace_path is not None else TRACE_LOG_PATH
    if path.exists():
        path.unlink()


def load_traces(trace_path: str | Path | None = None):
    path = Path(trace_path) if trace_path is not None else TRACE_LOG_PATH
    if not path.exists():
        return []
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except Exception:
                    continue
    return traces


def _trace_is_error(trace: dict) -> bool:
    status = trace.get("generation_status")
    if status is not None:
        return status != "success"

    # Legacy traces stored successful generations as error=True, but stored
    # failed provider calls as their error string.
    legacy_error = trace.get("error")
    return isinstance(legacy_error, str) and bool(legacy_error)


def compute_operational_metrics(trace_path: str | Path | None = None):
    """Computes latency percentiles, cost, cache-hit-rate, and reliability
    from the trace log."""
    traces = load_traces(trace_path)
    if not traces:
        return None

    # Cache responses and pre-generation blocks are useful operational events,
    # but they are not provider-generation latency samples.
    generation_traces = [
        t for t in traces
        if not t.get("cache_hit")
        and not t.get("toxicity_blocked")
        and not t.get("scope_blocked")
        and t.get("total_latency_s") is not None
    ]
    latencies = [t["total_latency_s"] for t in generation_traces]
    ttfts = [t["ttft_s"] for t in generation_traces if t.get("ttft_s") is not None]
    costs = [t["cost_usd"] for t in traces if isinstance(t.get("cost_usd"), (int, float))]
    cache_hits = sum(1 for t in traces if t.get("cache_hit"))
    cache_latencies = [
        t.get("cache_latency_s", t.get("total_latency_s"))
        for t in traces if t.get("cache_hit")
    ]
    cache_latencies = [value for value in cache_latencies if isinstance(value, (int, float))]
    # New traces carry an explicit status. Before that field existed, normal
    # successful generations were accidentally written with error=True, while
    # actual provider failures were stored as their error string. Preserve the
    # useful historical distinction without treating the legacy success flag
    # as a failure.
    errors = sum(1 for trace in traces if _trace_is_error(trace))

    def api_calls(trace):
        if "api_calls" in trace:
            return max(0, int(trace.get("api_calls") or 0))
        # Legacy traces represented one provider generation by one trace line.
        return 0 if trace.get("cache_hit") or trace.get("toxicity_blocked") or trace.get("scope_blocked") else 1

    total_api_calls = sum(api_calls(t) for t in traces)
    generation_api_calls = sum(api_calls(t) for t in generation_traces)

    return {
        "total_requests": total_api_calls,
        "generation_requests": generation_api_calls,
        "cache_hits": cache_hits,
        "p50_latency_s": float(np.percentile(latencies, 50)) if latencies else None,
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else None,
        "p99_latency_s": float(np.percentile(latencies, 99)) if latencies else None,
        "avg_ttft_s": float(np.mean(ttfts)) if ttfts else None,
        "avg_cache_latency_s": float(np.mean(cache_latencies)) if cache_latencies else None,
        "total_cost_usd": sum(costs) if costs else None,
        "avg_cost_usd": float(np.mean(costs)) if costs else None,
        "cost_estimates_count": len(costs),
        "cost_unknown_count": sum(
            1 for t in traces if t.get("cost_usd") is None and not t.get("cache_hit")
        ),
        "cache_hit_rate": cache_hits / len(traces) if traces else 0,
        "reliability": 1 - (errors / len(traces)) if traces else 1,
        "error_count": errors,
    }
