"""
Central config — paths, model names, thresholds. Every module imports from
here so all tunable numbers live in one place.
"""

import os
import logging
import math
from pathlib import Path

from core.runtime import bootstrap

# Telemetry-off defaults, allow-listed Streamlit secrets -> environment, and the
# Windows asyncio log filter. Runs before any setting below is read.
bootstrap()

logger = logging.getLogger(__name__)


def env_float(name: str, default: float, *, minimum=None, maximum=None) -> float:
    """Read a bounded numeric setting without making startup depend on it."""
    raw_value = os.getenv(name)
    try:
        value = float(raw_value) if raw_value is not None else float(default)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError):
        logger.warning("Invalid numeric configuration for %s; using the default.", name)
        return float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAPERS_DIR = PROJECT_ROOT / "papers"
DATA_DIR = PROJECT_ROOT / "data"
TRACES_DIR = PROJECT_ROOT / "traces"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CACHE_FILE = DATA_DIR / "answer_cache.json"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.json"
TRACE_LOG_PATH = TRACES_DIR / "traces.jsonl"

COLLECTION_NAME = "research_papers"

# ---- Models ----
# These model defaults support legacy callers. Streamlit UI requests pass the
# user's selected role model and API key directly and do not read provider keys
# from server environment variables.
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "groq").lower()

# ---- Retrieval ----
TOP_K = 4
RETRIEVAL_CANDIDATE_K = 32
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANKER_SCOPE_THRESHOLD = env_float(
    "RERANKER_SCOPE_THRESHOLD", -4.0, minimum=-100.0, maximum=100.0
)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
# Used only when the cross-encoder is unavailable; calibrate against the app's
# own labeled in-scope and out-of-scope questions before changing this value.
SCOPE_DISTANCE_THRESHOLD = env_float(
    "SCOPE_DISTANCE_THRESHOLD", 1.0, minimum=0.0, maximum=100.0
)

# ---- Token / cost budget (live chat guardrails) ----
MAX_INPUT_TOKENS = 1500     # system + retrieved context + question, combined
MAX_OUTPUT_TOKENS = 650     # A stricter 500 caused incomplete answers that
                            # hurt Correctness/Completeness scores — this is
                            # a balance between latency and answer quality.

# ---- Latency targets (seconds) ----
P50_TARGET_SECONDS = 2.0
P50_HARD_LIMIT_SECONDS = 2.5
P99_HARD_LIMIT_SECONDS = 3.5
# If a single call takes longer than this, it's forcibly cut off (whatever
# was generated so far is shown) — the deadline is enforced per request.
# Raise it with HARD_REQUEST_DEADLINE_SECONDS for slower providers or models.
HARD_REQUEST_DEADLINE_SECONDS = env_float(
    "HARD_REQUEST_DEADLINE_SECONDS", 4.0, minimum=1.0, maximum=120.0
)

# ---- Offline evaluation timeouts ----
EVALUATION_REQUEST_TIMEOUT_SECONDS = 90
EVALUATION_TASK_TIMEOUT_SECONDS = 900
# DeepEval reads its timeout settings when its models are first imported.
os.environ.setdefault(
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
    str(EVALUATION_TASK_TIMEOUT_SECONDS),
)

# ---- Eval thresholds ----
METRIC_THRESHOLD = 0.75
