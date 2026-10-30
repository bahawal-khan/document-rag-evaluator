"""
Golden dataset — the "ground truth" Q&A pairs evaluation needs.

Since the papers aren't known in advance (the user uploads their own), we
can't hand-write this dataset — instead we ask the LLM to read chunks and
write its own question + answer pairs ("self-QA generation", a common
technique in RAG evaluation).

For large N (up to 500) we use BATCHED generation — each model call produces
Q&A for several chunks at once, not one call per question. A dataset of 500
questions uses about 100 generation calls before retries and validation.

Each generated question is also tagged with a difficulty level
(easy / medium / hard) by the LLM itself, so you can later see whether the
RAG system is weaker on harder questions.

Robustness: batches can fail (rate limits, malformed JSON) — this module
retries only failures that can plausibly succeed on a second attempt (rate
limits, malformed output) with backoff and JSON auto-repair (same pattern as
core/judge.py), and reports exactly how many questions were requested vs. how
many were actually produced, instead of silently returning fewer.

Configuration errors (bad key, unknown model or endpoint) are identical for
every batch, so generation stops after the first such failure — and after
several consecutive failed batches of any kind — instead of spending the rest
of the run on calls that cannot succeed. A run that produces nothing never
overwrites a dataset that was already saved.

Users can also upload their own golden_dataset.json from the Evaluation
Suite page — auto-generation is only the default/fallback.
"""

import json
import os
import random
import re
import time
import logging
from pathlib import Path

from core.config import EVALUATION_REQUEST_TIMEOUT_SECONDS, GENERATOR_MODEL, GOLDEN_DATASET_PATH
from core.generator import format_llm_error
from core.provider import build_chat_model, close_chat_model, provider_base_url
from core.rate_limit import QuotaExhaustedError, RateLimiter, is_rate_limit_error, plan_rate_limit_retry
from core.tracer import get_langsmith_callbacks

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

BATCH_SIZE = 5              # chunks per LLM call
MAX_BATCH_RETRIES = 3       # retries per batch before giving up on it
MAX_CONSECUTIVE_FAILED_BATCHES = 3   # stop a run that keeps failing
MAX_GOLDEN_QUESTIONS = int(os.getenv("MAX_GOLDEN_QUESTIONS", "1000") or 1000)
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
# Visible output is ~100 tokens per Q&A pair, but reasoning models also spend
# completion tokens on hidden reasoning; too small a cap yields truncated JSON.
GENERATION_TOKENS_PER_CHUNK = 200
GENERATION_TOKEN_HEADROOM = 1024
logger = logging.getLogger(__name__)

# Errors that repeat identically for every batch: authentication, permission,
# unknown model/endpoint. Retrying or continuing only burns API calls.
_FATAL_ERROR_RE = re.compile(
    r"\b(?:401|403|404)\b|unauthori[sz]ed|invalid api key|authentication|"
    r"permission denied|not found|model_not_found|does not exist|decommissioned",
    flags=re.IGNORECASE,
)

QA_GEN_BATCH_PROMPT = """You are given {n} research paper excerpts below, each
numbered. For EACH excerpt, write ONE factual question and its correct answer,
answerable only from that excerpt. Also classify each question's difficulty:

- "easy": answer is a single fact, directly and explicitly stated
- "medium": answer requires combining 2+ facts from the excerpt, or light inference
- "hard": answer requires nuanced understanding, comparison, or synthesis of the excerpt

{excerpts}

Respond with ONLY a JSON array (no other text), in this exact order:
[{{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}}, ...]
There must be exactly {n} objects, one per excerpt, in the same order.
"""


def _extract_json_array(text: str) -> str:
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _try_repair(text: str):
    if not _HAS_JSON_REPAIR:
        return None
    try:
        fixed = repair_json(text)
        if fixed and fixed.strip() not in ("", "[]", "{}"):
            json.loads(fixed)
            return fixed
    except Exception:
        pass
    return None


def _build_llm(model_name=None, provider=None, api_key=None):
    return build_chat_model(
        model_name or GENERATOR_MODEL,
        provider=provider,
        api_key=api_key,
        temperature=0.3,
        max_tokens=BATCH_SIZE * GENERATION_TOKENS_PER_CHUNK + GENERATION_TOKEN_HEADROOM,
        timeout=EVALUATION_REQUEST_TIMEOUT_SECONDS,
    )


def _safe_generation_error(error, provider, model_name, api_key):
    message = str(error)
    if isinstance(error, QuotaExhaustedError):
        return message[:500]
    if (
        "json" in message.casefold()
        or "question/answer records" in message.casefold()
    ):
        return f"Golden-data response validation failed: {message[:400]}"
    try:
        base_url = provider_base_url(provider)
    except ValueError:
        base_url = None
    return format_llm_error(
        error,
        provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
    )


def _is_retryable_batch_error(error: Exception) -> bool:
    message = str(error).casefold()
    return (
        is_rate_limit_error(error)
        or "malformed json" in message
        or "could not parse or repair json" in message
        or "empty response" in message
    )


def _is_fatal_provider_error(error: Exception) -> bool:
    """True for configuration errors that every other batch would repeat."""
    if is_rate_limit_error(error):
        return False
    return bool(_FATAL_ERROR_RE.search(str(error)))


def _generate_batch(
    llm,
    batch_chunks,
    callbacks,
    provider="groq",
    api_key=None,
    model_name=None,
    error_callback=None,
    fatal_callback=None,
    quota_callback=None,
):
    """Generates Q&A for one batch of chunks, with retry + JSON repair.

    Returns a list of parsed dicts. Each dict carries ``_position``: the index
    of the excerpt it answers, so that a malformed or missing item in the
    middle of the array can never shift later questions onto the wrong source
    chunk. The list may be shorter than the batch."""
    excerpts_text = "\n\n".join(f"[{i+1}] {text}" for i, (text, _) in enumerate(batch_chunks))
    prompt = QA_GEN_BATCH_PROMPT.format(n=len(batch_chunks), excerpts=excerpts_text)
    last_error = RuntimeError("Golden-data batch failed without a provider response.")

    for attempt in range(MAX_BATCH_RETRIES):
        try:
            with RateLimiter(provider=provider):
                raw = llm.invoke(prompt, config={"callbacks": callbacks} if callbacks else None).content
            extracted = _extract_json_array(raw)
            try:
                parsed_list = json.loads(extracted)
            except json.JSONDecodeError:
                fixed = _try_repair(extracted)
                if not fixed:
                    raise ValueError("Could not parse or repair JSON from LLM response")
                parsed_list = json.loads(fixed)

            if not isinstance(parsed_list, list):
                raise ValueError("The model returned JSON, but not a JSON array.")
            valid = []
            for position, p in enumerate(parsed_list):
                if (
                    isinstance(p, dict)
                    and isinstance(p.get("question"), str)
                    and p["question"].strip()
                    and isinstance(p.get("answer"), str)
                    and p["answer"].strip()
                ):
                    valid.append({**p, "_position": position})
            if not valid:
                raise ValueError(
                    "The model returned a JSON array, but no valid question/answer records."
                )
            for p in valid:
                if p.get("difficulty") not in DIFFICULTY_LEVELS:
                    p["difficulty"] = "medium"  # safe default if missing/invalid
            return valid

        except Exception as e:
            last_error = e
            if is_rate_limit_error(e):
                try:
                    wait_seconds = plan_rate_limit_retry(
                        e, attempt, MAX_BATCH_RETRIES, provider=provider, base_wait_seconds=10
                    )
                except QuotaExhaustedError as quota:
                    # Daily/billing quota: every later batch would fail the same way.
                    last_error = quota
                    break
                if wait_seconds is None:
                    break
                logger.warning(
                    "Golden-data batch was rate limited; retrying in %.1f seconds (attempt %s/%s).",
                    wait_seconds,
                    attempt + 1,
                    MAX_BATCH_RETRIES,
                )
                time.sleep(wait_seconds)
                continue
            if _is_retryable_batch_error(e) and attempt < MAX_BATCH_RETRIES - 1:
                time.sleep(2)
                continue
            break

    safe_error = _safe_generation_error(
        last_error,
        provider,
        model_name,
        api_key,
    )
    if error_callback:
        error_callback(safe_error)
    if isinstance(last_error, QuotaExhaustedError):
        if quota_callback:
            quota_callback(safe_error)
    elif fatal_callback and _is_fatal_provider_error(last_error):
        fatal_callback(safe_error)
    logger.error(
        "Golden-data batch failed after %s attempts (%s): %s.",
        min(attempt + 1, MAX_BATCH_RETRIES),
        type(last_error).__name__,
        safe_error,
    )
    return []


def _sample_collection_chunks(collection, n: int, seed: int, progress_callback=None):
    total = collection.count()
    sample_count = min(n, total)
    if sample_count == 0:
        raise ValueError("The document index is empty. Upload and index documents first.")

    rng = random.Random(seed)
    reservoir = []
    scanned = 0
    offset = 0
    while offset < total:
        page = collection.get(
            limit=250,
            offset=offset,
            include=["documents", "metadatas"],
        )
        documents = page.get("documents") or []
        metadatas = page.get("metadatas") or []
        if not documents:
            break
        for document, metadata in zip(documents, metadatas):
            scanned += 1
            row = (document, metadata or {})
            if len(reservoir) < sample_count:
                reservoir.append(row)
            else:
                candidate = rng.randrange(scanned)
                if candidate < sample_count:
                    reservoir[candidate] = row
        offset += len(documents)
        if progress_callback and (
            offset % 2500 == 0 or offset >= total or scanned >= total
        ):
            progress_callback(min(scanned, total), total)
    return reservoir, scanned


def _save_checkpoint(dataset, dataset_path):
    if dataset_path is None:
        return
    target = Path(dataset_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)


def save_golden_dataset(dataset, dataset_path: str | Path | None = None):
    """Persist an uploaded or edited dataset with an atomic file replacement."""
    target = Path(dataset_path) if dataset_path is not None else GOLDEN_DATASET_PATH
    _save_checkpoint(validate_golden_dataset(dataset), target)


def validate_golden_dataset(dataset) -> list[dict]:
    """Validate uploaded or edited records before evaluation or persistence."""
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON array.")
    if len(dataset) > MAX_GOLDEN_QUESTIONS:
        raise ValueError(f"Dataset can contain at most {MAX_GOLDEN_QUESTIONS} records.")
    normalized = []
    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset record {index} must be a JSON object.")
        question = item.get("input")
        expected = item.get("expected_output")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Dataset record {index} needs a non-empty input question.")
        if len(question) > 8000:
            raise ValueError(f"Dataset record {index} question exceeds 8,000 characters.")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"Dataset record {index} needs a non-empty expected_output.")
        if len(expected) > 20000:
            raise ValueError(f"Dataset record {index} expected_output exceeds 20,000 characters.")
        page = item.get("page")
        if page is not None and (not isinstance(page, (int, str)) or len(str(page)) > 80):
            raise ValueError(f"Dataset record {index} has an invalid page reference.")
        for field in ("source", "source_reference", "location", "source_file_id"):
            value = item.get(field)
            if value is not None and (not isinstance(value, str) or len(value) > 500):
                raise ValueError(f"Dataset record {index} has an invalid {field} reference.")
        normalized_item = {
            "input": question.strip(),
            "expected_output": expected.strip(),
        }
        if item.get("difficulty") in DIFFICULTY_LEVELS:
            normalized_item["difficulty"] = item["difficulty"]
        else:
            normalized_item["difficulty"] = "medium"
        for field in ("source", "source_reference", "location", "page"):
            if item.get(field) is not None:
                normalized_item[field] = item[field]
        for field in ("relevant_chunk_ids", "relevant_sources", "relevant_locations", "relevant_documents"):
            value = item.get(field)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            if len(values) > 100:
                raise ValueError(f"Dataset record {index} has too many {field} references.")
            if field == "relevant_documents":
                if any(
                    not isinstance(reference, (str, dict))
                    or (isinstance(reference, str) and len(reference) > 500)
                    or (
                        isinstance(reference, dict)
                        and any(
                            key not in {"chunk_id", "source", "page", "location", "file_id"}
                            or (value is not None and len(str(value)) > 500)
                            for key, value in reference.items()
                        )
                    )
                    for reference in values
                ):
                    raise ValueError(f"Dataset record {index} has invalid relevant_documents references.")
            elif any(not isinstance(reference, str) or not reference.strip() or len(reference) > 500 for reference in values):
                raise ValueError(f"Dataset record {index} has invalid {field} references.")
            normalized_item[field] = values
        normalized.append(normalized_item)
    return normalized


def _question_key(question: str) -> str:
    return re.sub(r"\W+", " ", question.casefold()).strip()


def generate_golden_dataset(
    collection,
    n: int = 8,
    seed: int = 42,
    progress_callback=None,
    *,
    model_name: str | None = None,
    model_provider: str | None = None,
    api_key: str | None = None,
    dataset_path: str | Path | None = None,
    sampling_progress_callback=None,
):
    """
    progress_callback(done, total, succeeded_batches, failed_batches) — for a
    Streamlit progress bar, called after every batch.

    Returns (golden_data, stats) where stats reports exactly how many
    questions were requested vs. produced, and why any were missing
    (including ``aborted_reason`` / ``batches_skipped`` when the run stopped
    early because further calls could not succeed).
    """
    n = max(1, min(int(n), MAX_GOLDEN_QUESTIONS))
    sampled_chunks, scanned_chunks = _sample_collection_chunks(
        collection,
        n,
        seed,
        sampling_progress_callback,
    )
    llm = _build_llm(model_name, model_provider, api_key)
    callbacks = get_langsmith_callbacks()
    golden = []
    batches_total = -(-len(sampled_chunks) // BATCH_SIZE)
    batches_succeeded = 0
    batches_failed = 0
    batches_skipped = 0
    consecutive_failures = 0
    aborted_reason = None
    errors = []
    fatal_errors = []
    quota_errors = []
    duplicate_questions = 0
    seen_questions = set()

    for batch_index, batch_start in enumerate(range(0, len(sampled_chunks), BATCH_SIZE)):
        batch_chunks = [
            (document[:900], metadata)
            for document, metadata in sampled_chunks[batch_start: batch_start + BATCH_SIZE]
        ]

        results = _generate_batch(
            llm,
            batch_chunks,
            callbacks,
            provider=model_provider or "groq",
            api_key=api_key,
            model_name=model_name or GENERATOR_MODEL,
            error_callback=errors.append,
            fatal_callback=fatal_errors.append,
            quota_callback=quota_errors.append,
        )

        if results:
            batches_succeeded += 1
            consecutive_failures = 0
        else:
            batches_failed += 1
            consecutive_failures += 1

        for fallback_position, parsed in enumerate(results):
            position = parsed.get("_position", fallback_position)
            if not isinstance(position, int) or not 0 <= position < len(batch_chunks):
                continue
            meta = batch_chunks[position][1]
            question = str(parsed.get("question", "")).strip()
            answer = str(parsed.get("answer", "")).strip()
            if api_key:
                question = re.sub(re.escape(api_key), "[redacted]", question, flags=re.IGNORECASE)
                answer = re.sub(re.escape(api_key), "[redacted]", answer, flags=re.IGNORECASE)
            key = _question_key(question)
            if not question or not answer or not key or key in seen_questions:
                duplicate_questions += 1
                continue
            seen_questions.add(key)
            golden.append({
                "input": question,
                "expected_output": answer,
                "difficulty": parsed["difficulty"],
                "source": meta.get("source"),
                "page": meta.get("page"),
                "source_reference": meta.get("location"),
                "source_file_id": meta.get("file_id"),
            })
        if golden:
            _save_checkpoint(golden, dataset_path)

        if progress_callback:
            progress_callback(
                min(batch_start + BATCH_SIZE, len(sampled_chunks)), len(sampled_chunks),
                batches_succeeded, batches_failed,
            )

        remaining = batches_total - (batch_index + 1)
        if quota_errors and remaining > 0:
            aborted_reason = (
                f"Stopped because the provider quota is exhausted: {quota_errors[0]} "
                "No further calls were made."
            )
            batches_skipped = remaining
            break
        if fatal_errors and remaining > 0:
            aborted_reason = (
                "Stopped after the first batch because the provider rejected the request "
                "(authentication, permission, or unknown model/endpoint). Every remaining batch "
                "would fail the same way, so no further calls were made."
            )
            batches_skipped = remaining
            break
        if consecutive_failures >= MAX_CONSECUTIVE_FAILED_BATCHES and remaining > 0:
            aborted_reason = (
                f"Stopped after {consecutive_failures} consecutive failed batches to avoid "
                "spending more provider calls on a request that keeps failing."
            )
            batches_skipped = remaining
            break

    close_chat_model(llm)

    # Never replace a previously saved dataset with an empty one.
    if golden:
        _save_checkpoint(golden, dataset_path or GOLDEN_DATASET_PATH)

    stats = {
        "requested": len(sampled_chunks),
        "produced": len(golden),
        "scanned_chunks": scanned_chunks,
        "duplicate_questions": duplicate_questions,
        "batches_total": batches_total,
        "batches_succeeded": batches_succeeded,
        "batches_failed": batches_failed,
        "batches_skipped": batches_skipped,
        "aborted_reason": aborted_reason,
        "errors": list(dict.fromkeys(errors))[:batches_failed],
        "difficulty_counts": {
            level: sum(1 for g in golden if g["difficulty"] == level)
            for level in DIFFICULTY_LEVELS
        },
    }
    return golden, stats


def load_golden_dataset(dataset_path: str | Path | None = None):
    path = Path(dataset_path) if dataset_path is not None else GOLDEN_DATASET_PATH
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return validate_golden_dataset(data)
        except (OSError, json.JSONDecodeError, ValueError):
            return []
    return []


def difficulty_breakdown(golden_data):
    """Counts questions per difficulty level — used for display and for
    building a stratified sample when running evaluation on a subset."""
    counts = {level: 0 for level in DIFFICULTY_LEVELS}
    for item in golden_data:
        level = item.get("difficulty", "medium")
        if not isinstance(level, str) or level not in counts:
            level = "medium"
        counts[level] += 1
    return counts
