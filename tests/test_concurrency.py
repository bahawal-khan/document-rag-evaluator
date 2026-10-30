"""Multi-user isolation: rate limits, pacing, job guard and worker configuration."""
import contextvars
import threading
import time

import pytest

from core import rate_limit


@pytest.fixture(autouse=True)
def _clean_scope(tmp_path, monkeypatch):
    from core import evaluation_jobs

    monkeypatch.setattr(evaluation_jobs, "DATA_DIR", tmp_path)  # never write job files into the project
    yield
    rate_limit._scope_var.set("")


def test_backoff_is_isolated_between_users():
    with rate_limit.rate_limit_scope("user-a", "free"):
        slowed = rate_limit.note_rate_limited("groq")
        assert rate_limit.get_min_interval_seconds("groq") >= slowed
    with rate_limit.rate_limit_scope("user-b", "paid"):
        assert rate_limit.get_min_interval_seconds("groq") < slowed


def test_pacing_profile_does_not_leak_between_threads():
    seen = {}

    def worker(name, profile):
        with rate_limit.rate_limit_scope(name, profile):
            time.sleep(0.05)
            seen[name] = rate_limit.get_min_interval_seconds("groq")

    threads = [
        threading.Thread(target=worker, args=("free-user", "free")),
        threading.Thread(target=worker, args=("paid-user", "paid")),
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen["free-user"] > seen["paid-user"]


def test_scope_state_cleared_after_job():
    with rate_limit.rate_limit_scope("tmp-user", "free"):
        rate_limit.note_rate_limited("groq")
    with rate_limit.rate_limit_scope("tmp-user"):
        assert ("tmp-user", "groq") not in rate_limit._adaptive_intervals


def test_scope_propagates_into_copied_context():
    rate_limit.bind_rate_limit_scope("chat-user")
    out = []
    ctx = contextvars.copy_context()
    t = threading.Thread(target=ctx.run, args=(lambda: out.append(rate_limit.current_scope()),))
    t.start()
    t.join()
    assert out == ["chat-user"]


def test_one_active_job_per_workspace(monkeypatch):
    from core import evaluation_jobs as jobs

    ws = "a" * 32
    with jobs._workspace_lock:
        jobs._active_workspaces.add(ws)
    try:
        with pytest.raises(ValueError, match="already have an evaluation"):
            jobs.submit_evaluation_job(
                job_id="b" * 32,
                workspace_id=ws,
                golden_data=[{"input": "q", "expected_output": "a"}],
                options={"run_component": True},
                answer_settings={"provider": "groq", "model": "m", "api_key": "k"},
                judge_settings={"provider": "groq", "model": "m", "api_key": "k"},
                retrieval_settings={"method": "dense", "embedding_provider": "local", "embedding_model": "all-MiniLM-L6-v2"},
            )
    finally:
        jobs._release_workspace(ws)


def test_worker_pool_is_configurable(monkeypatch):
    from core import evaluation_jobs as jobs

    monkeypatch.setenv("EVAL_MAX_WORKERS", "3")
    assert jobs._env_int("EVAL_MAX_WORKERS", 2, 1, 16) == 3
    monkeypatch.setenv("EVAL_MAX_WORKERS", "junk")
    assert jobs._env_int("EVAL_MAX_WORKERS", 2, 1, 16) == 2
    monkeypatch.setenv("EVAL_MAX_WORKERS", "999")
    assert jobs._env_int("EVAL_MAX_WORKERS", 2, 1, 16) == 16


def test_deepeval_scoring_is_serialised_across_users(monkeypatch):
    from eval import common

    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    def fake_evaluate(*args, **kwargs):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with guard:
            active["now"] -= 1
        return "ok"

    monkeypatch.setattr(common, "evaluate", fake_evaluate)
    threads = [threading.Thread(target=common.evaluate_test_cases, args=([], [])) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert active["max"] == 1


def test_local_embedding_model_loads_once_for_simultaneous_users(monkeypatch):
    import sys
    import types

    from core import embeddings

    loads = []

    class FakeModel:
        def __init__(self, name):
            loads.append(name)
            time.sleep(0.1)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    embeddings._local_model.cache_clear()
    threads = [threading.Thread(target=embeddings._local_model, args=("m",)) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    embeddings._local_model.cache_clear()
    assert loads == ["m"]


def test_crashing_job_returns_its_slot_and_user_lock(monkeypatch):
    from core import evaluation_jobs as jobs

    ws, job_id = "c" * 32, "d" * 32
    assert jobs._queue_slots.acquire(blocking=False)
    with jobs._workspace_lock:
        jobs._active_workspaces.add(ws)
        jobs._live_jobs.add(job_id)
    free_before = jobs._queue_slots._value
    jobs._write_job(job_id, {"job_id": job_id, "workspace_id": ws, "status": "queued"})

    def boom(*args, **kwargs):
        raise RuntimeError("crash before the job's own try block")

    monkeypatch.setattr(jobs, "_run_job", boom)
    jobs._execute_job(job_id, ws, [], {}, {}, {}, {})
    assert ws not in jobs._active_workspaces
    assert job_id not in jobs._live_jobs
    assert jobs._queue_slots._value == free_before + 1
    assert jobs.load_evaluation_job(job_id)["status"] == "failed"


def test_job_left_running_by_a_restart_is_reported_as_failed():
    from core import evaluation_jobs as jobs

    job_id, ws = "e" * 32, "f" * 32
    jobs._write_job(job_id, {"job_id": job_id, "workspace_id": ws, "status": "running"})
    record = jobs.load_evaluation_job(job_id, workspace_id=ws)
    assert record["status"] == "failed"
    assert "restarted" in record["error_message"]


def test_document_with_absurd_chunk_count_is_rejected_and_rolled_back(tmp_path, monkeypatch):
    from core import ingest

    monkeypatch.setattr(ingest, "MAX_CHUNKS_PER_DOCUMENT", 3)
    monkeypatch.setattr(ingest, "INDEX_BATCH_SIZE", 2)
    path = tmp_path / "big.txt"
    path.write_text("\n".join(f"paragraph {i} " + "word " * 120 for i in range(60)), encoding="utf-8")

    class Collection:
        def __init__(self):
            self.ids = []

        def get(self, **kwargs):
            return {"ids": []}

        def delete(self, **kwargs):
            for item in kwargs.get("ids", []):
                self.ids.remove(item)

    class Lexical:
        def upsert_batch(self, *args, **kwargs): pass
        def delete_chunk_ids(self, *args, **kwargs): pass
        def delete_stale_source_versions(self, *args, **kwargs): pass

    collection = Collection()
    monkeypatch.setattr(ingest, "_upsert_batch", lambda col, fid, first, batch, emb: col.ids.extend(ingest._chunk_ids(fid, first, len(batch))))
    with pytest.raises(ValueError, match="more than 3"):
        ingest._index_files([("big.txt", path)], collection, Lexical(), object(), None)
    assert collection.ids == []  # nothing from the rejected document is left in the index


def test_fake_pdf_gives_clean_error(tmp_path):
    from core import ingest

    path = tmp_path / "fake.pdf"
    path.write_bytes(b"not a pdf at all")
    with pytest.raises(ValueError, match="not a valid or readable PDF"):
        list(ingest.iter_document_chunks(path, "fake.pdf"))


def test_retry_wait_never_returns_infinity():
    wait = rate_limit.get_retry_wait("try again in " + "9" * 40 + "h")
    assert wait == wait and wait < float("inf")
