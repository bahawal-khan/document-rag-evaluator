import json
import os
import sys
import time
import types
import zipfile

import pytest
from reportlab.pdfgen import canvas

from core import cache, cost, golden, ingest, provider, safety_fast, tracer, workspace
from core.generator import format_llm_error


def test_plain_text_is_chunked_with_source_locations(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(("A grounded fact about the system. " * 50), encoding="utf-8")

    chunks = list(ingest.iter_document_chunks(source, "notes.txt"))

    assert len(chunks) > 1
    assert all(chunk["source"] == "notes.txt" for chunk in chunks)
    assert all(chunk["location"].startswith("Chunk ") for chunk in chunks)
    assert all(len(chunk["text"]) <= ingest.CHUNK_SIZE for chunk in chunks)


def test_docx_paragraphs_are_extracted_incrementally(tmp_path):
    source = tmp_path / "notes.docx"
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>"
        + ("A document paragraph with grounded details. " * 12)
        + "</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    chunks = list(ingest.iter_document_chunks(source, "notes.docx"))

    assert chunks
    assert "grounded details" in chunks[0]["text"]
    assert chunks[0]["source"] == "notes.docx"


def test_pdf_chunks_keep_page_references(tmp_path):
    source = tmp_path / "research.pdf"
    pdf = canvas.Canvas(str(source))
    pdf.drawString(50, 750, "Page one contains a grounded research finding.")
    pdf.showPage()
    pdf.drawString(50, 750, "Page two describes a separate grounded result.")
    pdf.save()

    chunks = list(ingest.iter_document_chunks(source, "research.pdf"))

    assert [chunk["page"] for chunk in chunks] == [1, 2]
    assert [chunk["location"] for chunk in chunks] == ["Page 1", "Page 2"]


def test_index_upserts_identical_content_without_duplicate_ids(monkeypatch, tmp_path):
    class FakeCollection:
        def __init__(self):
            self.rows = {}

        def upsert(self, *, ids, documents, metadatas, embeddings=None):
            self.rows.update(
                {
                    row_id: (document, metadata)
                    for row_id, document, metadata in zip(ids, documents, metadatas)
                }
            )

        def get(self, *, where, include):
            file_id = where["file_id"]
            return {
                "ids": [row_id for row_id, row in self.rows.items() if row[1]["file_id"] == file_id]
            }

        def delete(self, *, where=None, ids=None):
            if ids is not None:
                for row_id in ids:
                    self.rows.pop(row_id, None)
                return
            source_filter, file_filter = where["$and"]
            source = source_filter["source"]
            active_file_id = file_filter["file_id"]["$ne"]
            self.rows = {
                row_id: row
                for row_id, row in self.rows.items()
                if row[1]["source"] != source or row[1]["file_id"] == active_file_id
            }

    collection = FakeCollection()
    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(ingest, "_collection", lambda *_args: (None, collection))
    monkeypatch.setattr(ingest, "build_embedding_backend", lambda *_args: FakeEmbeddings())
    source = tmp_path / "document.txt"
    source.write_text("A sufficiently long document fact. " * 40, encoding="utf-8")

    _, first_count = ingest.build_vectorstore([("document.txt", source)], chroma_dir=tmp_path / "index")
    first_ids = set(collection.rows)
    _, second_count = ingest.build_vectorstore([("document.txt", source)], chroma_dir=tmp_path / "index")

    assert first_count == second_count
    assert set(collection.rows) == first_ids

    source.write_text("A replacement document with new facts. " * 40, encoding="utf-8")
    ingest.build_vectorstore([("document.txt", source)], chroma_dir=tmp_path / "index")

    assert not set(collection.rows).intersection(first_ids)
    assert all("replacement" in row[0] for row in collection.rows.values())


def test_failed_document_embedding_rolls_back_new_vector_and_bm25_rows(monkeypatch, tmp_path):
    class FakeCollection:
        def __init__(self):
            self.rows = {}

        def get(self, *, where, include):
            return {"ids": []}

        def upsert(self, *, ids, documents, metadatas, embeddings):
            self.rows.update({
                row_id: (document, metadata)
                for row_id, document, metadata in zip(ids, documents, metadatas)
            })

        def delete(self, *, ids=None, where=None):
            for row_id in ids or []:
                self.rows.pop(row_id, None)

    class FlakyEmbeddings:
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("provider request failed")
            return [[1.0, 0.0] for _ in texts]

    collection = FakeCollection()
    monkeypatch.setattr(ingest, "_collection", lambda *_args: (None, collection))
    monkeypatch.setattr(ingest, "build_embedding_backend", lambda *_args: FlakyEmbeddings())
    monkeypatch.setattr(ingest, "INDEX_BATCH_SIZE", 1)
    source = tmp_path / "document.txt"
    source.write_text("A sufficiently long document fact. " * 40, encoding="utf-8")

    with pytest.raises(RuntimeError, match="provider request failed"):
        ingest.build_vectorstore([("document.txt", source)], chroma_dir=tmp_path / "index")

    assert collection.rows == {}
    lexical = ingest.LexicalIndex(tmp_path / "index" / "hybrid_search.sqlite3")
    assert lexical.indexed_count() == 0


def test_golden_sampling_pages_through_index_without_loading_all_chunks():
    class FakeCollection:
        def __init__(self):
            self.rows = [
                (f"document chunk {index}", {"source": "doc.txt", "page": index})
                for index in range(620)
            ]
            self.page_sizes = []

        def count(self):
            return len(self.rows)

        def get(self, *, limit, offset, include):
            page = self.rows[offset:offset + limit]
            self.page_sizes.append(len(page))
            return {
                "documents": [row[0] for row in page],
                "metadatas": [row[1] for row in page],
            }

    collection = FakeCollection()

    sampled, scanned = golden._sample_collection_chunks(collection, 5, seed=42)

    assert scanned == 620
    assert len(sampled) == 5
    assert max(collection.page_sizes) <= 250


def test_golden_generation_deduplicates_and_checkpoints(monkeypatch, tmp_path):
    chunks = [
        ("Excerpt one with enough text.", {"source": "one.pdf", "page": 1, "location": "Page 1"}),
        ("Excerpt two with enough text.", {"source": "two.pdf", "page": 2, "location": "Page 2"}),
    ]
    monkeypatch.setattr(golden, "_sample_collection_chunks", lambda *_args: (chunks, 2))
    monkeypatch.setattr(golden, "_build_llm", lambda *_args: object())
    monkeypatch.setattr(
        golden,
        "_generate_batch",
        lambda *_args, **_kwargs: [
            {"question": "What is the result?", "answer": "It is successful.", "difficulty": "easy"},
            {"question": "  WHAT is the result? ", "answer": "A duplicate.", "difficulty": "medium"},
        ],
    )
    dataset_path = tmp_path / "golden.json"

    dataset, stats = golden.generate_golden_dataset(
        object(),
        n=2,
        dataset_path=dataset_path,
    )

    assert len(dataset) == 1
    assert stats["duplicate_questions"] == 1
    assert json.loads(dataset_path.read_text(encoding="utf-8")) == dataset


def test_golden_batch_reports_provider_error_without_retrying_connection_failure(monkeypatch):
    from types import SimpleNamespace

    class FailingLLM:
        calls = 0

        def invoke(self, prompt, config=None):
            self.calls += 1
            raise RuntimeError("APIConnectionError: ConnectError: All connection attempts failed")

    llm = FailingLLM()
    errors = []
    monkeypatch.setattr(golden.time, "sleep", lambda *_args: None)

    result = golden._generate_batch(
        llm,
        [("excerpt", {})],
        callbacks=None,
        provider="groq",
        api_key="session-secret",
        model_name="test-model",
        error_callback=errors.append,
    )

    assert result == []
    assert llm.calls == 1
    assert len(errors) == 1
    assert "could not reach" in errors[0].lower()
    assert "session-secret" not in errors[0]


def test_golden_batch_rejects_non_array_json_without_provider_retry(monkeypatch):
    from types import SimpleNamespace

    class InvalidJSONLLM:
        calls = 0

        def invoke(self, prompt, config=None):
            self.calls += 1
            return SimpleNamespace(content='{"question": "not an array"}')

    llm = InvalidJSONLLM()
    errors = []
    monkeypatch.setattr(golden.time, "sleep", lambda *_args: None)

    result = golden._generate_batch(
        llm,
        [("excerpt", {})],
        callbacks=None,
        provider="groq",
        api_key="session-secret",
        model_name="test-model",
        error_callback=errors.append,
    )

    assert result == []
    assert llm.calls == 1
    assert "not a json array" in errors[0].lower()


def test_golden_difficulty_breakdown_tolerates_invalid_optional_metadata():
    counts = golden.difficulty_breakdown([
        {"difficulty": ["not", "a", "level"]},
        {"difficulty": "easy"},
    ])

    assert counts == {"easy": 1, "medium": 1, "hard": 0}


def test_openrouter_provider_receives_user_key_and_compatible_endpoint(monkeypatch):
    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "langchain_openai", module)

    model_args = provider.build_chat_model(
        "openai/gpt-4o-mini",
        provider="openrouter",
        api_key="user-provided-key",
    )

    assert model_args["api_key"] == "user-provided-key"
    assert model_args["base_url"] == "https://openrouter.ai/api/v1"


def test_groq_provider_uses_chat_completions_base_url_without_live_call(monkeypatch):
    module = types.ModuleType("langchain_groq")

    class FakeChatGroq:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.ChatGroq = FakeChatGroq
    monkeypatch.setitem(sys.modules, "langchain_groq", module)
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)

    model = provider.build_chat_model(
        "openai/gpt-oss-20b",
        provider="groq",
        api_key="session-key",
    )

    assert model.kwargs["model"] == "openai/gpt-oss-20b"
    # The Groq SDK appends /openai/v1 itself; a base URL that already ends in
    # /openai/v1 produces /openai/v1/openai/v1/... and a 404.
    assert model.kwargs["base_url"] == "https://api.groq.com"
    assert model.kwargs["max_retries"] == 0
    assert "responses" not in model.kwargs


def test_workspace_paths_are_isolated_and_reject_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "DATA_DIR", tmp_path)

    first = workspace.get_workspace_paths("a" * 32)
    second = workspace.get_workspace_paths("b" * 32)

    assert first["root"] != second["root"]
    assert first["chroma"].parent == first["root"]
    with pytest.raises(ValueError, match="Workspace ID is invalid"):
        workspace.get_workspace_paths("..")


def test_workspace_cleanup_only_deletes_the_resolved_session_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "DATA_DIR", tmp_path)
    paths = workspace.get_workspace_paths("c" * 32)
    marker = paths["uploads"] / "private.txt"
    paths["uploads"].mkdir()
    marker.write_text("private", encoding="utf-8")

    workspace.clear_workspace(paths)

    assert not paths["root"].exists()
    assert marker.parent.parent == paths["root"]


def test_workspace_sweeper_removes_expired_data_and_keeps_recent_sessions(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workspace, "WORKSPACE_TTL_DAYS", 7)
    monkeypatch.setattr(workspace, "_last_sweep_monotonic", 0.0)
    workspaces_dir = tmp_path / "workspaces"
    old_workspace = workspaces_dir / ("d" * 32)
    active_workspace = workspaces_dir / ("e" * 32)
    old_workspace.mkdir(parents=True)
    active_workspace.mkdir(parents=True)
    old_marker = old_workspace / ".last_activity"
    active_marker = active_workspace / ".last_activity"
    old_marker.touch()
    active_marker.touch()
    now = time.time()
    os.utime(old_marker, (now - 8 * 86400, now - 8 * 86400))
    os.utime(active_marker, (now, now))

    removed = workspace.sweep_expired_workspaces(force=True, now=now)

    assert removed == 1
    assert not old_workspace.exists()
    assert active_workspace.exists()


def test_evaluation_job_state_file_round_trips_without_exposing_secrets(monkeypatch, tmp_path):
    from core import evaluation_jobs

    monkeypatch.setattr(evaluation_jobs, "DATA_DIR", tmp_path)
    job_id = evaluation_jobs.new_job_id()
    monkeypatch.setattr(evaluation_jobs, "_live_jobs", {job_id})  # this process owns the job
    evaluation_jobs._write_job(job_id, {
        "job_id": job_id,
        "workspace_id": "f" * 32,
        "status": "running",
        "phase": "Scoring",
        "api_key": "session-secret-value",
        "result": {"nested": {"judge_api_key": "another-secret"}},
    })

    record = evaluation_jobs.load_evaluation_job(job_id)

    assert record["status"] == "running"
    assert "api_key" not in record
    saved = (tmp_path / "evaluation_jobs" / f"{job_id}.json").read_text(encoding="utf-8")
    assert "session-secret-value" not in saved
    assert "another-secret" not in saved
    assert evaluation_jobs.load_evaluation_job("../invalid") is None


def test_answer_cache_is_scoped_to_model_configuration_and_does_not_store_keys(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache.set_cached(
        "question",
        {"answer": "answer"},
        cache_file,
        cache_context="groq:model-a",
    )

    assert cache.get_cached("question", cache_file, cache_context="groq:model-a") == {
        "answer": "answer"
    }
    assert cache.get_cached("question", cache_file, cache_context="openai:model-b") is None
    assert "api-key-value" not in cache_file.read_text(encoding="utf-8")


def test_request_cache_includes_prompt_parameters_and_context(tmp_path):
    from core.cache import get_cached_request, set_cached_request

    cache_file = tmp_path / "requests.json"
    set_cached_request(
        "groq",
        "model-a",
        "prompt",
        {"answer": "answer"},
        parameters={"temperature": 0},
        context=[{"source": "a.pdf", "text": "context"}],
        cache_file=cache_file,
    )

    assert get_cached_request(
        "groq", "model-a", "prompt",
        parameters={"temperature": 0},
        context=[{"source": "a.pdf", "text": "context"}],
        cache_file=cache_file,
    )["answer"] == "answer"
    assert get_cached_request(
        "groq", "model-a", "prompt",
        parameters={"temperature": 1},
        context=[{"source": "a.pdf", "text": "context"}],
        cache_file=cache_file,
    ) is None
    assert get_cached_request(
        "openai", "model-a", "prompt",
        parameters={"temperature": 0},
        context=[{"source": "a.pdf", "text": "context"}],
        cache_file=cache_file,
    ) is None


def test_operational_metrics_count_actual_api_calls_separately_from_cache_hits(tmp_path):
    from core.tracer import compute_operational_metrics, log_trace

    trace_file = tmp_path / "traces.jsonl"
    log_trace({"cache_hit": False, "api_calls": 1, "generation_status": "success",
               "total_latency_s": 0.2, "ttft_s": 0.1}, trace_file)
    log_trace({"cache_hit": True, "api_calls": 0, "cache_latency_s": 0.001}, trace_file)

    metrics = compute_operational_metrics(trace_file)
    assert metrics["total_requests"] == 1
    assert metrics["generation_requests"] == 1
    assert metrics["cache_hits"] == 1


def test_cache_rejects_malformed_and_timed_out_entries(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("[]", encoding="utf-8")

    assert cache.load_cache(cache_file) == {}
    assert not cache.is_usable_cache_entry([])
    assert not cache.is_usable_cache_entry({"answer": ""})
    assert not cache.is_usable_cache_entry({
        "answer": "Partial answer *(response cut off due to the time limit)*"
    })
    assert cache.is_usable_cache_entry({"answer": "Complete answer"})


def test_trace_writer_keeps_only_safe_operational_fields(tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    tracer.log_trace({
        "provider": "openai",
        "retrieval_method": "hybrid",
        "total_latency_s": 1.2,
        "api_key": "session-secret-value",
        "prompt": "private prompt",
        "model": "custom/model",
    }, trace_file)

    saved = trace_file.read_text(encoding="utf-8")
    entry = json.loads(saved)
    assert entry["provider"] == "openai"
    assert entry["retrieval_method"] == "hybrid"
    assert "session-secret-value" not in saved
    assert "private prompt" not in saved
    assert "custom/model" not in saved


def test_operational_reliability_handles_legacy_and_current_trace_statuses(tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    entries = [
        # Legacy uncached successes were mistakenly marked error=True.
        {"total_latency_s": 1.0, "error": True},
        # Legacy provider failures stored their error string.
        {"total_latency_s": 2.0, "error": "provider request failed"},
        {"total_latency_s": 3.0, "error": True, "generation_status": "error"},
        {"total_latency_s": 4.0, "error": True, "generation_status": "timed_out"},
        {"total_latency_s": 5.0, "error": False, "generation_status": "success"},
    ]
    trace_file.write_text(
        "\n".join(json.dumps(entry) for entry in entries),
        encoding="utf-8",
    )

    metrics = tracer.compute_operational_metrics(trace_file)

    assert metrics["error_count"] == 3
    assert metrics["reliability"] == 0.4


def test_cache_hits_are_separate_from_generated_latency_percentiles(tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    entries = [
        {"cache_hit": True, "cache_latency_s": 0.012, "total_latency_s": 0.001, "ttft_s": 0.001},
        {"cache_hit": False, "total_latency_s": 1.5, "ttft_s": 0.2, "generation_status": "success"},
        {"toxicity_blocked": True, "total_latency_s": 0.01, "ttft_s": 0.01},
    ]
    trace_file.write_text(
        "\n".join(json.dumps(entry) for entry in entries),
        encoding="utf-8",
    )

    metrics = tracer.compute_operational_metrics(trace_file)

    assert metrics["generation_requests"] == 1
    assert metrics["p50_latency_s"] == 1.5
    assert metrics["avg_ttft_s"] == 0.2
    assert metrics["avg_cache_latency_s"] == 0.012


def test_cost_estimates_use_selected_model_and_unknown_models_are_unpriced():
    gpt_4o = cost.estimate_cost(1000, 1000, provider="openai", model_name="gpt-4o")
    gpt_oss = cost.estimate_cost(
        1000, 1000, provider="groq", model_name="openai/gpt-oss-20b"
    )

    assert gpt_4o == pytest.approx(0.0125)
    assert gpt_oss == pytest.approx(0.000375)
    assert gpt_4o > gpt_oss
    assert cost.estimate_cost(1000, 1000, provider="openrouter", model_name="custom/model") is None
    assert cost.estimate_cost(1000, 1000, provider="ollama", model_name="qwen2.5:7b") == 0


def test_toxicity_heuristic_avoids_scientific_substring_false_positives():
    assert not safety_fast.check_toxicity("What concentration kills the cell culture?")
    assert not safety_fast.check_toxicity("The sample contained a bactericide.")
    assert safety_fast.check_toxicity("You are an idiot.")


def test_upload_batch_has_per_file_count_and_total_caps():
    assert ingest.MAX_DOCUMENT_SIZE_BYTES == 25 * 1024 * 1024
    assert ingest.MAX_UPLOAD_FILE_COUNT == 3
    assert ingest.MAX_TOTAL_UPLOAD_SIZE_BYTES == 75 * 1024 * 1024


def test_provider_error_redacts_user_key_from_unclassified_errors():
    message = format_llm_error(
        RuntimeError("provider failure including api-key-value"),
        provider="openai",
        api_key="api-key-value",
    )

    assert "api-key-value" not in message
    assert "[redacted]" in message
