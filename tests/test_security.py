"""Attacker-style regression tests: each one failed before the matching fix."""
import io
import threading
import time
import uuid

import pytest

from core import embeddings, ingest, workspace


@pytest.mark.parametrize("model", ["/etc", "../../root/.ssh", "C:\\Windows\\System32", "meta-llama/Llama-3.1-405B", "~/x"])
def test_local_embedding_model_cannot_be_an_arbitrary_repo_or_path(model, monkeypatch):
    monkeypatch.delenv("ALLOW_CUSTOM_LOCAL_MODELS", raising=False)
    with pytest.raises(ValueError, match="built-in local embedding models"):
        embeddings.normalize_embedding_settings("local", model)


def test_builtin_and_operator_allowlisted_models_still_work(monkeypatch):
    assert embeddings.normalize_embedding_settings("local", "all-MiniLM-L6-v2")["model"] == "all-MiniLM-L6-v2"
    monkeypatch.setenv("LOCAL_EMBEDDING_ALLOWLIST", "my-org/vetted-model")
    assert embeddings.normalize_embedding_settings("local", "my-org/vetted-model")["model"] == "my-org/vetted-model"
    monkeypatch.setenv("ALLOW_CUSTOM_LOCAL_MODELS", "1")
    assert embeddings.normalize_embedding_settings("local", "anything/else")["model"] == "anything/else"


def test_hosted_embedding_providers_keep_custom_model_ids():
    assert embeddings.normalize_embedding_settings("openai", "text-embedding-new", "k")["model"] == "text-embedding-new"


def test_workspace_flood_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workspace, "MAX_WORKSPACES", 5)
    monkeypatch.setattr(workspace, "_last_sweep_monotonic", time.monotonic())
    for _ in range(5):
        workspace.get_workspace_paths(uuid.uuid4().hex)
    with pytest.raises(workspace.WorkspaceLimitError):
        workspace.get_workspace_paths(uuid.uuid4().hex)
    existing = next((tmp_path / "workspaces").iterdir()).name
    assert workspace.get_workspace_paths(existing)["root"].exists()  # real users keep working


def test_oversized_index_blocks_more_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "MAX_INDEX_BYTES", 1000)
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    (chroma / "big.bin").write_bytes(b"x" * 5000)
    with pytest.raises(ValueError, match="larger than"):
        ingest.build_vectorstore([("a.txt", tmp_path / "a.txt")], chroma_dir=chroma)


def test_indexing_concurrency_is_gated(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "_INGEST_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(ingest, "INGEST_WAIT_SECONDS", 0.2)
    running = {"now": 0, "max": 0}
    guard = threading.Lock()

    def fake_build(*args, **kwargs):
        with guard:
            running["now"] += 1
            running["max"] = max(running["max"], running["now"])
        time.sleep(0.5)
        with guard:
            running["now"] -= 1
        return "ok"

    monkeypatch.setattr(ingest, "_build_vectorstore", fake_build)
    results = []

    def attempt():
        try:
            results.append(ingest.build_vectorstore([]))
        except ValueError as exc:
            results.append(str(exc))

    threads = [threading.Thread(target=attempt) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert running["max"] == 1
    assert sum("busy" in str(r) for r in results) == 2


def test_pdf_report_escapes_hostile_titles_and_metric_names():
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate

    from eval import report_pdf

    evil = "<b>open <font color='red'> & <img src='http://evil/x.png'/> </para>"
    summary = {evil: {"average": 0.1, "pass_rate": 0, "passed": 0, "failed": 1, "total": 1,
                      "failure_reasons": [{"input": evil, "score": 0.1, "reason": evil}]}}
    styles = getSampleStyleSheet()
    story = report_pdf._failure_reasons_section({evil: summary}, styles) + report_pdf._metric_table(summary, evil, styles)
    SimpleDocTemplate(io.BytesIO()).build(story)


def test_docx_entity_expansion_bomb_is_rejected(tmp_path):
    import zipfile

    bomb = (
        '<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
        '<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>&c;</w:t></w:r></w:p></w:body></w:document>'
    )
    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", bomb)
    with pytest.raises(Exception, match="(?i)entit|forbidden|DTD"):
        list(ingest.iter_document_chunks(path, "bomb.docx"))


def test_chat_throttle_blocks_a_flooding_session_but_not_others(monkeypatch):
    from core import abuse

    monkeypatch.setattr(abuse, "MAX_CHAT_PER_MINUTE", 3)
    abuse._recent.clear()
    assert [abuse.chat_allowed("a" * 32, now=100.0 + i)[0] for i in range(3)] == [True, True, True]
    allowed, wait = abuse.chat_allowed("a" * 32, now=104.0)
    assert not allowed and 1 <= wait <= 60
    assert abuse.chat_allowed("b" * 32, now=104.0)[0]  # a different user is unaffected
    assert abuse.chat_allowed("a" * 32, now=170.0)[0]  # the window slides


def test_empty_sessions_expire_quickly_but_sessions_with_data_do_not(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workspace, "_last_sweep_monotonic", 0.0)
    empty, used = uuid.uuid4().hex, uuid.uuid4().hex
    workspace.get_workspace_paths(empty)
    paths = workspace.get_workspace_paths(used)
    paths["chroma"].mkdir()
    removed = workspace.sweep_expired_workspaces(force=True, now=time.time() + 3 * 3600)
    assert removed == 1
    assert not (tmp_path / "workspaces" / empty).exists()
    assert (tmp_path / "workspaces" / used).exists()


def test_workspace_cap_holds_under_a_simultaneous_burst(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workspace, "MAX_WORKSPACES", 10)
    monkeypatch.setattr(workspace, "_last_sweep_monotonic", time.monotonic())
    outcomes = []

    def visit():
        try:
            workspace.get_workspace_paths(uuid.uuid4().hex)
            outcomes.append("ok")
        except workspace.WorkspaceLimitError:
            outcomes.append("limit")

    threads = [threading.Thread(target=visit) for _ in range(60)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(list((tmp_path / "workspaces").iterdir())) == 10
    assert outcomes.count("ok") == 10


def test_chroma_client_cache_is_bounded_and_closes_what_it_evicts(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_CLIENT_CACHE_SIZE", 3)
    monkeypatch.setattr(ingest, "_CLIENT_CLOSE_DELAY_SECONDS", 0.05)
    ingest._persistent_client.cache_clear()
    closed = []
    real_close = ingest._close_client
    monkeypatch.setattr(ingest, "_close_client", lambda client: (closed.append(1), real_close(client)))
    for i in range(8):
        ingest._persistent_client(str((tmp_path / f"w{i}").resolve()))
    time.sleep(0.5)
    assert len(ingest._clients) == 3
    assert len(closed) == 5
    ingest._persistent_client.cache_clear()


def test_visitor_without_documents_does_not_start_a_database_runtime(tmp_path):
    before = len(ingest._clients)
    assert ingest.get_existing_collection(chroma_dir=tmp_path / "chroma") is None
    assert len(ingest._clients) == before
    assert not (tmp_path / "chroma").exists()
