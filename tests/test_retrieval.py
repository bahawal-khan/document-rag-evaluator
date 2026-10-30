import math
import os
from types import SimpleNamespace

import pytest

from core import embeddings, evaluation_jobs, provider, retriever
from core.golden import validate_golden_dataset
from eval.retrieval_eval import score_retrieval


class _FakeCollection:
    def __init__(self):
        self.query_calls = 0
        self.rows = [
            ("chunk-a", "alpha text", {"source": "paper.pdf", "page": 1, "location": "Page 1"}, 0.1),
            ("chunk-b", "beta text", {"source": "paper.pdf", "page": 2, "location": "Page 2"}, 0.2),
            ("chunk-c", "gamma text", {"source": "other.pdf", "page": 1, "location": "Page 1"}, 0.3),
        ]

    def count(self):
        return len(self.rows)

    def query(self, *, query_embeddings, n_results, include):
        self.query_calls += 1
        candidates = self.rows[:n_results]
        return {
            "ids": [[row[0] for row in candidates] for _ in query_embeddings],
            "documents": [[row[1] for row in candidates] for _ in query_embeddings],
            "metadatas": [[row[2] for row in candidates] for _ in query_embeddings],
            "distances": [[row[3] for row in candidates] for _ in query_embeddings],
        }


class _FakeLexicalIndex:
    def __init__(self, *_args):
        pass

    def ensure_synced(self, _collection):
        pass

    def query(self, _query, _limit):
        return [
            {
                "chunk_id": "chunk-b",
                "text": "beta text",
                "source": "paper.pdf",
                "page": 2,
                "location": "Page 2",
                "lexical_score": -1.0,
            },
            {
                "chunk_id": "chunk-c",
                "text": "gamma text",
                "source": "other.pdf",
                "page": 1,
                "location": "Page 1",
                "lexical_score": -0.5,
            },
        ]

    def count_matching(self, **_kwargs):
        return 2


class _FakeEmbeddings:
    def __init__(self):
        self.query_batches = []

    def embed_queries(self, texts):
        self.query_batches.append(list(texts))
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _fake_retriever(monkeypatch, *, use_reranker=False):
    collection = _FakeCollection()
    lexical = _FakeLexicalIndex()
    backend = _FakeEmbeddings()
    monkeypatch.setattr(retriever, "get_existing_collection", lambda **_kwargs: collection)
    monkeypatch.setattr(retriever, "LexicalIndex", lambda *_args: lexical)
    monkeypatch.setattr(retriever, "build_embedding_backend", lambda *_args: backend)
    instance = retriever.Retriever(
        chroma_dir="unused",
        method="hybrid",
        api_key="session-only-key",
        use_reranker=use_reranker,
        top_k=3,
        candidate_k=3,
    )
    return instance, collection, lexical, backend


def test_retriever_registry_supports_dense_bm25_and_hybrid(monkeypatch):
    instance, collection, _lexical, backend = _fake_retriever(monkeypatch)

    result = instance.retrieve_many(
        ["same question"],
        methods=["dense", "bm25", "hybrid"],
    )

    assert set(result) == {"dense", "bm25", "hybrid"}
    assert [item["chunk_id"] for item in result["dense"][0]] == [
        "chunk-a", "chunk-b", "chunk-c"
    ]
    assert [item["chunk_id"] for item in result["bm25"][0]] == [
        "chunk-b", "chunk-c"
    ]
    assert result["hybrid"][0][0]["chunk_id"] == "chunk-b"
    assert collection.query_calls == 1
    assert backend.query_batches == [["same question"]]


def test_retriever_rejects_unknown_method_and_bounds_query_batches(monkeypatch):
    instance, _collection, _lexical, _backend = _fake_retriever(monkeypatch)

    with pytest.raises(ValueError, match="Unknown retriever"):
        instance.retrieve_many(["question"], methods=["made_up"])
    with pytest.raises(ValueError, match="batches"):
        instance.retrieve_many(["question"], batch_size=33)


def test_retrieval_metrics_cover_reference_match_and_empty_results():
    fake_index = SimpleNamespace(count_matching=lambda **_kwargs: 2)
    fake_retriever = SimpleNamespace(lexical_index=fake_index)
    item = {
        "input": "question",
        "expected_output": "answer",
        "source": "paper.pdf",
        "page": 2,
        "source_reference": "Page 2",
    }
    ranked = [
        {"chunk_id": "irrelevant", "source": "other.pdf", "page": 1, "location": "Page 1"},
        {"chunk_id": "match", "source": "paper.pdf", "page": 2, "location": "Page 2"},
        {"chunk_id": "match-2", "source": "paper.pdf", "page": 2, "location": "Page 2"},
    ]

    scores = score_retrieval(item, ranked, retriever=fake_retriever, k=3)

    assert scores["Precision@K"] == pytest.approx(2 / 3)
    assert scores["Recall@K"] == 1.0
    assert scores["MRR"] == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert scores["nDCG@K"] == pytest.approx((1 / math.log2(3) + 1 / 2) / ideal)

    missing = score_retrieval(item, [], retriever=fake_retriever, k=3)
    assert missing["Precision@K"] == 0
    assert missing["Recall@K"] == 0
    assert missing["MRR"] == 0
    assert missing["nDCG@K"] == 0


def test_retrieval_metrics_report_unlabeled_rows_as_not_applicable():
    scores = score_retrieval(
        {"input": "question", "expected_output": "answer"},
        [],
        retriever=SimpleNamespace(lexical_index=None),
        k=4,
    )

    assert scores["label_available"] is False
    assert all(scores[metric] is None for metric in ("Precision@K", "Recall@K", "MRR", "nDCG@K"))


def test_embedding_settings_require_keys_only_for_vector_operations():
    bm25_settings = embeddings.normalize_embedding_settings(
        "openai", "text-embedding-3-small", require_key=False
    )
    assert bm25_settings["api_key"] == ""
    with pytest.raises(ValueError, match="API key"):
        embeddings.normalize_embedding_settings("openai", "text-embedding-3-small")


def test_chat_models_do_not_fall_back_to_server_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-side-key")

    with pytest.raises(ValueError, match="entered.*this session"):
        provider.build_chat_model("gpt-4o-mini", provider="openai")


def test_gemini_adapter_receives_explicit_session_key_without_environment_mutation(monkeypatch):
    import core.google_chat as google_chat

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        google_chat,
        "SessionGoogleChatModel",
        lambda **kwargs: kwargs,
    )

    result = provider.build_chat_model(
        "gemini-embedding-test",
        provider="gemini",
        api_key="session-key-value",
        timeout=12,
    )

    assert result["api_key"] == "session-key-value"
    assert result["timeout"] == 12
    assert os.getenv("GOOGLE_API_KEY") is None
    assert os.getenv("GEMINI_API_KEY") is None


def test_evaluation_job_records_are_workspace_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation_jobs, "DATA_DIR", tmp_path)
    job_id = evaluation_jobs.new_job_id()
    owner = "a" * 32
    evaluation_jobs._write_job(job_id, {"workspace_id": owner, "status": "complete"})

    assert evaluation_jobs.load_evaluation_job(job_id, workspace_id=owner)["status"] == "complete"
    assert evaluation_jobs.load_evaluation_job(job_id, workspace_id="b" * 32) is None


def test_dataset_validation_preserves_retrieval_labels_and_rejects_oversized_questions():
    dataset = validate_golden_dataset([{
        "input": " question ",
        "expected_output": " answer ",
        "source": "paper.pdf",
        "page": 3,
        "source_reference": "Page 3",
        "ignored_private_field": "unused",
    }])

    assert dataset[0]["input"] == "question"
    assert dataset[0]["source_reference"] == "Page 3"
    assert "ignored_private_field" not in dataset[0]
    with pytest.raises(ValueError, match="8,000"):
        validate_golden_dataset([{
            "input": "q" * 8001,
            "expected_output": "answer",
        }])

