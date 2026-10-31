import pytest

from core import ingest


def _write(tmp_path, text="Alpha beta gamma. " * 400):
    path = tmp_path / "doc.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_larger_chunk_size_gives_fewer_longer_chunks(tmp_path):
    path = _write(tmp_path)
    small = list(ingest.iter_document_chunks(path, "doc.txt", 200, 20))
    large = list(ingest.iter_document_chunks(path, "doc.txt", 1000, 100))
    assert len(large) < len(small)
    assert max(len(c["text"]) for c in large) > max(len(c["text"]) for c in small)
    assert max(len(c["text"]) for c in small) <= 200


@pytest.mark.parametrize("size,overlap", [(50, 0), (5000, 0), (300, 200), (300, -1), ("x", 10)])
def test_invalid_chunk_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        ingest.normalize_chunk_settings(size, overlap)


def test_default_chunk_settings_use_configured_values():
    assert ingest.normalize_chunk_settings() == (ingest.CHUNK_SIZE, ingest.CHUNK_OVERLAP)


def test_reingesting_with_new_chunking_replaces_old_chunks(tmp_path):
    import chromadb  # noqa: F401  (skip cleanly if unavailable)

    path = _write(tmp_path)
    chroma = tmp_path / "chroma"

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    import core.ingest as module

    original = module.build_embedding_backend
    module.build_embedding_backend = lambda *a, **k: FakeEmbeddings()
    try:
        collection, first = module.build_vectorstore(
            [("doc.txt", path)], chroma_dir=chroma, chunk_size=200, chunk_overlap=20
        )
        count_small = collection.count()
        collection, second = module.build_vectorstore(
            [("doc.txt", path)], chroma_dir=chroma, chunk_size=1000, chunk_overlap=100
        )
        count_after = collection.count()
    finally:
        module.build_embedding_backend = original
    assert count_small == first
    assert count_after == second
    assert count_after < count_small  # old small chunks were removed, not mixed in
