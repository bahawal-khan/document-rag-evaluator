from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from core import ingest, retriever, workspace


class _FakeRetriever:
    def __init__(self, chroma_dir=None, **kwargs):
        self.chroma_dir = chroma_dir
        self._api_key = kwargs.get("api_key", "")
        self.method = kwargs.get("method", "hybrid")
        self.last_method_errors = {}

    def close(self):
        self._api_key = ""

    def is_ready(self):
        return False


def _patch_streamlit_dependencies(monkeypatch, tmp_path, collection=None):
    monkeypatch.setattr(workspace, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workspace, "_last_sweep_monotonic", 0.0)
    monkeypatch.setattr(retriever, "Retriever", _FakeRetriever)
    monkeypatch.setattr(
        ingest,
        "get_existing_collection",
        lambda **kwargs: collection,
    )


def test_main_chat_page_loads_without_provider_credentials(monkeypatch, tmp_path):
    _patch_streamlit_dependencies(monkeypatch, tmp_path)

    app = AppTest.from_file("../app.py", default_timeout=30).run()

    assert not app.exception
    assert any("Document RAG Assistant" in item.value for item in app.markdown)


def test_evaluation_page_loads_without_running_provider_calls(monkeypatch, tmp_path):
    collection = SimpleNamespace(count=lambda: 0)
    _patch_streamlit_dependencies(monkeypatch, tmp_path, collection=collection)

    app = AppTest.from_file("../pages/1_Evaluation_Suite.py", default_timeout=30).run()

    assert not app.exception
    assert any("RAG Evaluation Suite" in item.value for item in app.markdown)
