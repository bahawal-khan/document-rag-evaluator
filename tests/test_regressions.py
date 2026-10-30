"""Regression tests for the Groq 404, torchvision watcher noise, WinError 10054 and judge bugs."""
import asyncio
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from core import provider, runtime


@pytest.mark.parametrize(
    "configured",
    [
        "https://api.groq.com",
        "https://api.groq.com/",
        "https://api.groq.com/openai/v1",
        "https://api.groq.com/openai/v1/",
        "https://api.groq.com/openai",
    ],
)
def test_groq_base_url_is_normalised_to_the_bare_host(monkeypatch, configured):
    monkeypatch.setenv("GROQ_BASE_URL", configured)
    assert provider.provider_base_url("groq") == "https://api.groq.com"


def test_openai_style_providers_keep_their_v1_suffix(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert provider.provider_base_url("openai") == "https://api.openai.com/v1"


class _FakeGroq(BaseHTTPRequestHandler):
    paths: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).paths.append(self.path)
        if self.path != "/openai/v1/chat/completions":
            body, code = {"error": {"message": "Not Found"}}, 404
        else:
            code = 200
            body = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "pong"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.mark.parametrize("suffix", ["", "/openai/v1"])
def test_real_groq_sdk_reaches_the_chat_completions_route(monkeypatch, suffix):
    """Uses the real langchain-groq/groq client against a route-faithful local server."""
    _FakeGroq.paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGroq)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("GROQ_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}{suffix}")
        model = provider.build_chat_model("openai/gpt-oss-20b", provider="groq", api_key="k", timeout=10)
        assert model.invoke("ping").content == "pong"
    finally:
        server.shutdown()
    assert _FakeGroq.paths == ["/openai/v1/chat/completions"]


def test_model_not_found_is_not_reported_as_a_bad_endpoint():
    from core.generator import format_llm_error

    error = Exception("Error code: 404 - {'error': {'code': 'model_not_found', "
                      "'message': 'The model `x` does not exist or you do not have access to it.'}}")
    message = format_llm_error(error, "groq", model_name="x")
    assert "not available for this API key" in message
    assert "endpoint" not in message.lower()


def test_not_found_message_reports_the_url_actually_called():
    from core.generator import format_llm_error

    error = Exception("Error code: 404 - Not Found")
    error.response = SimpleNamespace(request=SimpleNamespace(url="https://api.groq.com/openai/v1/openai/v1/chat/completions?x=1"))
    message = format_llm_error(error, "groq", model_name="m", base_url="https://api.groq.com")
    assert "https://api.groq.com/openai/v1/openai/v1/chat/completions" in message
    assert "x=1" not in message


def _record(name, exc):
    try:
        raise exc
    except Exception:
        return logging.LogRecord("streamlit.watcher.local_sources_watcher", logging.WARNING, __file__, 1,
                                 "Examining the path of %s raised:", (name,), sys.exc_info())


def test_watcher_filter_drops_only_transformers_missing_dependency_records():
    flt = runtime._OptionalDependencyWatcherFilter()
    missing = ModuleNotFoundError("No module named 'torchvision'")
    assert flt.filter(_record("transformers.models.zoedepth.image_processing_zoedepth_fast", missing)) is False
    assert flt.filter(_record("mypackage.module", missing)) is True
    assert flt.filter(_record("transformers.models.x", RuntimeError("other"))) is True


def test_windows_asyncio_filter_drops_only_the_benign_connection_lost_record():
    flt = runtime._WindowsConnectionLostFilter()
    reset = ConnectionResetError(10054, "forcibly closed")
    reset.winerror = 10054

    def record(message, exc):
        try:
            raise exc
        except Exception:
            return logging.LogRecord("asyncio", logging.ERROR, __file__, 1, message, (), sys.exc_info())

    assert flt.filter(record("Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)", reset)) is False
    assert flt.filter(record("Exception in callback something_else()", reset)) is True


def test_evaluation_config_runs_deepeval_without_an_event_loop():
    from eval.common import eval_async_config

    config = eval_async_config()
    assert config.run_async is False
    assert config.max_concurrent == 1


def test_synchronous_deepeval_run_creates_no_event_loop(monkeypatch):
    """A real DeepEval run with a stub judge must never touch asyncio."""
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase

    from eval.common import evaluate_test_cases

    class StubJudge(DeepEvalBaseLLM):
        def __init__(self):
            self.model_name = "stub"

        def load_model(self):
            return self

        def generate(self, prompt, *args, **kwargs):
            if "statements" in prompt.lower():
                return json.dumps({"statements": ["A statement."]})
            return json.dumps({"verdicts": [{"verdict": "yes", "reason": None}], "reason": "ok"})

        async def a_generate(self, prompt, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("async judge path used")

        def get_model_name(self):
            return "stub"

    calls = []
    real_new_loop = asyncio.new_event_loop
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: calls.append("new_event_loop") or real_new_loop())
    monkeypatch.setattr(asyncio, "run", lambda *a, **k: calls.append("run"))

    case = LLMTestCase(input="q", actual_output="a", expected_output="a", retrieval_context=["ctx"])
    result = evaluate_test_cases([case], [AnswerRelevancyMetric(model=StubJudge(), include_reason=False)])
    assert result.test_results
    assert calls == []


def test_judge_sync_rate_limit_is_retried_not_masked(monkeypatch):
    """The sync path used to raise NameError, hiding 429s from the retry logic."""
    from core import judge as judge_module

    attempts = {"n": 0}

    class FlakyModel:
        temperature = 0

        def invoke(self, prompt, config=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise Exception("Error code: 429 - rate limit reached. Please try again in 1s")
            return SimpleNamespace(content='{"ok": true}')

    monkeypatch.setattr(judge_module, "build_chat_model", lambda *a, **k: FlakyModel())
    monkeypatch.setattr(judge_module, "wait_for_rate_limit", lambda **k: None)
    monkeypatch.setattr(judge_module.time, "sleep", lambda s: None)
    judge = judge_module.LLMJudge("m", provider="groq", api_key="k")
    assert json.loads(judge.generate("prompt")) == {"ok": True}
    assert attempts["n"] == 2


def test_importing_ingest_does_not_load_transformers():
    import subprocess

    code = ("import sys, core.ingest; "
            "print(any(m == 'transformers' or m.startswith('transformers.') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().endswith("False")


def test_golden_diagnostics_are_deduplicated(monkeypatch):
    from core import golden

    class Collection:
        def count(self): return 10
        def get(self, limit, offset, include):
            n = min(limit, 10 - offset)
            return {"documents": ["text " * 20] * n, "metadatas": [{"source": "a.pdf"}] * n}

    class Llm:
        def invoke(self, prompt, config=None):
            raise RuntimeError("Error code: 500 - upstream exploded")

    monkeypatch.setattr(golden, "_build_llm", lambda *a, **k: Llm())
    monkeypatch.setattr(golden, "close_chat_model", lambda m: None)
    monkeypatch.setattr(golden.time, "sleep", lambda s: None)
    _, stats = golden.generate_golden_dataset(Collection(), n=10, api_key="k", model_provider="groq")
    assert stats["produced"] == 0
    assert len(stats["errors"]) == len(set(stats["errors"]))


def test_unclassified_provider_error_keeps_status_and_provider_message():
    from core.generator import format_llm_error

    error = Exception("boom")
    error.status_code = 400
    error.body = {"error": {"code": "invalid_request_error", "message": "Bad schema for key gsk_ABCDEFGH12345678"}}
    message = format_llm_error(error, "groq", model_name="m")
    assert "HTTP 400" in message and "invalid_request_error" in message and "Bad schema" in message
    assert "gsk_ABCDEFGH12345678" not in message
