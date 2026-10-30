"""Process-level runtime setup shared by every entry point.

This module is deliberately small and has no dependency on the rest of the
application so that it can run before any heavy library is imported.

It provides:

* ``bootstrap()``                – privacy-friendly library defaults, optional
                                   bridging of *non-provider* Streamlit secrets
                                   into the environment, a narrowly scoped
                                   Windows asyncio log filter, and a narrow
                                   Streamlit-watcher filter for transformers'
                                   optional-dependency lazy modules.
* ``evaluation_event_loop()``    – an event loop whose lifetime is owned by one
                                   evaluation job, so pooled HTTP connections can
                                   never outlive the loop that created them.
* ``new_event_loop()``           – Selector loop on Windows, default elsewhere.

Provider API keys are intentionally NOT bridged from secrets or the
environment: the application keeps them in each user's Streamlit session only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

# Defaults only: a value that is already set in the real environment wins.
_ENVIRONMENT_DEFAULTS = {
    "ANONYMIZED_TELEMETRY": "False",          # ChromaDB product telemetry
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",      # DeepEval usage telemetry
    "HF_HUB_DISABLE_TELEMETRY": "1",          # Hugging Face Hub telemetry
    "TOKENIZERS_PARALLELISM": "false",        # avoids fork/thread warnings
}

# Operational settings that may come from Streamlit secrets / the environment.
# Provider API keys are deliberately absent from this list.
_ALLOWED_SECRET_NAMES = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "WORKSPACE_TTL_DAYS",
    "HARD_REQUEST_DEADLINE_SECONDS",
    "LLM_MIN_INTERVAL_SECONDS",
    "GROQ_MIN_INTERVAL_SECONDS",
    "OPENAI_MIN_INTERVAL_SECONDS",
    "GEMINI_MIN_INTERVAL_SECONDS",
    "ANTHROPIC_MIN_INTERVAL_SECONDS",
    "OPENROUTER_MIN_INTERVAL_SECONDS",
    "OLLAMA_MIN_INTERVAL_SECONDS",
    "GROQ_BASE_URL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OLLAMA_BASE_URL",
    "RERANKER_MODEL",
    "RERANKER_SCOPE_THRESHOLD",
    "SCOPE_DISTANCE_THRESHOLD",
)

_bootstrap_lock = threading.Lock()
_bootstrapped = False


class _WindowsConnectionLostFilter(logging.Filter):
    """Drop only asyncio's harmless ``_call_connection_lost`` WinError 10054 record.

    On Windows, asyncio's Proactor transport calls ``socket.shutdown`` while it
    is already closing a connection. When the peer has reset that connection
    first, the call raises ``ConnectionResetError: [WinError 10054]`` and asyncio
    logs it. The socket is being closed anyway, so no data is lost; the record is
    CPython noise. Every other asyncio record, including a reset during an
    active read, still reaches the log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        if not isinstance(error, ConnectionResetError):
            return True
        if getattr(error, "winerror", None) != 10054:
            return True
        return "_call_connection_lost" not in record.getMessage()


def install_windows_asyncio_filter() -> None:
    """Attach the narrow WinError 10054 filter once (Windows only)."""
    if sys.platform != "win32":
        return
    asyncio_logger = logging.getLogger("asyncio")
    if not any(isinstance(item, _WindowsConnectionLostFilter) for item in asyncio_logger.filters):
        asyncio_logger.addFilter(_WindowsConnectionLostFilter())


class _OptionalDependencyWatcherFilter(logging.Filter):
    """Drop Streamlit's watcher warning for transformers' unusable lazy aliases.

    Streamlit's file watcher calls ``hasattr(module, "__path__")`` on every
    entry in ``sys.modules``. Transformers 5.x registers lazy
    ``...image_processing_*_fast`` alias modules (ZoeDepth, Aria, BEiT, ...);
    touching one imports the real module, which needs ``torchvision``. That
    raises ``ModuleNotFoundError`` (not ``AttributeError``), so ``hasattr``
    does not swallow it and Streamlit logs a ~100-line traceback per rerun.

    Such a module has no source file to watch, and this app never uses image
    models, so installing torchvision would only add weight. Only records for
    ``transformers.*`` modules whose error is a missing optional import are
    dropped; every other watcher warning is still logged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        if not isinstance(error, ModuleNotFoundError):
            return True
        args = record.args if isinstance(record.args, tuple) else ()
        module_name = str(args[0]) if args else ""
        return not module_name.startswith("transformers.")


def install_watcher_log_filter() -> None:
    """Attach the optional-dependency filter once (all platforms)."""
    watcher_logger = logging.getLogger("streamlit.watcher.local_sources_watcher")
    if not any(isinstance(item, _OptionalDependencyWatcherFilter) for item in watcher_logger.filters):
        watcher_logger.addFilter(_OptionalDependencyWatcherFilter())


def _apply_streamlit_secrets() -> None:
    """Copy allow-listed secrets into ``os.environ`` without overriding it."""
    try:
        import streamlit as st

        secrets = st.secrets
        for name in _ALLOWED_SECRET_NAMES:
            if name in os.environ:
                continue
            try:
                if name in secrets:
                    os.environ[name] = str(secrets[name])
            except Exception:  # a malformed single entry must not stop the rest
                continue
    except Exception:
        # No secrets file (local runs, tests) or Streamlit not importable.
        return


def bootstrap() -> None:
    """Idempotent process setup. Safe to call from every page and module."""
    global _bootstrapped
    with _bootstrap_lock:
        if _bootstrapped:
            return
        _bootstrapped = True
    for name, value in _ENVIRONMENT_DEFAULTS.items():
        os.environ.setdefault(name, value)
    _apply_streamlit_secrets()
    install_windows_asyncio_filter()
    install_watcher_log_filter()


def new_event_loop() -> asyncio.AbstractEventLoop:
    """Create a loop; use the Selector loop on Windows (no Proactor transports)."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel leftovers and close a loop that is no longer running."""
    try:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        shutdown_executor = getattr(loop, "shutdown_default_executor", None)
        if shutdown_executor is not None:
            loop.run_until_complete(shutdown_executor())
    except Exception:
        logger.debug("Event loop cleanup was incomplete.", exc_info=True)
    finally:
        loop.close()


@contextlib.contextmanager
def evaluation_event_loop():
    """Own the asyncio loop that DeepEval will use on the current thread.

    DeepEval calls ``asyncio.get_event_loop()`` and, when the thread has none,
    creates one that is never closed. On a long-lived worker thread that loop
    outlives every job, so idle keep-alive sockets stay registered on a loop
    that is not running; when the provider later resets them, the cleanup runs
    during an unrelated job and surfaces as ``ConnectionResetError`` (WinError
    10054 on Windows). Giving each job its own loop and closing it afterwards
    removes that lifetime mismatch.
    """
    loop = new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        try:
            _shutdown_loop(loop)
        finally:
            asyncio.set_event_loop(None)
