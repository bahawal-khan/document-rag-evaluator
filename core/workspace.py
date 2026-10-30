"""Session-scoped paths with inactivity-based cleanup for local data."""

from pathlib import Path
import logging
import os
import re
import shutil
import threading
import time
from uuid import uuid4

import streamlit as st

from core.config import DATA_DIR, env_float

logger = logging.getLogger(__name__)
WORKSPACE_TTL_DAYS = env_float(
    "WORKSPACE_TTL_DAYS", 7.0, minimum=1.0, maximum=3650.0
)
_SWEEP_INTERVAL_SECONDS = 5 * 60
EMPTY_WORKSPACE_TTL_MINUTES = env_float("EMPTY_WORKSPACE_TTL_MINUTES", 60.0, minimum=1.0, maximum=1440.0)
_sweep_lock = threading.Lock()
_last_sweep_monotonic = 0.0


def get_workspace_id() -> str:
    if "workspace_id" not in st.session_state:
        st.session_state.workspace_id = uuid4().hex
    workspace_id = st.session_state.workspace_id
    if not isinstance(workspace_id, str) or not re.fullmatch(r"[0-9a-f]{32}", workspace_id):
        raise ValueError("Workspace ID is invalid.")
    return workspace_id


def _sweep_old_records(now: float, cutoff: float) -> None:
    """Expire completed job records using the same retention window as workspaces."""
    job_dir = DATA_DIR / "evaluation_jobs"
    if not job_dir.exists():
        return
    for record in job_dir.iterdir():
        try:
            if record.is_file() and record.stat().st_mtime < cutoff:
                record.unlink()
        except OSError as exc:
            logger.warning("Could not expire an old evaluation record (%s).", type(exc).__name__)


def sweep_expired_workspaces(*, force: bool = False, now: float | None = None) -> int:
    """Delete inactive workspaces older than ``WORKSPACE_TTL_DAYS``.

    Activity is refreshed whenever a Streamlit page accesses its workspace.
    The scan is throttled so normal reruns do not repeatedly walk the data tree.
    """
    global _last_sweep_monotonic
    current = now if now is not None else time.time()
    monotonic_now = time.monotonic()
    with _sweep_lock:
        if not force and monotonic_now - _last_sweep_monotonic < _SWEEP_INTERVAL_SECONDS:
            return 0
        _last_sweep_monotonic = monotonic_now

    cutoff = current - WORKSPACE_TTL_DAYS * 24 * 60 * 60
    workspaces_dir = DATA_DIR / "workspaces"
    removed = 0
    if workspaces_dir.exists():
        for root in workspaces_dir.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}", root.name) or root.is_symlink():
                continue
            try:
                marker = root / ".last_activity"
                modified = marker.stat().st_mtime if marker.exists() else root.stat().st_mtime
                has_data = (root / "chroma").exists() or (root / "uploads").exists() or (root / "golden_dataset.json").exists()
                # Sessions that never indexed anything (bots, bounced visitors) expire quickly,
                # so a flood of empty sessions frees its slots within the hour.
                if modified < cutoff or (not has_data and modified < current - EMPTY_WORKSPACE_TTL_MINUTES * 60):
                    clear_workspace({"root": root})
                    removed += 1
            except (OSError, ValueError) as exc:
                logger.warning("Could not expire a workspace (%s).", type(exc).__name__)
    _sweep_old_records(current, cutoff)
    return removed


_create_lock = threading.Lock()
MAX_WORKSPACES = int(os.getenv("MAX_WORKSPACES", "300") or 300)


class WorkspaceLimitError(ValueError):
    """Too many active workspaces for this server."""


def get_workspace_paths(workspace_id: str | None = None) -> dict[str, Path]:
    workspace_id = workspace_id or get_workspace_id()
    if not isinstance(workspace_id, str) or not re.fullmatch(r"[0-9a-f]{32}", workspace_id):
        raise ValueError("Workspace ID is invalid.")

    sweep_expired_workspaces()
    root = DATA_DIR / "workspaces" / workspace_id
    with _create_lock:  # check-and-create must be atomic, or a burst overshoots the cap
        if not root.exists():
            # A bot that opens thousands of sessions must not fill the disk with empty workspaces.
            sweep_expired_workspaces()
            existing = DATA_DIR / "workspaces"
            count = sum(1 for _ in existing.iterdir()) if existing.exists() else 0
            if count >= MAX_WORKSPACES:
                raise WorkspaceLimitError(
                    "The server has reached its limit of active sessions. Please try again later."
                )
        root.mkdir(parents=True, exist_ok=True)
    (root / ".last_activity").touch()
    return {
        "root": root,
        "chroma": root / "chroma",
        "golden_dataset": root / "golden_dataset.json",
        "uploads": root / "uploads",
        "cache": root / "answer_cache.json",
        "traces": root / "traces.jsonl",
    }


def clear_workspace(paths: dict[str, Path]) -> None:
    workspace_root = paths["root"].resolve()
    workspace_parent = (DATA_DIR / "workspaces").resolve()
    if workspace_root.parent != workspace_parent or not re.fullmatch(
        r"[0-9a-f]{32}", workspace_root.name
    ):
        raise ValueError("Refusing to delete a path outside the current workspace.")
    try:
        from core.ingest import release_chroma_client

        release_chroma_client(workspace_root / "chroma")
    except Exception:
        logger.debug("Could not release the Chroma client before deleting a workspace.", exc_info=True)
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
