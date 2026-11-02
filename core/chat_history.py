"""Optional saved chat history for signed-in users (PostgreSQL, e.g. Supabase).

History is switched on by a database URL, either in the ``CHAT_HISTORY_DATABASE_URL``
environment variable or in the Streamlit secrets::

    [chat_history]
    database_url = "postgresql://..."

Without it (or without ``psycopg``) every function here quietly does nothing, so the
chat itself never depends on the database. Only the question/answer text and the
small "meta" line are stored, never API keys. Chats are keyed by a hash of the
signed-in email, so the table does not contain addresses.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid

logger = logging.getLogger(__name__)

MAX_CHATS_PER_USER = int(os.getenv("MAX_SAVED_CHATS_PER_USER", "50") or 50)
MAX_MESSAGES_PER_CHAT = int(os.getenv("MAX_MESSAGES_PER_CHAT", "400") or 400)
MAX_MESSAGE_CHARS = 20000
TITLE_CHARS = 60
CONNECT_TIMEOUT_SECONDS = 8
RETRY_AFTER_FAILURE_SECONDS = 60

_SCHEMA = """
create table if not exists rag_chats (
    id text primary key,
    user_key text not null,
    title text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists rag_chats_user_idx on rag_chats (user_key, updated_at desc);
create table if not exists rag_chat_messages (
    id bigserial primary key,
    chat_id text not null references rag_chats(id) on delete cascade,
    role text not null,
    content text not null,
    meta text,
    created_at timestamptz not null default now()
);
create index if not exists rag_chat_messages_chat_idx on rag_chat_messages (chat_id, id);
alter table rag_chats enable row level security;
alter table rag_chat_messages enable row level security;
"""

_lock = threading.Lock()
_schema_ready_for: set[str] = set()
_failed_until: dict[str, float] = {}
last_error: str | None = None


def database_url() -> str:
    """The configured database URL, or an empty string."""
    url = (os.getenv("CHAT_HISTORY_DATABASE_URL") or "").strip()
    if url:
        return url
    try:
        import streamlit as st

        section = st.secrets.get("chat_history")
        if section:
            return str(section.get("database_url", "") or "").strip()
    except Exception:
        pass
    return ""


def is_enabled() -> bool:
    """True when a database URL is set and the driver is installed."""
    if not database_url():
        return False
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def user_key(email: str | None) -> str | None:
    """Stable pseudonymous key for a signed-in email (None when not signed in)."""
    normalized = (email or "").strip().lower()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_title(text: str) -> str:
    title = " ".join((text or "").split())
    if len(title) > TITLE_CHARS:
        title = title[: TITLE_CHARS - 1].rstrip() + "…"
    return title or "New chat"


def _connect(url: str):
    import psycopg

    return psycopg.connect(
        url,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        autocommit=True,
        prepare_threshold=None,  # poolers do not like server-side prepared statements
    )


def _record_failure(url: str, exc: Exception) -> None:
    global last_error
    last_error = type(exc).__name__
    _failed_until[url] = time.monotonic() + RETRY_AFTER_FAILURE_SECONDS
    logger.warning("Chat history unavailable (%s).", type(exc).__name__)


def _run(work, fallback):
    """Run ``work(connection)`` and return its result, or ``fallback`` on any failure."""
    url = database_url()
    if not url or _failed_until.get(url, 0.0) > time.monotonic():
        return fallback
    try:
        with _connect(url) as connection:
            with _lock:
                needs_schema = url not in _schema_ready_for
            if needs_schema:
                connection.execute(_SCHEMA)
                with _lock:
                    _schema_ready_for.add(url)
            return work(connection)
    except Exception as exc:  # network, auth, driver: history must never break the chat
        _record_failure(url, exc)
        return fallback


def list_chats(key: str | None) -> list[dict]:
    if not key:
        return []

    def work(connection):
        rows = connection.execute(
            "select id, title, updated_at from rag_chats where user_key = %s "
            "order by updated_at desc limit %s",
            (key, MAX_CHATS_PER_USER),
        ).fetchall()
        return [{"id": row[0], "title": row[1], "updated_at": row[2]} for row in rows]

    return _run(work, [])


def create_chat(key: str | None, title: str) -> str | None:
    if not key:
        return None
    chat_id = uuid.uuid4().hex

    def work(connection):
        connection.execute(
            "insert into rag_chats (id, user_key, title) values (%s, %s, %s)",
            (chat_id, key, make_title(title)),
        )
        # keep only the newest chats per user
        connection.execute(
            "delete from rag_chats where user_key = %s and id not in ("
            "select id from rag_chats where user_key = %s order by updated_at desc limit %s)",
            (key, key, MAX_CHATS_PER_USER),
        )
        return chat_id

    return _run(work, None)


def add_message(key: str | None, chat_id: str | None, role: str, content: str, meta: str | None = None) -> bool:
    """Append a message to a chat that belongs to ``key``. Returns True when saved."""
    if not key or not chat_id or role not in {"user", "assistant"}:
        return False
    content = (content or "")[:MAX_MESSAGE_CHARS]
    meta = (meta or "")[:MAX_MESSAGE_CHARS] or None

    def work(connection):
        cursor = connection.execute(
            "insert into rag_chat_messages (chat_id, role, content, meta) "
            "select %s, %s, %s, %s where exists "
            "(select 1 from rag_chats where id = %s and user_key = %s) "
            "and (select count(*) from rag_chat_messages where chat_id = %s) < %s",
            (chat_id, role, content, meta, chat_id, key, chat_id, MAX_MESSAGES_PER_CHAT),
        )
        if cursor.rowcount:
            connection.execute("update rag_chats set updated_at = now() where id = %s", (chat_id,))
        return bool(cursor.rowcount)

    return bool(_run(work, False))


def load_messages(key: str | None, chat_id: str | None) -> list[dict] | None:
    """Messages of one chat in order, or None if it is missing, not yours, or the database is down."""
    if not key or not chat_id:
        return None

    def work(connection):
        owned = connection.execute(
            "select 1 from rag_chats where id = %s and user_key = %s", (chat_id, key)
        ).fetchone()
        if not owned:
            return None
        rows = connection.execute(
            "select role, content, meta from rag_chat_messages where chat_id = %s order by id",
            (chat_id,),
        ).fetchall()
        messages = []
        for role, content, meta in rows:
            message = {"role": role, "content": content}
            if meta:
                message["meta"] = meta
            messages.append(message)
        return messages

    return _run(work, None)


def delete_chat(key: str | None, chat_id: str | None) -> bool:
    if not key or not chat_id:
        return False

    def work(connection):
        cursor = connection.execute(
            "delete from rag_chats where id = %s and user_key = %s", (chat_id, key)
        )
        return bool(cursor.rowcount)

    return bool(_run(work, False))


def delete_all_chats(key: str | None) -> int:
    if not key:
        return 0

    def work(connection):
        return connection.execute("delete from rag_chats where user_key = %s", (key,)).rowcount

    return int(_run(work, 0) or 0)
