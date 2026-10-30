"""Persistent lexical search index used by the hybrid retriever.

SQLite FTS5 is used when available. A parameterized LIKE-based lexical search
keeps keyword retrieval available on Python builds without the FTS5 extension.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "according", "be", "by",
    "can", "does", "do", "for", "from", "how", "in", "is", "it", "me", "of",
    "on", "or", "please", "tell", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with",
}


class LexicalIndex:
    """A small SQLite FTS index colocated with a workspace's Chroma index."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fts_available = True
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER,
                    location TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS chunks_source_page ON chunks(source, page, file_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS chunks_location ON chunks(location)"
            )
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                    "USING fts5(text, content='chunks', content_rowid='rowid', "
                    "tokenize='unicode61 remove_diacritics 2')"
                )
                connection.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
                    AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
                    AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text)
                        VALUES ('delete', old.rowid, old.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_update
                    AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, text)
                        VALUES ('delete', old.rowid, old.text);
                        INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                    END;
                    """
                )
            except sqlite3.OperationalError as exc:
                if "fts5" not in str(exc).lower() and "no such module" not in str(exc).lower():
                    raise
                self.fts_available = False
                logger.warning("SQLite FTS5 is unavailable; using SQLite keyword matching.")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _tokens(query: str) -> list[str]:
        tokens = []
        seen = set()
        for token in _TOKEN_RE.findall((query or "").casefold()):
            if (len(token) < 2 and not token.isdigit()) or token in _STOP_WORDS or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) == 24:
                break
        return tokens

    def upsert_batch(self, chunk_ids: list[str], chunks: list[dict], file_id: str) -> None:
        rows = [
            (
                chunk_id,
                str(chunk.get("file_id") or file_id or ""),
                str(chunk.get("text") or ""),
                str(chunk.get("source") or ""),
                chunk.get("page"),
                str(chunk.get("location") or ""),
            )
            for chunk_id, chunk in zip(chunk_ids, chunks)
        ]
        if not rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks(chunk_id, file_id, text, source, page, location)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    file_id=excluded.file_id,
                    text=excluded.text,
                    source=excluded.source,
                    page=excluded.page,
                    location=excluded.location
                """,
                rows,
            )

    def delete_stale_source_versions(self, source: str, current_file_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM chunks WHERE source = ? AND file_id != ?",
                (source, current_file_id),
            )

    def delete_chunk_ids(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as connection:
            for start in range(0, len(chunk_ids), 500):
                batch = chunk_ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                connection.execute(
                    f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
                    batch,
                )

    def indexed_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_matching(
        self,
        *,
        source: str | None = None,
        page: int | str | None = None,
        location: str | None = None,
        chunk_ids: list[str] | None = None,
        file_id: str | None = None,
    ) -> int:
        """Count indexed chunks matching an explicit ground-truth reference."""
        clauses = []
        parameters = []
        if chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            clauses.append(f"chunk_id IN ({placeholders})")
            parameters.extend(str(value) for value in chunk_ids)
        if source:
            source_clause = "source = ?"
            source_parameters = [str(source)]
            if page is not None:
                source_clause += " AND page = ?"
                source_parameters.append(page)
            if file_id:
                source_clause += " AND file_id = ?"
                source_parameters.append(str(file_id))
            clauses.append(f"({source_clause})")
            parameters.extend(source_parameters)
        if location:
            location_clause = "location = ?"
            location_parameters = [str(location)]
            if source:
                location_clause += " AND source = ?"
                location_parameters.append(str(source))
            if file_id:
                location_clause += " AND file_id = ?"
                location_parameters.append(str(file_id))
            clauses.append(f"({location_clause})")
            parameters.extend(location_parameters)
        if file_id and not (source or location):
            clauses.append("file_id = ?")
            parameters.append(str(file_id))
        if not clauses:
            return 0
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM chunks WHERE {' OR '.join(clauses)}",
                parameters,
            ).fetchone()
            return int(row[0])

    def ensure_synced(self, collection) -> None:
        """Backfill or repair the sidecar index for Chroma collections created earlier."""
        vector_count = int(collection.count())
        if self.indexed_count() == vector_count:
            return
        logger.info(
            "Rebuilding lexical index to match vector index (%s vs %s chunks).",
            self.indexed_count(),
            vector_count,
        )
        self.rebuild_from_collection(collection)

    def rebuild_from_collection(self, collection, batch_size: int = 500) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
        total = int(collection.count())
        offset = 0
        while offset < total:
            page = collection.get(
                limit=min(batch_size, total - offset),
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = page.get("ids") or []
            documents = page.get("documents") or []
            metadatas = page.get("metadatas") or []
            if not ids:
                break
            rows = []
            for chunk_id, text, metadata in zip(ids, documents, metadatas):
                metadata = metadata or {}
                rows.append(
                    {
                        "chunk_id": str(chunk_id),
                        "file_id": str(metadata.get("file_id") or ""),
                        "text": str(text or ""),
                        "source": str(metadata.get("source") or ""),
                        "page": metadata.get("page"),
                        "location": str(metadata.get("location") or ""),
                    }
                )
            self.upsert_batch(
                [row["chunk_id"] for row in rows],
                rows,
                "",
            )
            offset += len(ids)

    def query(self, query: str, limit: int) -> list[dict]:
        tokens = self._tokens(query)
        if not tokens or limit <= 0:
            return []
        with self._connect() as connection:
            if self.fts_available:
                match_query = " OR ".join(f'"{token}"' for token in tokens)
                try:
                    rows = connection.execute(
                        """
                        SELECT c.chunk_id, c.text, c.source, c.page, c.location,
                               bm25(chunks_fts) AS lexical_score
                        FROM chunks_fts
                        JOIN chunks c ON c.rowid = chunks_fts.rowid
                        WHERE chunks_fts MATCH ?
                        ORDER BY lexical_score ASC
                        LIMIT ?
                        """,
                        (match_query, int(limit)),
                    ).fetchall()
                except sqlite3.OperationalError:
                    logger.warning("FTS query failed; falling back to SQLite keyword matching.")
                    rows = self._like_query(connection, tokens, limit)
            else:
                rows = self._like_query(connection, tokens, limit)
        return [dict(row) for row in rows]

    @staticmethod
    def _like_query(connection, tokens: list[str], limit: int):
        conditions = " OR ".join("lower(text) LIKE lower(?)" for _ in tokens)
        score = " + ".join(
            "CASE WHEN lower(text) LIKE lower(?) THEN 1 ELSE 0 END"
            for _ in tokens
        )
        patterns = [f"%{token}%" for token in tokens]
        return connection.execute(
            f"""
            SELECT chunk_id, text, source, page, location, ({score}) AS lexical_score
            FROM chunks
            WHERE {conditions}
            ORDER BY lexical_score DESC, length(text) ASC
            LIMIT ?
            """,
            (*patterns, *patterns, int(limit)),
        ).fetchall()
