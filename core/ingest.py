"""Incremental, page-aware document parsing and persistent Chroma indexing."""

import hashlib
from collections import OrderedDict
from functools import lru_cache
import os
import threading
import xml.etree.ElementTree as ET
from defusedxml import ElementTree as DefusedET  # rejects entity-expansion (billion-laughs) payloads
import zipfile
from pathlib import Path

import pdfplumber

from core.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME
from core.embeddings import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    build_embedding_backend,
    close_embedding_backend,
    collection_name_for_embedding,
    lexical_index_name,
    normalize_embedding_settings,
    validate_embeddings,
)
from core.lexical import LexicalIndex

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
INDEX_BATCH_SIZE = 100
MAX_DOCUMENT_SIZE_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_FILE_COUNT = max(1, int(os.getenv("MAX_UPLOAD_FILE_COUNT", "5") or 5))
MAX_TOTAL_UPLOAD_SIZE_BYTES = MAX_DOCUMENT_SIZE_BYTES * MAX_UPLOAD_FILE_COUNT
MAX_DOCX_XML_SIZE_BYTES = 100 * 1024 * 1024
# A 25 MB file of unbroken text would otherwise become ~80,000 chunks (hours of CPU embedding).
MAX_CHUNKS_PER_DOCUMENT = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "20000") or 20000)
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4000
LEGACY_CHUNKING = (300, 50)  # file ids for this setting stay unchanged so existing indexes keep working


def normalize_chunk_settings(chunk_size=None, chunk_overlap=None) -> tuple[int, int]:
    """Validate a chunk size/overlap pair; None means the configured default."""
    size = CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    try:
        size, overlap = int(size), int(overlap)
    except (TypeError, ValueError) as exc:
        raise ValueError("Chunk size and overlap must be whole numbers.") from exc
    if not MIN_CHUNK_SIZE <= size <= MAX_CHUNK_SIZE:
        raise ValueError(f"Chunk size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE} characters.")
    if overlap < 0 or overlap * 2 > size:
        raise ValueError("Chunk overlap must be zero or more and at most half of the chunk size.")
    return size, overlap


def _splitter(chunk_size=None, chunk_overlap=None):
    # Imported lazily: ``import langchain_text_splitters`` eagerly imports
    # sentence-transformers, transformers and torch (~6 s, ~800 MB), which the
    # chat page does not need until documents are actually indexed.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size, chunk_overlap = normalize_chunk_settings(chunk_size, chunk_overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# One Chroma client owns a Rust runtime (~5 threads and several MB). ``lru_cache`` evicted
# clients without closing them, so every visiting session leaked its runtime: 1000 visitors
# left ~4,000 threads and 1.3 GB behind. This cache closes what it evicts.
CHROMA_CLIENT_CACHE_SIZE = max(4, int(os.getenv("CHROMA_CLIENT_CACHE", "32") or 32))
_CLIENT_CLOSE_DELAY_SECONDS = 120.0  # grace period so a query already in flight can finish
_clients: "OrderedDict[str, object]" = OrderedDict()
_clients_lock = threading.Lock()


def _close_client(client) -> None:
    try:
        client.close()
    except Exception:
        pass


def _close_client_later(client) -> None:
    timer = threading.Timer(_CLIENT_CLOSE_DELAY_SECONDS, _close_client, args=(client,))
    timer.daemon = True
    timer.start()


def _persistent_client(resolved_chroma_dir: str):
    """Reuse one Chroma client per isolated workspace path, with a bounded, closing cache."""
    with _clients_lock:
        client = _clients.get(resolved_chroma_dir)
        if client is not None and not getattr(client, "_closed", False):
            _clients.move_to_end(resolved_chroma_dir)
            return client
        import chromadb

        client = chromadb.PersistentClient(path=resolved_chroma_dir)
        _clients[resolved_chroma_dir] = client
        _clients.move_to_end(resolved_chroma_dir)
        while len(_clients) > CHROMA_CLIENT_CACHE_SIZE:
            _, evicted = _clients.popitem(last=False)
            _close_client_later(evicted)
        return client


def _clear_client_cache() -> None:
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        _close_client(client)


_persistent_client.cache_clear = _clear_client_cache


def release_chroma_client(chroma_dir) -> None:
    """Close and forget the client for a workspace that is being deleted."""
    key = str(Path(chroma_dir).resolve())
    with _clients_lock:
        client = _clients.pop(key, None)
    if client is not None:
        _close_client(client)


def _text_chunks(lines, source: str, location_prefix: str, chunk_size=None, chunk_overlap=None):
    chunk_size, chunk_overlap = normalize_chunk_settings(chunk_size, chunk_overlap)
    splitter = _splitter(chunk_size, chunk_overlap)
    buffer = ""
    part_number = 0
    for line in lines:
        line = (line or "").strip()
        if not line:
            continue
        buffer = f"{buffer}\n{line}" if buffer else line
        if len(buffer) >= chunk_size * 4:
            pieces = splitter.split_text(buffer)
            for piece in pieces[:-1]:
                if len(piece.strip()) >= 30:
                    part_number += 1
                    yield {
                        "text": piece,
                        "source": source,
                        "page": part_number,
                        "location": f"{location_prefix} {part_number}",
                    }
            buffer = pieces[-1] if pieces else ""
    for piece in splitter.split_text(buffer) if buffer else []:
        if len(piece.strip()) >= 30:
            part_number += 1
            yield {
                "text": piece,
                "source": source,
                "page": part_number,
                "location": f"{location_prefix} {part_number}",
            }


def _pdf_chunks(file_path: Path, file_name: str, chunk_size=None, chunk_overlap=None):
    from pdfminer.pdfpage import PDFPage
    from pdfplumber.page import Page

    splitter = _splitter(chunk_size, chunk_overlap)

    with pdfplumber.open(file_path) as pdf:
        doctop = 0
        for page_number, pdf_page in enumerate(PDFPage.create_pages(pdf.doc), start=1):
            if pdf.pages_to_parse is not None and page_number not in pdf.pages_to_parse:
                continue
            page = Page(pdf, pdf_page, page_number=page_number, initial_doctop=doctop)
            page_height = page.height
            try:
                text = (page.extract_text() or "").strip()
                for chunk in splitter.split_text(text) if text else []:
                    if len(chunk.strip()) >= 30:
                        yield {
                            "text": chunk,
                            "source": file_name,
                            "page": page_number,
                            "location": f"Page {page_number}",
                        }
            finally:
                page.close()
            doctop += page_height


def _docx_lines(file_path: Path):
    document_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(file_path) as archive:
        try:
            document_xml = archive.getinfo("word/document.xml")
        except KeyError as exc:
            raise ValueError("The DOCX file is missing its main document content.") from exc
        if document_xml.file_size > MAX_DOCX_XML_SIZE_BYTES:
            raise ValueError("The expanded DOCX document exceeds the 100MB processing limit.")
        with archive.open("word/document.xml") as xml_file:
            parents = []
            for event, element in DefusedET.iterparse(xml_file, events=("start", "end")):
                if event == "start":
                    parents.append(element)
                    continue
                if element.tag == f"{document_namespace}p":
                    text = "".join(
                        child.text or ""
                        for child in element.iter()
                        if child.tag == f"{document_namespace}t"
                    )
                    if text.strip():
                        yield text
                    element.clear()
                    if len(parents) > 1:
                        parents[-2].remove(element)
                parents.pop()


def iter_document_chunks(file_path, file_name: str | None = None, chunk_size=None, chunk_overlap=None):
    """Yield chunks without accumulating a document's extracted text in memory."""
    path = Path(file_path)
    source = Path(file_name or path.name).name
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {suffix or 'unknown'}")

    if suffix == ".pdf":
        try:
            yield from _pdf_chunks(path, source, chunk_size, chunk_overlap)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"{source}: this is not a valid or readable PDF ({type(exc).__name__})."
            ) from exc
    elif suffix == ".docx":
        yield from _text_chunks(_docx_lines(path), source, "Chunk", chunk_size, chunk_overlap)
    else:
        with path.open("r", encoding="utf-8-sig", errors="replace") as document:
            blocks = iter(lambda: document.read(64 * 1024), "")
            yield from _text_chunks(blocks, source, "Chunk", chunk_size, chunk_overlap)


def extract_pdf_chunks(file_path, file_name: str):
    """Backwards-compatible list helper for small files and existing callers."""
    return list(_pdf_chunks(Path(file_path), file_name))


def _file_digest(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as document:
        for block in iter(lambda: document.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection(chroma_dir: str | Path, collection_name: str):
    import chromadb

    client = _persistent_client(str(Path(chroma_dir).resolve()))
    collection = client.get_or_create_collection(
        name=collection_name,
    )
    return client, collection


# Indexing is CPU and RAM heavy (embedding on CPU). Without a gate, a handful of
# simultaneous uploads can push a small host out of memory.
INGEST_MAX_CONCURRENT = max(1, int(os.getenv("INGEST_MAX_CONCURRENT", "2") or 2))
INGEST_WAIT_SECONDS = float(os.getenv("INGEST_WAIT_SECONDS", "120") or 120)
MAX_INDEX_BYTES = int(float(os.getenv("MAX_WORKSPACE_INDEX_MB", "400") or 400) * 1024 * 1024)
_INGEST_SLOTS = threading.BoundedSemaphore(INGEST_MAX_CONCURRENT)


def _directory_exceeds(path: Path, limit: int) -> bool:
    total = 0
    if not path.exists():
        return False
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
        if total > limit:
            return True
    return False


def build_vectorstore(*args, **kwargs):
    """Index documents, but only ``INGEST_MAX_CONCURRENT`` users at a time."""
    chroma_dir = kwargs.get("chroma_dir")
    if chroma_dir is not None and _directory_exceeds(Path(chroma_dir), MAX_INDEX_BYTES):
        raise ValueError(
            f"Your index is larger than {MAX_INDEX_BYTES // (1024 * 1024)} MB. "
            "Clear the workspace before adding more documents."
        )
    if not _INGEST_SLOTS.acquire(timeout=INGEST_WAIT_SECONDS):
        raise ValueError("The server is busy indexing other users' documents. Please try again in a minute.")
    try:
        return _build_vectorstore(*args, **kwargs)
    finally:
        _INGEST_SLOTS.release()


def _build_vectorstore(
    uploaded_files,
    *,
    chroma_dir: str | Path | None = None,
    collection_name: str = COLLECTION_NAME,
    progress_callback=None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
    embedding_api_key: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """Incrementally upsert files; repeated uploads of the same content are idempotent.

    uploaded_files is a sequence of (file_name, local_path) pairs. Files are
    parsed in pages/lines and indexed in bounded batches.
    """
    if not isinstance(uploaded_files, (list, tuple)):
        raise ValueError("Uploaded documents must be supplied as a bounded list.")
    if not uploaded_files or len(uploaded_files) > MAX_UPLOAD_FILE_COUNT:
        raise ValueError(f"Upload between 1 and {MAX_UPLOAD_FILE_COUNT} documents per batch.")
    chunk_size, chunk_overlap = normalize_chunk_settings(chunk_size, chunk_overlap)
    normalized_embedding = normalize_embedding_settings(
        embedding_provider,
        embedding_model,
        embedding_api_key,
    )
    uploaded_files = list(uploaded_files)
    total_upload_bytes = 0
    for file_name, file_path in uploaded_files:
        if not isinstance(file_name, str) or not file_name.strip() or len(file_name) > 255:
            raise ValueError("Every uploaded document must have a valid file name.")
        path = Path(file_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"{Path(file_name).name}: supported formats are PDF, DOCX, TXT, and Markdown.")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"{Path(file_name).name}: uploaded document is empty.")
        if size > MAX_DOCUMENT_SIZE_BYTES:
            raise ValueError(f"{Path(file_name).name}: each document must be 25MB or smaller.")
        total_upload_bytes += size
    if total_upload_bytes > MAX_TOTAL_UPLOAD_SIZE_BYTES:
        raise ValueError("The total upload size must be 75MB or less.")

    embedding_provider = normalized_embedding["provider"]
    embedding_model = normalized_embedding["model"]
    embedding_api_key = normalized_embedding["api_key"]
    index_dir = Path(chroma_dir) if chroma_dir is not None else CHROMA_DIR
    resolved_collection_name = collection_name_for_embedding(
        embedding_provider, embedding_model, collection_name
    )
    _, collection = _collection(index_dir, resolved_collection_name)
    lexical_index = LexicalIndex(
        index_dir / lexical_index_name(
            embedding_provider,
            embedding_model,
            collection_base_name=collection_name,
        )
    )
    embeddings = build_embedding_backend(
        embedding_provider,
        embedding_model,
        embedding_api_key,
    )
    try:
        total_chunks = _index_files(
            uploaded_files,
            collection,
            lexical_index,
            embeddings,
            progress_callback,
            chunk_size,
            chunk_overlap,
        )
    finally:
        close_embedding_backend(embeddings)

    if total_chunks == 0:
        raise ValueError("No readable text was found in the uploaded documents.")
    return collection, total_chunks


def _index_files(
    uploaded_files, collection, lexical_index, embeddings, progress_callback,
    chunk_size=None, chunk_overlap=None,
):
    chunk_size, chunk_overlap = normalize_chunk_settings(chunk_size, chunk_overlap)
    chunking_tag = "" if (chunk_size, chunk_overlap) == LEGACY_CHUNKING else f":c{chunk_size}o{chunk_overlap}"
    total_chunks = 0
    for file_name, file_path in uploaded_files:
        path = Path(file_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"{file_name}: supported formats are PDF, DOCX, TXT, and Markdown.")
        if path.stat().st_size > MAX_DOCUMENT_SIZE_BYTES:
            raise ValueError(f"{file_name}: each document must be 25MB or smaller.")
        if path.stat().st_size == 0:
            raise ValueError(f"{file_name}: uploaded document is empty.")
        content_digest = _file_digest(path)
        file_id = hashlib.sha256(
            f"{Path(file_name).name.casefold()}:{content_digest}{chunking_tag}".encode("utf-8")
        ).hexdigest()
        existing = collection.get(where={"file_id": file_id}, include=["metadatas"])
        known_ids = set(existing.get("ids") or [])
        added_ids = set()
        batch = []
        chunk_index = 0
        try:
            for chunk in iter_document_chunks(path, file_name, chunk_size, chunk_overlap):
                chunk_index += 1
                if chunk_index > MAX_CHUNKS_PER_DOCUMENT:
                    raise ValueError(
                        f"{Path(file_name).name}: this document produces more than "
                        f"{MAX_CHUNKS_PER_DOCUMENT:,} text chunks. Upload a shorter file or split it."
                    )
                batch.append(chunk)
                if len(batch) == INDEX_BATCH_SIZE:
                    first_index = chunk_index - len(batch)
                    batch_ids = _chunk_ids(file_id, first_index, len(batch))
                    added_ids.update(set(batch_ids) - known_ids)
                    _upsert_batch(collection, file_id, first_index, batch, embeddings)
                    lexical_index.upsert_batch(batch_ids, batch, file_id)
                    known_ids.update(batch_ids)
                    total_chunks += len(batch)
                    batch.clear()
                    if progress_callback:
                        progress_callback(file_name, chunk_index)
            if batch:
                first_index = chunk_index - len(batch)
                batch_ids = _chunk_ids(file_id, first_index, len(batch))
                added_ids.update(set(batch_ids) - known_ids)
                _upsert_batch(collection, file_id, first_index, batch, embeddings)
                lexical_index.upsert_batch(batch_ids, batch, file_id)
                known_ids.update(batch_ids)
                total_chunks += len(batch)
                if progress_callback:
                    progress_callback(file_name, chunk_index)
        except Exception:
            if added_ids:
                added = list(added_ids)
                try:
                    collection.delete(ids=added)
                    lexical_index.delete_chunk_ids(added)
                except Exception:
                    # The next retrieval detects a Chroma/SQLite count mismatch
                    # and rebuilds the sidecar before serving BM25 results.
                    pass
            raise
        if chunk_index:
            source_name = Path(file_name).name
            collection.delete(
                where={
                    "$and": [
                        {"source": source_name},
                        {"file_id": {"$ne": file_id}},
                    ]
                }
            )
            lexical_index.delete_stale_source_versions(source_name, file_id)
    return total_chunks


def _upsert_batch(
    collection,
    file_id: str,
    first_index: int,
    chunks: list[dict],
    embeddings,
):
    ids = _chunk_ids(file_id, first_index, len(chunks))
    collection.upsert(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        embeddings=validate_embeddings(
            embeddings.embed_documents([chunk["text"] for chunk in chunks]),
            len(chunks),
        ),
        metadatas=[
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "location": chunk["location"],
                "file_id": file_id,
            }
            for chunk in chunks
        ],
    )
    return ids


def _chunk_ids(file_id: str, first_index: int, count: int) -> list[str]:
    return [
        hashlib.sha256(f"{file_id}:{first_index + offset}".encode("utf-8")).hexdigest()
        for offset in range(count)
    ]


def get_existing_collection(
    *,
    chroma_dir: str | Path | None = None,
    collection_name: str = COLLECTION_NAME,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
):
    import chromadb

    index_dir = Path(chroma_dir) if chroma_dir is not None else CHROMA_DIR
    if not (index_dir / "chroma.sqlite3").exists():
        return None  # nothing indexed yet: do not start a database runtime for a brand-new visitor
    client = _persistent_client(str(index_dir.resolve()))
    resolved_collection_name = collection_name_for_embedding(
        embedding_provider, embedding_model, collection_name
    )
    collection_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    if resolved_collection_name not in collection_names:
        return None
    return client.get_collection(
        name=resolved_collection_name,
    )
