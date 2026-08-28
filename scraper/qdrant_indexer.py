"""
Incremental index scraped articles vào Qdrant (chạy nền).

Dùng chung client local với retrieval (`qdrant_store`) + lock để tránh race.
Point id = UUID5(scrape_{article_id}#{chunk_idx}) — upsert idempotent.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from ingest import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_BATCH_SIZE, get_collection_name
from qdrant_store import get_qdrant_client, qdrant_lock
from scraper import db

logger = logging.getLogger(__name__)

_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

_embedder: SentenceTransformer | None = None
_embedder_lock = threading.Lock()

_task_queue: queue.Queue[tuple[int, Path | str | None, bool]] = queue.Queue()
_worker_started = False
_worker_guard = threading.Lock()
_queue_ids_lock = threading.Lock()
_queued_ids: set[int] = set()


def rag_article_id(db_article_id: int) -> str:
    """article_id trong payload Qdrant (phân biệt với skds_/umc_ từ CSV)."""
    return f"scrape_{db_article_id}"


def point_uuid(rag_id: str, chunk_idx: int) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{rag_id}#{chunk_idx}"))


def _get_embedder() -> SentenceTransformer:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            logger.info("Loading embedding model %s for indexer...", EMBEDDING_MODEL)
            _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        return _embedder


def ensure_collection() -> str:
    client = get_qdrant_client()
    name = get_collection_name()
    embedder = _get_embedder()
    dim = embedder.get_sentence_embedding_dimension()
    with qdrant_lock():
        existing = {c.name for c in client.get_collections().collections}
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection %s", name)
    return name


def _chunk_article(article: dict) -> list[dict]:
    title = article.get("title") or ""
    content = article.get("content") or ""
    source = article.get("source") or ""
    text = f"{title}\n\n{content}".strip()
    if not text:
        return []

    rag_id = rag_article_id(int(article["id"]))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return [
        {
            "text": chunk,
            "article_id": rag_id,
            "title": title,
            "source": source,
            "chunk_idx": i,
            "db_id": int(article["id"]),
            "link": article.get("link"),
            "publish_time": article.get("publish_time"),
        }
        for i, chunk in enumerate(splitter.split_text(text))
    ]


def index_article(
    article_id: int,
    db_path: Path | str | None = None,
    *,
    force: bool = False,
) -> int:
    """
    Index 1 bài vào Qdrant. Trả về số chunk.

    Nếu đã có ``index_chunks > 0`` và ``index_status=indexed`` thì bỏ qua
    (trừ khi ``force=True`` sau khi reset marker).
    """
    article = db.get_article(article_id, db_path=db_path)
    if not article:
        raise ValueError(f"Không tìm thấy article id={article_id}")
    if article.get("scrape_status") != "success":
        db.set_index_status(article_id, "skipped", index_error="not success", db_path=db_path)
        return 0

    if not force and db.is_qdrant_indexed(article):
        logger.info(
            "Skip article %s — already indexed (%s chunks)",
            article_id,
            article.get("index_chunks"),
        )
        return int(article.get("index_chunks") or 0)

    db.set_index_status(article_id, "indexing", index_error=None, db_path=db_path)
    chunks = _chunk_article(article)
    if not chunks:
        db.set_index_status(
            article_id,
            "failed",
            index_error="empty content",
            index_chunks=0,
            db_path=db_path,
        )
        return 0

    collection = ensure_collection()
    client = get_qdrant_client()
    embedder = _get_embedder()
    rag_id = rag_article_id(article_id)

    try:
        texts = [c["text"] for c in chunks]
        vectors = embedder.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = [
            PointStruct(
                id=point_uuid(rag_id, c["chunk_idx"]),
                vector=vectors[i].tolist(),
                payload={
                    "text": c["text"],
                    "article_id": c["article_id"],
                    "title": c["title"],
                    "source": c["source"],
                    "link": c.get("link"),
                    "publish_time": c.get("publish_time"),
                    "chunk_idx": c["chunk_idx"],
                },
            )
            for i, c in enumerate(chunks)
        ]

        with qdrant_lock():
            # Xóa chunk cũ cùng article_id (nếu re-index, số chunk có thể đổi)
            try:
                client.delete(
                    collection_name=collection,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="article_id",
                                    match=MatchValue(value=rag_id),
                                )
                            ]
                        )
                    ),
                )
            except UnexpectedResponse:
                pass

            client.upsert(collection_name=collection, points=points)

        db.set_index_status(
            article_id,
            "indexed",
            indexed_at=db.utcnow(),
            index_error=None,
            index_chunks=len(points),
            db_path=db_path,
        )
        logger.info("Indexed article %s → %s chunks", article_id, len(points))
        return len(points)
    except Exception as exc:  # noqa: BLE001
        db.set_index_status(
            article_id,
            "failed",
            index_error=str(exc)[:500],
            db_path=db_path,
        )
        logger.exception("Index failed for article %s", article_id)
        raise


def _worker_loop() -> None:
    while True:
        article_id, db_path, force = _task_queue.get()
        try:
            index_article(article_id, db_path=db_path, force=force)
        except Exception:  # noqa: BLE001 - already logged / status updated
            pass
        finally:
            with _queue_ids_lock:
                _queued_ids.discard(article_id)
            _task_queue.task_done()


def start_indexer_worker() -> None:
    global _worker_started
    with _worker_guard:
        if _worker_started:
            return
        thread = threading.Thread(
            target=_worker_loop,
            name="qdrant-indexer",
            daemon=True,
        )
        thread.start()
        _worker_started = True
        logger.info("Background Qdrant indexer started")


def enqueue_articles(
    article_ids: list[int],
    *,
    db_path: Path | str | None = None,
    force: bool = False,
) -> int:
    """
    Đưa các bài vào hàng đợi index nền.

    Mặc định bỏ qua bài đã có ``index_chunks > 0`` (đã nạp Qdrant).
    ``force=True`` dùng sau ``reset_index_markers``.
    """
    if not article_ids:
        return 0
    start_indexer_worker()
    added = 0
    with _queue_ids_lock:
        for aid in article_ids:
            if aid in _queued_ids:
                continue
            article = db.get_article(aid, db_path=db_path)
            if not article or article.get("scrape_status") != "success":
                continue
            if not force and db.is_qdrant_indexed(article):
                continue
            _queued_ids.add(aid)
            if not force:
                # Giữ pending nếu chưa indexed; không đụng bài đã indexed
                if article.get("index_status") != "indexing":
                    db.set_index_status(aid, "pending", index_error=None, db_path=db_path)
            _task_queue.put((aid, db_path, force))
            added += 1
    return added


def enqueue_pending(*, db_path: Path | str | None = None, limit: int = 500) -> int:
    rows = db.list_articles_needing_index(limit=limit, include_failed=True, db_path=db_path)
    return enqueue_articles([r["id"] for r in rows], db_path=db_path, force=False)


def reindex_articles(
    article_ids: list[int],
    *,
    db_path: Path | str | None = None,
) -> int:
    """Reset marker rồi xếp hàng nạp lại Qdrant (cố ý)."""
    db.reset_index_markers(article_ids, db_path=db_path)
    return enqueue_articles(article_ids, db_path=db_path, force=True)


def wait_until_idle(timeout: float | None = None) -> bool:
    """Chờ hàng đợi index trống. Trả về True nếu xong."""
    import time

    start = time.time()
    _task_queue.join()
    if timeout is not None and (time.time() - start) > timeout:
        return False
    return True


def queue_size() -> int:
    return _task_queue.qsize()


def index_pending_sync(*, db_path: Path | str | None = None, limit: int = 500) -> dict:
    """Index đồng bộ (CLI) các bài pending/failed."""
    rows = db.list_articles_needing_index(limit=limit, include_failed=True, db_path=db_path)
    ok = fail = 0
    chunks = 0
    for row in rows:
        try:
            chunks += index_article(row["id"], db_path=db_path)
            ok += 1
        except Exception:  # noqa: BLE001
            fail += 1
    return {"queued": len(rows), "ok": ok, "failed": fail, "chunks": chunks}
