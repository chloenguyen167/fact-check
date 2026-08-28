"""SQLite persistence for scrape sources, jobs, and articles."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scraper.db"

ARTICLE_COLUMNS = [
    "id",
    "title",
    "publish_time",
    "content",
    "source",
    "link",
    "author",
    "tags",
    "related_one_news",
    "scrape_status",
    "scraped_at",
    "job_id",
    "error_message",
    "external_id",
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def utcnow() -> str:
    return _utcnow()


@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                base_url TEXT NOT NULL,
                list_url TEXT NOT NULL,
                scraper_key TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                config_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scrape_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_date TEXT NOT NULL,
                from_time TEXT,
                to_time TEXT,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                total_found INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                skipped_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                FOREIGN KEY (source_id) REFERENCES sources(id)
            );

            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                publish_time TEXT,
                content TEXT,
                source TEXT NOT NULL,
                link TEXT,
                author TEXT,
                tags TEXT,
                related_one_news TEXT,
                scrape_status TEXT NOT NULL,
                scraped_at TEXT,
                job_id INTEGER,
                error_message TEXT,
                external_id TEXT,
                index_status TEXT,
                indexed_at TEXT,
                index_error TEXT,
                index_chunks INTEGER,
                UNIQUE (title, source, publish_time),
                FOREIGN KEY (job_id) REFERENCES scrape_jobs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_articles_source_date
                ON articles(source, publish_time);
            CREATE INDEX IF NOT EXISTS idx_articles_status
                ON articles(scrape_status);
            CREATE INDEX IF NOT EXISTS idx_jobs_source_date
                ON scrape_jobs(source_id, target_date);
            """
        )
        _migrate_scrape_jobs(conn)
        _migrate_articles_index(conn)
        _seed_default_sources(conn)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_scrape_jobs(conn: sqlite3.Connection) -> None:
    """Thêm cột from_time/to_time cho DB đã tạo trước đó."""
    cols = _table_columns(conn, "scrape_jobs")
    if "from_time" not in cols:
        conn.execute("ALTER TABLE scrape_jobs ADD COLUMN from_time TEXT")
    if "to_time" not in cols:
        conn.execute("ALTER TABLE scrape_jobs ADD COLUMN to_time TEXT")


def _migrate_articles_index(conn: sqlite3.Connection) -> None:
    """Thêm cột trạng thái index Qdrant."""
    cols = _table_columns(conn, "articles")
    alters = {
        "index_status": "ALTER TABLE articles ADD COLUMN index_status TEXT",
        "indexed_at": "ALTER TABLE articles ADD COLUMN indexed_at TEXT",
        "index_error": "ALTER TABLE articles ADD COLUMN index_error TEXT",
        "index_chunks": "ALTER TABLE articles ADD COLUMN index_chunks INTEGER",
    }
    for col, sql in alters.items():
        if col not in cols:
            conn.execute(sql)
    # Bài success cũ chưa index → pending
    conn.execute(
        """
        UPDATE articles
        SET index_status = 'pending'
        WHERE scrape_status = 'success'
          AND (index_status IS NULL OR index_status = '')
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_articles_index_status
            ON articles(index_status)
        """
    )


def _seed_default_sources(conn: sqlite3.Connection) -> None:
    now = _utcnow()
    conn.execute(
        """
        INSERT OR IGNORE INTO sources
            (code, name, base_url, list_url, scraper_key, enabled, config_json, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
        """,
        (
            "suckhoedoisong",
            "Báo Sức khỏe & Đời sống",
            "https://suckhoedoisong.vn",
            "https://suckhoedoisong.vn/y-te.htm",
            "skds",
            json.dumps({"zone_id": None}, ensure_ascii=False),
            "Chuyên mục Y tế. Scraper dùng timeline /timelinelist/{zone}/{page}.htm",
            now,
            now,
        ),
    )


def list_sources(enabled_only: bool = False, db_path: Path | str | None = None) -> list[dict]:
    init_db(db_path)
    q = "SELECT * FROM sources"
    if enabled_only:
        q += " WHERE enabled = 1"
    q += " ORDER BY id"
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(q).fetchall()]


def get_source(source_id: int, db_path: Path | str | None = None) -> Optional[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        return dict(row) if row else None


def get_source_by_code(code: str, db_path: Path | str | None = None) -> Optional[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sources WHERE code = ?", (code,)).fetchone()
        return dict(row) if row else None


def upsert_source(
    *,
    code: str,
    name: str,
    base_url: str,
    list_url: str,
    scraper_key: str,
    enabled: bool = True,
    config: Optional[dict] = None,
    notes: str = "",
    source_id: Optional[int] = None,
    db_path: Path | str | None = None,
) -> int:
    init_db(db_path)
    now = _utcnow()
    cfg = json.dumps(config or {}, ensure_ascii=False)
    with connect(db_path) as conn:
        if source_id is None:
            cur = conn.execute(
                """
                INSERT INTO sources
                    (code, name, base_url, list_url, scraper_key, enabled, config_json, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    code,
                    name,
                    base_url,
                    list_url,
                    scraper_key,
                    int(enabled),
                    cfg,
                    notes,
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)
        conn.execute(
            """
            UPDATE sources
            SET code=?, name=?, base_url=?, list_url=?, scraper_key=?, enabled=?,
                config_json=?, notes=?, updated_at=?
            WHERE id=?
            """,
            (
                code,
                name,
                base_url,
                list_url,
                scraper_key,
                int(enabled),
                cfg,
                notes,
                now,
                source_id,
            ),
        )
        return source_id


def set_source_enabled(source_id: int, enabled: bool, db_path: Path | str | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE sources SET enabled = ?, updated_at = ? WHERE id = ?",
            (int(enabled), _utcnow(), source_id),
        )


def delete_source(source_id: int, db_path: Path | str | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))


def create_job(
    source_id: int,
    target_date: str,
    *,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    db_path: Path | str | None = None,
) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO scrape_jobs
                (source_id, target_date, from_time, to_time, status, started_at)
            VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (source_id, target_date, from_time, to_time, _utcnow()),
        )
        return int(cur.lastrowid)


def update_job(job_id: int, db_path: Path | str | None = None, **fields: Any) -> None:
    if not fields:
        return
    init_db(db_path)
    cols = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    with connect(db_path) as conn:
        conn.execute(f"UPDATE scrape_jobs SET {cols} WHERE id = ?", values)


def finish_job(
    job_id: int,
    *,
    status: str,
    total_found: int = 0,
    success_count: int = 0,
    failed_count: int = 0,
    skipped_count: int = 0,
    error_message: Optional[str] = None,
    db_path: Path | str | None = None,
) -> None:
    update_job(
        job_id,
        db_path=db_path,
        status=status,
        finished_at=_utcnow(),
        total_found=total_found,
        success_count=success_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        error_message=error_message,
    )


def list_jobs(limit: int = 100, db_path: Path | str | None = None) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT j.*, s.name AS source_name, s.code AS source_code
            FROM scrape_jobs j
            JOIN sources s ON s.id = j.source_id
            ORDER BY j.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_job(job_id: int, db_path: Path | str | None = None) -> Optional[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT j.*, s.name AS source_name, s.code AS source_code
            FROM scrape_jobs j
            JOIN sources s ON s.id = j.source_id
            WHERE j.id = ?
            """,
            (job_id,),
        ).fetchone()
        return dict(row) if row else None


def article_exists(
    title: str,
    source: str,
    publish_time: Optional[str],
    db_path: Path | str | None = None,
) -> bool:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM articles
            WHERE title = ? AND source = ? AND IFNULL(publish_time, '') = ?
            LIMIT 1
            """,
            (title, source, publish_time or ""),
        ).fetchone()
        return row is not None


def insert_article(article: dict, db_path: Path | str | None = None) -> tuple[str, Optional[int]]:
    """Insert article. Returns (status, article_id). status: success | skipped_duplicate."""
    init_db(db_path)
    publish_time = article.get("publish_time") or ""
    payload = {
        "title": (article.get("title") or "").strip(),
        "publish_time": publish_time,
        "content": article.get("content"),
        "source": article.get("source"),
        "link": article.get("link"),
        "author": article.get("author"),
        "tags": article.get("tags"),
        "related_one_news": article.get("related_one_news"),
        "scrape_status": article.get("scrape_status") or "success",
        "scraped_at": article.get("scraped_at") or _utcnow(),
        "job_id": article.get("job_id"),
        "error_message": article.get("error_message"),
        "external_id": article.get("external_id"),
        "index_status": article.get("index_status"),
    }
    if not payload["title"] or not payload["source"]:
        raise ValueError("title và source là bắt buộc")
    if payload["scrape_status"] == "success" and not payload["index_status"]:
        payload["index_status"] = "pending"

    with connect(db_path) as conn:
        exists = conn.execute(
            """
            SELECT 1 FROM articles
            WHERE title = ? AND source = ? AND IFNULL(publish_time, '') = ?
            LIMIT 1
            """,
            (payload["title"], payload["source"], publish_time),
        ).fetchone()
        if exists:
            return "skipped_duplicate", None
        try:
            cur = conn.execute(
                """
                INSERT INTO articles
                    (title, publish_time, content, source, link, author, tags,
                     related_one_news, scrape_status, scraped_at, job_id, error_message,
                     external_id, index_status)
                VALUES
                    (:title, :publish_time, :content, :source, :link, :author, :tags,
                     :related_one_news, :scrape_status, :scraped_at, :job_id, :error_message,
                     :external_id, :index_status)
                """,
                payload,
            )
            return payload["scrape_status"], int(cur.lastrowid)
        except sqlite3.IntegrityError:
            return "skipped_duplicate", None


def insert_failed_attempt(
    *,
    job_id: int,
    source: str,
    link: Optional[str],
    title: Optional[str],
    publish_time: Optional[str],
    error_message: str,
    external_id: Optional[str] = None,
    db_path: Path | str | None = None,
) -> None:
    """Log a failed scrape without colliding on unique key when title/time unknown."""
    init_db(db_path)
    safe_title = (title or f"[FAILED] {link or 'unknown'}").strip()
    # Avoid unique collision with previous failures of same link
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    if not title:
        safe_title = f"{safe_title} ({suffix})"
    with connect(db_path) as conn:
        try:
            conn.execute(
                """
                INSERT INTO articles
                    (title, publish_time, content, source, link, author, tags,
                     related_one_news, scrape_status, scraped_at, job_id, error_message, external_id)
                VALUES (?, ?, NULL, ?, ?, NULL, NULL, NULL, 'failed', ?, ?, ?, ?)
                """,
                (
                    safe_title,
                    publish_time,
                    source,
                    link,
                    _utcnow(),
                    job_id,
                    error_message,
                    external_id,
                ),
            )
        except sqlite3.IntegrityError:
            # Already have a success/skip for this key; ignore fail log collision
            pass


def list_articles(
    *,
    source: Optional[str] = None,
    status: Optional[str] = None,
    index_status: Optional[str] = None,
    job_id: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 200,
    db_path: Path | str | None = None,
) -> list[dict]:
    init_db(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if source:
        clauses.append("source = ?")
        params.append(source)
    if status:
        clauses.append("scrape_status = ?")
        params.append(status)
    if index_status:
        clauses.append("index_status = ?")
        params.append(index_status)
    if job_id is not None:
        clauses.append("job_id = ?")
        params.append(job_id)
    if date_from:
        clauses.append("IFNULL(publish_time, '') >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("IFNULL(publish_time, '') <= ?")
        params.append(date_to + " 23:59:59" if len(date_to) == 10 else date_to)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM articles
            {where}
            ORDER BY scraped_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]


def get_article(article_id: int, db_path: Path | str | None = None) -> Optional[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        return dict(row) if row else None


def set_index_status(
    article_id: int,
    status: str,
    *,
    indexed_at: Optional[str] = None,
    index_error: Optional[str] = None,
    index_chunks: Optional[int] = None,
    clear_chunks: bool = False,
    db_path: Path | str | None = None,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        if clear_chunks:
            conn.execute(
                """
                UPDATE articles
                SET index_status = ?,
                    indexed_at = ?,
                    index_error = ?,
                    index_chunks = NULL
                WHERE id = ?
                """,
                (status, indexed_at, index_error, article_id),
            )
        elif index_chunks is not None:
            conn.execute(
                """
                UPDATE articles
                SET index_status = ?,
                    indexed_at = COALESCE(?, indexed_at),
                    index_error = ?,
                    index_chunks = ?
                WHERE id = ?
                """,
                (status, indexed_at, index_error, index_chunks, article_id),
            )
        else:
            conn.execute(
                """
                UPDATE articles
                SET index_status = ?,
                    indexed_at = COALESCE(?, indexed_at),
                    index_error = ?
                WHERE id = ?
                """,
                (status, indexed_at, index_error, article_id),
            )


def is_qdrant_indexed(article: dict) -> bool:
    """Đã nạp Qdrant khi index_status=indexed và index_chunks > 0."""
    chunks = article.get("index_chunks")
    try:
        n = int(chunks) if chunks is not None else 0
    except (TypeError, ValueError):
        n = 0
    return article.get("index_status") == "indexed" and n > 0


def reset_index_markers(
    article_ids: list[int],
    *,
    db_path: Path | str | None = None,
) -> int:
    """Reset marker để cố ý nạp lại Qdrant (index_chunks → NULL, status → pending)."""
    if not article_ids:
        return 0
    init_db(db_path)
    with connect(db_path) as conn:
        placeholders = ",".join("?" * len(article_ids))
        cur = conn.execute(
            f"""
            UPDATE articles
            SET index_status = 'pending',
                indexed_at = NULL,
                index_error = NULL,
                index_chunks = NULL
            WHERE id IN ({placeholders})
              AND scrape_status = 'success'
            """,
            article_ids,
        )
        return int(cur.rowcount)


def list_articles_needing_index(
    *,
    limit: int = 500,
    include_failed: bool = True,
    db_path: Path | str | None = None,
) -> list[dict]:
    init_db(db_path)
    statuses = ("pending", "failed") if include_failed else ("pending",)
    placeholders = ",".join("?" * len(statuses))
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM articles
            WHERE scrape_status = 'success'
              AND index_status IN ({placeholders})
            ORDER BY id ASC
            LIMIT ?
            """,
            (*statuses, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def article_stats(db_path: Path | str | None = None) -> dict:
    init_db(db_path)
    with connect(db_path) as conn:
        by_status = {
            r["scrape_status"]: r["cnt"]
            for r in conn.execute(
                "SELECT scrape_status, COUNT(*) AS cnt FROM articles GROUP BY scrape_status"
            )
        }
        by_index = {
            (r["index_status"] or "null"): r["cnt"]
            for r in conn.execute(
                "SELECT index_status, COUNT(*) AS cnt FROM articles GROUP BY index_status"
            )
        }
        by_source = [
            dict(r)
            for r in conn.execute(
                """
                SELECT source, scrape_status, COUNT(*) AS cnt
                FROM articles
                GROUP BY source, scrape_status
                ORDER BY source, scrape_status
                """
            )
        ]
        total = conn.execute("SELECT COUNT(*) AS cnt FROM articles").fetchone()["cnt"]
    return {
        "total": total,
        "by_status": by_status,
        "by_index": by_index,
        "by_source": by_source,
    }
