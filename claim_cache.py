"""Cache kết quả kiểm chứng claim — SQLite, key theo claim đã chuẩn hóa."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from guard.rules import normalize

_DEFAULT_DB = Path(__file__).resolve().parent / "data" / "claim_cache.db"
_DEFAULT_TTL_SECONDS = 2 * 60 * 60  # 2 giờ


def _cache_version() -> str:
    try:
        import config as cfg

        parts = [
            getattr(cfg, "COLLECTION_NAME", ""),
            getattr(cfg, "EMBEDDING_MODEL", ""),
            getattr(cfg, "NLI_MODEL", ""),
            getattr(cfg, "CLAIM_CACHE_VERSION", "1"),
        ]
        return "|".join(parts)
    except Exception:  # noqa: BLE001
        return "1"


def _ttl_seconds() -> int:
    try:
        import config as cfg

        return int(getattr(cfg, "CLAIM_CACHE_TTL_SECONDS", _DEFAULT_TTL_SECONDS))
    except Exception:  # noqa: BLE001
        return _DEFAULT_TTL_SECONDS


def claim_cache_key(claim: str) -> str:
    """Key ổn định: normalize + hash."""
    norm = normalize(claim)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]
    return digest


@contextmanager
def _connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or _DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_cache (
                cache_key TEXT PRIMARY KEY,
                claim_text TEXT NOT NULL,
                claim_normalized TEXT NOT NULL,
                cache_version TEXT NOT NULL,
                verdict TEXT,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claim_cache_accessed ON claim_cache(accessed_at DESC)"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _is_expired(created_at: str) -> bool:
    age = datetime.now(timezone.utc) - _parse_iso(created_at)
    return age.total_seconds() > _ttl_seconds()


def purge_expired(db_path: Path | str | None = None) -> int:
    """Xóa các mục cache đã hết TTL. Trả về số dòng đã xóa."""
    init_db(db_path)
    version = _cache_version()
    removed = 0
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT cache_key, created_at FROM claim_cache WHERE cache_version = ?",
            (version,),
        ).fetchall()
        for row in rows:
            if _is_expired(row["created_at"]):
                conn.execute(
                    "DELETE FROM claim_cache WHERE cache_key = ?",
                    (row["cache_key"],),
                )
                removed += 1
    return removed


def get_cached(claim: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Trả về kết quả đã cache hoặc None (hết TTL / không tồn tại)."""
    init_db(db_path)
    purge_expired(db_path)
    key = claim_cache_key(claim)
    version = _cache_version()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM claim_cache WHERE cache_key = ? AND cache_version = ?",
            (key, version),
        ).fetchone()
        if not row:
            return None
        if _is_expired(row["created_at"]):
            conn.execute("DELETE FROM claim_cache WHERE cache_key = ?", (key,))
            return None
        conn.execute(
            """
            UPDATE claim_cache
            SET accessed_at = ?, hit_count = hit_count + 1
            WHERE cache_key = ?
            """,
            (_now_iso(), key),
        )
        result = json.loads(row["result_json"])
        result["from_cache"] = True
        return result


def save_cached(
    claim: str,
    result: dict[str, Any],
    db_path: Path | str | None = None,
) -> None:
    """Lưu kết quả pipeline vào cache."""
    init_db(db_path)
    purge_expired(db_path)
    key = claim_cache_key(claim)
    version = _cache_version()
    now = _now_iso()
    payload = {k: v for k, v in result.items() if not str(k).startswith("_")}
    payload.pop("from_cache", None)

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO claim_cache (
                cache_key, claim_text, claim_normalized, cache_version,
                verdict, result_json, created_at, accessed_at, hit_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(cache_key) DO UPDATE SET
                claim_text = excluded.claim_text,
                claim_normalized = excluded.claim_normalized,
                cache_version = excluded.cache_version,
                verdict = excluded.verdict,
                result_json = excluded.result_json,
                created_at = excluded.created_at,
                accessed_at = excluded.accessed_at,
                hit_count = 0
            """,
            (
                key,
                claim.strip(),
                normalize(claim),
                version,
                payload.get("verdict"),
                json.dumps(payload, ensure_ascii=False),
                now,
                now,
            ),
        )


def clear_cache(db_path: Path | str | None = None) -> int:
    """Xóa toàn bộ cache. Trả về số dòng đã xóa."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM claim_cache")
        return cur.rowcount
