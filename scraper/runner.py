"""Orchestrate scrape jobs for a source + publish-time range."""

from __future__ import annotations

from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Callable, Optional

from scraper import db
from scraper.base import ArticleDetail, ArticleListing, ScrapeProgress, day_range
from scraper.sources import build_scraper
from scraper.sources.skds import detail_to_record


ProgressCallback = Callable[[ScrapeProgress, str], None]

DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
)


def parse_datetime(value: str | date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, dtime.min)
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Thời gian không hợp lệ: {value!r} "
        f"(dùng YYYY-MM-DD HH:MM:SS hoặc YYYY-MM-DD)"
    )


def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def resolve_time_range(
    *,
    target_date: str | date | None = None,
    from_time: str | date | datetime | None = None,
    to_time: str | date | datetime | None = None,
) -> tuple[datetime, datetime]:
    """
    Ưu tiên from_time/to_time.
    Nếu chỉ có target_date → cả ngày [00:00:00, 23:59:59].
    """
    if from_time is not None or to_time is not None:
        if from_time is None or to_time is None:
            raise ValueError("Cần cả from_time và to_time")
        start = parse_datetime(from_time)
        end = parse_datetime(to_time)
        # Nếu chỉ nhập ngày (không giờ) cho to_time dạng date-only string → cuối ngày
        if isinstance(to_time, str) and len(to_time.strip()) == 10:
            end = datetime.combine(end.date(), dtime(23, 59, 59))
        elif isinstance(to_time, date) and not isinstance(to_time, datetime):
            end = datetime.combine(to_time, dtime(23, 59, 59))
        if end < start:
            raise ValueError(f"to_time ({format_datetime(end)}) < from_time ({format_datetime(start)})")
        return start, end

    if target_date is None:
        raise ValueError("Cần target_date hoặc cặp from_time/to_time")

    if isinstance(target_date, datetime):
        return day_range(target_date.date())
    if isinstance(target_date, date):
        return day_range(target_date)
    parsed = parse_datetime(target_date)
    if len(str(target_date).strip()) == 10:
        return day_range(parsed.date())
    # datetime string used as single target → that exact second only is odd;
    # treat date part as full day for --date compatibility
    return day_range(parsed.date())


def run_scrape_job(
    *,
    source_id: Optional[int] = None,
    source_code: Optional[str] = None,
    target_date: str | date | None = None,
    from_time: str | date | datetime | None = None,
    to_time: str | date | datetime | None = None,
    db_path: Path | str | None = None,
    on_progress: Optional[ProgressCallback] = None,
    request_delay: float = 0.5,
    auto_index: bool = True,
) -> dict:
    """
    Chạy một job scrape theo khoảng thời gian xuất bản.

    Nếu ``auto_index=True``, bài success được đưa vào hàng đợi index Qdrant (nền).

    Returns dict gồm job_id và thống kê.
    """
    db.init_db(db_path)
    start, end = resolve_time_range(
        target_date=target_date, from_time=from_time, to_time=to_time
    )

    if source_id is not None:
        source = db.get_source(source_id, db_path=db_path)
    elif source_code:
        source = db.get_source_by_code(source_code, db_path=db_path)
    else:
        raise ValueError("Cần source_id hoặc source_code")

    if not source:
        raise ValueError("Không tìm thấy nguồn báo")
    if not source.get("enabled"):
        raise ValueError(f"Nguồn '{source['name']}' đang bị tắt")

    job_id = db.create_job(
        source["id"],
        target_date=start.date().isoformat(),
        from_time=format_datetime(start),
        to_time=format_datetime(end),
        db_path=db_path,
    )
    progress = ScrapeProgress()
    new_article_ids: list[int] = []

    def emit(msg: str) -> None:
        progress.messages.append(msg)
        if on_progress:
            on_progress(progress, msg)

    try:
        scraper = build_scraper(source)
        scraper.request_delay = request_delay
        emit(
            f"Bắt đầu scrape {source['name']} "
            f"từ {format_datetime(start)} đến {format_datetime(end)}"
        )

        for status, payload, err in scraper.scrape_range(start, end):
            progress.found += 1

            if status == "error":
                assert isinstance(payload, ArticleListing)
                progress.failed += 1
                db.insert_failed_attempt(
                    job_id=job_id,
                    source=source["code"],
                    link=payload.link,
                    title=payload.title,
                    publish_time=(
                        payload.publish_time.strftime("%Y-%m-%d %H:%M:%S.000000")
                        if payload.publish_time
                        else None
                    ),
                    error_message=err or "unknown error",
                    external_id=payload.external_id,
                    db_path=db_path,
                )
                emit(f"FAIL: {payload.link} — {err}")
                continue

            assert isinstance(payload, ArticleDetail)
            if payload.publish_time is None or not (start <= payload.publish_time <= end):
                progress.skipped += 1
                pt = format_datetime(payload.publish_time) if payload.publish_time else "N/A"
                emit(f"SKIP time: {payload.title[:60]} ({pt} ngoài khoảng)")
                continue

            record = detail_to_record(payload)
            record["job_id"] = job_id

            if db.article_exists(
                record["title"],
                record["source"],
                record["publish_time"],
                db_path=db_path,
            ):
                progress.skipped += 1
                emit(f"SKIP dup: {record['title'][:80]}")
                continue

            result_status, aid = db.insert_article(record, db_path=db_path)
            if result_status == "skipped_duplicate" or aid is None:
                progress.skipped += 1
                emit(f"SKIP dup: {record['title'][:80]}")
            else:
                progress.success += 1
                new_article_ids.append(aid)
                emit(f"OK: {record['title'][:80]}")

        db.finish_job(
            job_id,
            status="success",
            total_found=progress.found,
            success_count=progress.success,
            failed_count=progress.failed,
            skipped_count=progress.skipped,
            db_path=db_path,
        )
        emit(
            f"Hoàn tất: found={progress.found}, ok={progress.success}, "
            f"fail={progress.failed}, skip={progress.skipped}"
        )

        indexed_queued = 0
        if auto_index and new_article_ids:
            from scraper.qdrant_indexer import enqueue_articles

            indexed_queued = enqueue_articles(new_article_ids, db_path=db_path)
            emit(
                f"Đã xếp hàng index Qdrant (nền): {indexed_queued} bài "
                f"(article_id=scrape_<id>)"
            )
    except Exception as exc:  # noqa: BLE001
        db.finish_job(
            job_id,
            status="failed",
            total_found=progress.found,
            success_count=progress.success,
            failed_count=progress.failed,
            skipped_count=progress.skipped,
            error_message=str(exc),
            db_path=db_path,
        )
        emit(f"JOB FAILED: {exc}")
        raise

    return {
        "job_id": job_id,
        "source": source["code"],
        "target_date": start.date().isoformat(),
        "from_time": format_datetime(start),
        "to_time": format_datetime(end),
        "total_found": progress.found,
        "success_count": progress.success,
        "failed_count": progress.failed,
        "skipped_count": progress.skipped,
        "new_article_ids": new_article_ids,
        "index_queued": indexed_queued if auto_index else 0,
        "messages": progress.messages,
    }


def export_articles_to_csv(
    csv_path: Path | str,
    *,
    source: Optional[str] = None,
    status: str = "success",
    db_path: Path | str | None = None,
) -> int:
    """Export bài success ra CSV (schema gần với articles_skds.csv)."""
    import pandas as pd

    rows = db.list_articles(source=source, status=status, limit=1_000_000, db_path=db_path)
    if not rows:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
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
            ]
        ).to_csv(csv_path, index=False)
        return 0

    df = pd.DataFrame(rows)
    cols = [
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
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(csv_path, index=False)
    return len(df)
