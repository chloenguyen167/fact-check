"""CLI: scrape tin theo khoảng thời gian xuất bản (+ index Qdrant nền).

Ví dụ:
  python -m scraper.cli --source suckhoedoisong --date 2026-07-12
  python -m scraper.cli --from "2026-07-12 00:00:00" --to "2026-07-12 23:59:59"
  python -m scraper.cli --index-pending
  python -m scraper.cli --list-sources
"""

from __future__ import annotations

import argparse
import json
import sys

from scraper import db
from scraper.runner import export_articles_to_csv, run_scrape_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Thu thập tin tức theo khoảng thời gian xuất bản"
    )
    parser.add_argument("--list-sources", action="store_true", help="Liệt kê nguồn báo")
    parser.add_argument(
        "--index-pending",
        action="store_true",
        help="Index đồng bộ các bài success còn pending/failed vào Qdrant",
    )
    parser.add_argument("--source", default="suckhoedoisong", help="Mã nguồn (code)")
    parser.add_argument(
        "--date",
        help="Ngày scrape YYYY-MM-DD (cả ngày 00:00:00 → 23:59:59). "
        "Bỏ qua nếu dùng --from/--to.",
    )
    parser.add_argument(
        "--from",
        dest="from_time",
        help='Thời gian bắt đầu, vd: "2026-07-12 00:00:00"',
    )
    parser.add_argument(
        "--to",
        dest="to_time",
        help='Thời gian kết thúc, vd: "2026-07-12 23:59:59"',
    )
    parser.add_argument("--delay", type=float, default=0.5, help="Delay giữa requests (s)")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Không xếp hàng index Qdrant sau scrape",
    )
    parser.add_argument(
        "--no-wait-index",
        action="store_true",
        help="Không chờ index nền xong (mặc định CLI sẽ chờ để tránh thoát sớm)",
    )
    parser.add_argument("--export", help="Đường dẫn CSV export sau khi scrape")
    args = parser.parse_args(argv)

    db.init_db()

    if args.list_sources:
        for s in db.list_sources():
            flag = "ON" if s["enabled"] else "OFF"
            print(f"[{flag}] {s['id']}: {s['code']} — {s['name']} ({s['list_url']})")
        return 0

    if args.index_pending:
        from scraper.qdrant_indexer import index_pending_sync

        result = index_pending_sync()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["failed"] == 0 else 1

    if not args.from_time and not args.to_time and not args.date:
        parser.error('Cần --date YYYY-MM-DD hoặc --from/--to "YYYY-MM-DD HH:MM:SS"')

    if (args.from_time and not args.to_time) or (args.to_time and not args.from_time):
        parser.error("Cần cả --from và --to")

    def on_progress(_p, msg: str) -> None:
        print(msg)

    result = run_scrape_job(
        source_code=args.source,
        target_date=None if args.from_time else args.date,
        from_time=args.from_time,
        to_time=args.to_time,
        request_delay=args.delay,
        on_progress=on_progress,
        auto_index=not args.no_index,
    )

    if (
        not args.no_index
        and not args.no_wait_index
        and result.get("index_queued")
    ):
        from scraper.qdrant_indexer import wait_until_idle

        print("Đang chờ index Qdrant nền...")
        wait_until_idle()
        print("Index nền xong.")

    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "messages"},
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.export:
        n = export_articles_to_csv(args.export, source=args.source)
        print(f"Exported {n} rows -> {args.export}")

    return 0 if result["failed_count"] == 0 or result["success_count"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
