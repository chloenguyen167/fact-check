"""Trang lịch sử / thống kê các job scrape."""

from __future__ import annotations

import streamlit as st

from scraper import db
from ui.theme import mark_admin_page

mark_admin_page()

st.markdown(
    """
    <div class="page-hero">
      <h1>📋 Lịch sử thu thập</h1>
      <p>Theo dõi các lần thu thập bài viết và trạng thái nạp vào kho tài liệu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

db.init_db()
stats = db.article_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng bài trong DB", stats["total"])
c2.metric("Success", stats["by_status"].get("success", 0))
c3.metric("Qdrant indexed", stats.get("by_index", {}).get("indexed", 0))
c4.metric(
    "Index pending/fail",
    stats.get("by_index", {}).get("pending", 0) + stats.get("by_index", {}).get("failed", 0),
)

if stats["by_source"]:
    st.subheader("Theo nguồn")
    st.dataframe(stats["by_source"], use_container_width=True)

st.subheader("Các lần chạy gần đây")
jobs = db.list_jobs(limit=100)
if not jobs:
    st.info("Chưa có job nào.")
    st.stop()

st.dataframe(
    [
        {
            "job_id": j["id"],
            "nguồn": j["source_name"],
            "ngày": j["target_date"],
            "từ": j.get("from_time"),
            "đến": j.get("to_time"),
            "status": j["status"],
            "found": j["total_found"],
            "ok": j["success_count"],
            "fail": j["failed_count"],
            "skip": j["skipped_count"],
            "bắt đầu": j["started_at"],
            "kết thúc": j["finished_at"],
            "lỗi": j["error_message"],
        }
        for j in jobs
    ],
    use_container_width=True,
)

job_ids = [j["id"] for j in jobs]
selected = st.selectbox("Xem chi tiết job", job_ids)
if selected:
    job = db.get_job(selected)
    st.write(job)
    arts = db.list_articles(job_id=selected, limit=500)
    st.write(f"**{len(arts)} bài** gắn với job #{selected}")
    if arts:
        st.dataframe(
            [
                {
                    "id": a["id"],
                    "title": a["title"],
                    "publish_time": a["publish_time"],
                    "status": a["scrape_status"],
                    "link": a["link"],
                    "error": a["error_message"],
                }
                for a in arts
            ],
            use_container_width=True,
        )
