"""Trang quản lý bài báo đã scrape."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.theme import mark_admin_page

mark_admin_page()

_load = st.progress(0, text="Đang tải trang...")
_load.progress(30, text="Khởi tạo database...")
from scraper import db
from scraper.runner import export_articles_to_csv

_load.progress(100, text="Sẵn sàng")
_load.empty()

st.markdown(
    """
    <div class="page-hero">
      <h1>📚 Kho bài viết</h1>
      <p>Xem, lọc và quản lý các bài đã thu thập. Có thể nạp lại vào kho tài liệu khi cần.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

db.init_db()

sources = db.list_sources()
source_codes = ["(tất cả)"] + [s["code"] for s in sources]
status_opts = ["(tất cả)", "success", "failed"]
index_opts = ["(tất cả)", "indexed", "pending", "failed", "indexing"]

f1, f2, f3, f4 = st.columns(4)
with f1:
    src = st.selectbox("Nguồn", source_codes)
with f2:
    status = st.selectbox("Trạng thái scrape", status_opts)
with f3:
    index_status = st.selectbox("Trạng thái Qdrant", index_opts)
with f4:
    limit = st.number_input("Giới hạn", min_value=10, max_value=5000, value=200, step=10)

articles = db.list_articles(
    source=None if src == "(tất cả)" else src,
    status=None if status == "(tất cả)" else status,
    index_status=None if index_status == "(tất cả)" else index_status,
    limit=int(limit),
)

st.write(f"Hiển thị **{len(articles)}** bài")

if articles:
    st.dataframe(
        [
            {
                "id": a["id"],
                "title": a["title"],
                "publish_time": a["publish_time"],
                "source": a["source"],
                "author": a["author"],
                "scrape": a["scrape_status"],
                "index_status": a.get("index_status"),
                "index_chunks": a.get("index_chunks"),
                "indexed_at": a.get("indexed_at"),
                "scraped_at": a["scraped_at"],
                "link": a["link"],
                "index_error": a.get("index_error"),
            }
            for a in articles
        ],
        use_container_width=True,
        height=420,
    )

    selected_id = st.selectbox("Xem nội dung chi tiết", [a["id"] for a in articles])
    detail = next(a for a in articles if a["id"] == selected_id)
    st.markdown(f"### {detail['title']}")
    st.caption(
        f"{detail.get('publish_time')} · {detail.get('author') or 'N/A'} · "
        f"Qdrant: {detail.get('index_status')} / {detail.get('index_chunks') or 0} chunks · "
        f"[{detail.get('link')}]({detail.get('link')})"
    )
    if detail.get("tags"):
        st.write(f"**Tags:** {detail['tags']}")
    if detail.get("related_one_news"):
        st.write(f"**Related:** {detail['related_one_news']}")
    if detail.get("error_message"):
        st.error(detail["error_message"])
    if detail.get("index_error"):
        st.warning(f"Index error: {detail['index_error']}")
    st.text_area("Content", detail.get("content") or "", height=280)

st.divider()
idx1, idx2, idx3 = st.columns(3)
with idx1:
    if st.button("🔄 Index pending/failed (bỏ qua đã nạp)"):
        from scraper.qdrant_indexer import enqueue_pending, queue_size

        n = enqueue_pending()
        st.success(f"Đã xếp {n} bài. Queue: {queue_size()}")
with idx2:
    if articles and st.button("♻️ Reset + nạp lại bài đang xem"):
        from scraper.qdrant_indexer import queue_size, reindex_articles

        n = reindex_articles([int(selected_id)])
        st.success(f"Reset & xếp lại {n} bài. Queue: {queue_size()}")
with idx3:
    st.caption(
        "`index_chunks` > 0 ⇒ đã nạp. Auto-index skip. "
        "Nút reset xóa marker rồi nạp lại."
    )

export_path = st.text_input(
    "Export CSV",
    value=str(Path("data") / "articles_scraped.csv"),
)
if st.button("Export bài success ra CSV"):
    n = export_articles_to_csv(
        export_path,
        source=None if src == "(tất cả)" else src,
        status="success",
    )
    st.success(f"Đã export {n} dòng → {export_path}")
