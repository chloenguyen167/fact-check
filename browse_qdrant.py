"""
Xem dữ liệu Qdrant local (embedded mode).

Chạy:
    streamlit run browse_qdrant.py
"""

import streamlit as st
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_PATH
from ingest import get_collection_name
from qdrant_store import get_qdrant_client


def scroll_points(client, collection, limit=20, offset=None, article_id=None):
  kwargs = {
      "collection_name": collection,
      "limit": limit,
      "offset": offset,
      "with_payload": True,
      "with_vectors": False,
  }
  if article_id:
      kwargs["scroll_filter"] = Filter(
          must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
      )
  return client.scroll(**kwargs)


def count_unique_articles(client, collection, sample_size=2000):
  seen = set()
  offset = None
  while True:
      points, offset = client.scroll(
          collection_name=collection,
          limit=500,
          offset=offset,
          with_payload=["article_id"],
          with_vectors=False,
      )
      for point in points:
          aid = point.payload.get("article_id")
          if aid:
              seen.add(aid)
      if offset is None or len(seen) >= sample_size:
          break
  return len(seen)


def render_point(point):
  payload = point.payload or {}
  st.markdown(f"**ID:** `{point.id}` · **article_id:** `{payload.get('article_id', '—')}`")
  st.caption(f"Nguồn: {payload.get('source', '—')} · {payload.get('title', '—')}")
  text = payload.get("text", "")
  st.text_area("Nội dung chunk", text, height=180, disabled=True, label_visibility="collapsed")


st.set_page_config(page_title="Qdrant Browser", page_icon="🗄️", layout="wide")
st.title("Qdrant Local Browser")
st.caption(f"Path: `{QDRANT_PATH}`")

client = get_qdrant_client()
collection = get_collection_name()

try:
    info = client.get_collection(collection)
except Exception as exc:
    st.error(f"Không đọc được collection `{collection}`: {exc}")
    st.info("Chạy `python ingest.py` để tạo index trước.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Collection", collection)
col2.metric("Tổng chunks", f"{info.points_count:,}")
with col3:
    if st.button("Đếm bài unique", use_container_width=True):
        with st.spinner("Đang quét..."):
            st.session_state["unique_articles"] = count_unique_articles(client, collection)
if "unique_articles" in st.session_state:
    col3.metric("Bài unique (ước lượng)", st.session_state["unique_articles"])

tab_browse, tab_article, tab_search = st.tabs([
    "Duyệt chunks",
    "Theo article_id",
    "Tìm kiếm ngữ nghĩa",
])

if "scroll_offset" not in st.session_state:
    st.session_state.scroll_offset = None

with tab_browse:
    page_size = st.slider("Số chunk / trang", 5, 50, 10, key="browse_page_size")
    if st.button("Tải từ đầu", key="browse_reset"):
        st.session_state.scroll_offset = None
        st.rerun()

    points, next_offset = scroll_points(
        client, collection,
        limit=page_size,
        offset=st.session_state.scroll_offset,
    )
    st.write(
        f"Hiển thị {len(points)} chunk"
        + (f" · offset={st.session_state.scroll_offset}" if st.session_state.scroll_offset else "")
        + (" · còn trang sau" if next_offset else " · hết")
    )
    for point in points:
        with st.expander(
            f"`{point.payload.get('article_id', point.id)}` — "
            f"{(point.payload.get('title') or '')[:60]}",
            expanded=False,
        ):
            render_point(point)

    if next_offset and st.button("Trang tiếp ▶", key="browse_more"):
        st.session_state.scroll_offset = next_offset
        st.rerun()

with tab_article:
    article_id = st.text_input("article_id", placeholder="vd: skds_357, umc_417")
    limit = st.number_input("Số chunk tối đa", 1, 100, 20, key="article_limit")
    if article_id.strip():
        points, _ = scroll_points(
            client, collection,
            limit=int(limit),
            article_id=article_id.strip(),
        )
        if not points:
            st.warning(f"Không có chunk nào với article_id = `{article_id}`")
        else:
            st.success(f"Tìm thấy {len(points)} chunk")
            for i, point in enumerate(points, 1):
                st.markdown(f"#### Chunk {i}")
                render_point(point)

with tab_search:
    st.caption("Dùng cùng embedding BAAI/bge-m3 như hệ thống retrieval.")
    query = st.text_input("Câu truy vấn", placeholder="viêm cột sống dính khớp")
    top_k = st.slider("Top K", 1, 20, 5, key="search_top_k")
    if query.strip():
        from retrieval import retrieve_with_metadata

        with st.spinner("Đang tìm..."):
            hits = retrieve_with_metadata(query.strip(), limit=top_k)
        for i, hit in enumerate(hits, 1):
            with st.expander(
                f"#{i} `{hit.get('article_id')}` — {(hit.get('title') or '')[:60]}",
                expanded=i == 1,
            ):
                st.caption(f"Nguồn: {hit.get('source', '—')}")
                st.text_area(
                    "text",
                    hit.get("text", ""),
                    height=180,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"search_{i}",
                )
