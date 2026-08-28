"""Trang quản lý các nguồn báo sẽ scrape."""

from __future__ import annotations

import json

import streamlit as st

from scraper import db
from scraper.sources import available_scraper_keys
from ui.theme import mark_admin_page

mark_admin_page()

st.markdown(
    """
    <div class="page-hero">
      <h1>🗂️ Nguồn tin tức</h1>
      <p>Thêm, sửa hoặc bật/tắt các trang báo dùng để thu thập tài liệu y khoa.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

db.init_db()

sources = db.list_sources()
if sources:
    st.subheader("Danh sách nguồn")
    for s in sources:
        with st.expander(f"{'✅' if s['enabled'] else '⏸️'} {s['name']} — `{s['code']}`", expanded=False):
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.write(f"**URL chuyên mục:** {s['list_url']}")
            c2.write(f"**Scraper:** `{s['scraper_key']}` · **Base:** {s['base_url']}")
            c3.write(f"**ID:** {s['id']}")
            if s.get("notes"):
                st.caption(s["notes"])
            st.code(s.get("config_json") or "{}", language="json")

            b1, b2, b3 = st.columns(3)
            if b1.button("Bật" if not s["enabled"] else "Tắt", key=f"toggle_{s['id']}"):
                db.set_source_enabled(s["id"], not bool(s["enabled"]))
                st.rerun()
            if b2.button("Sửa", key=f"edit_{s['id']}"):
                st.session_state["edit_source_id"] = s["id"]
                st.rerun()
            if b3.button("Xóa", key=f"del_{s['id']}"):
                db.delete_source(s["id"])
                st.rerun()
else:
    st.info("Chưa có nguồn nào.")

st.divider()
edit_id = st.session_state.get("edit_source_id")
editing = db.get_source(edit_id) if edit_id else None
st.subheader("Sửa nguồn" if editing else "Thêm nguồn mới")

keys = available_scraper_keys()
default = editing or {
    "code": "suckhoedoisong",
    "name": "Báo Sức khỏe & Đời sống",
    "base_url": "https://suckhoedoisong.vn",
    "list_url": "https://suckhoedoisong.vn/y-te.htm",
    "scraper_key": "skds",
    "enabled": 1,
    "config_json": "{}",
    "notes": "",
}

with st.form("source_form"):
    code = st.text_input("Mã nguồn (code)", value=default.get("code") or "")
    name = st.text_input("Tên hiển thị", value=default.get("name") or "")
    base_url = st.text_input("Base URL", value=default.get("base_url") or "")
    list_url = st.text_input("URL chuyên mục / list", value=default.get("list_url") or "")
    scraper_key = st.selectbox(
        "Scraper key",
        keys,
        index=keys.index(default["scraper_key"]) if default.get("scraper_key") in keys else 0,
    )
    enabled = st.checkbox("Bật nguồn", value=bool(default.get("enabled", 1)))
    config_json = st.text_area(
        "Config JSON (vd: {\"zone_id\": 1691474, \"max_pages\": 40})",
        value=default.get("config_json") or "{}",
        height=100,
    )
    notes = st.text_area("Ghi chú", value=default.get("notes") or "", height=80)
    submitted = st.form_submit_button("Lưu")

if submitted:
    try:
        config = json.loads(config_json or "{}")
        db.upsert_source(
            source_id=edit_id,
            code=code.strip(),
            name=name.strip(),
            base_url=base_url.strip(),
            list_url=list_url.strip(),
            scraper_key=scraper_key,
            enabled=enabled,
            config=config,
            notes=notes.strip(),
        )
        st.session_state.pop("edit_source_id", None)
        st.success("Đã lưu nguồn báo.")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Lỗi lưu nguồn: {exc}")

if editing and st.button("Hủy sửa"):
    st.session_state.pop("edit_source_id", None)
    st.rerun()

st.divider()
st.markdown(
    """
**Ghi chú mở rộng**
- Hiện có scraper: `skds` (Sức khỏe & Đời sống — chuyên mục Y tế).
- Để thêm báo khác: implement subclass trong `scraper/sources/` và đăng ký trong `SCRAPER_REGISTRY`.
- Tự động scrape mỗi ngày: sẽ bổ sung (cron / APScheduler) sau; hiện tại chạy thủ công theo ngày.
"""
)
