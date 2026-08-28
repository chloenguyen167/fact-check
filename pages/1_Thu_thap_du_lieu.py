"""Trang Streamlit: chạy scrape tin theo khoảng thời gian xuất bản."""

from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta

import streamlit as st

from ui.theme import mark_admin_page

mark_admin_page()

_load = st.progress(0, text="Đang tải trang...")
_load.progress(20, text="Khởi tạo database...")
from scraper import db
from scraper.runner import run_scrape_job

_load.progress(70, text="Đọc danh sách nguồn báo...")
db.init_db()
sources = db.list_sources(enabled_only=True)
_load.progress(100, text="Sẵn sàng")
_load.empty()

st.markdown(
    """
    <div class="page-hero">
      <h1>📰 Thu thập bài viết</h1>
      <p>Chọn nguồn tin và khoảng thời gian. Bài mới sẽ được lưu và tự nạp vào kho tài liệu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not sources:
    st.warning("Chưa có nguồn báo nào được bật. Vào trang **Quản lý nguồn báo** để thêm/bật nguồn.")
    st.stop()

today = date.today()
END_OF_DAY = dtime(23, 59, 59)


def _init_range_state() -> None:
    if "range_from_date" not in st.session_state:
        st.session_state["range_from_date"] = today
    if "range_to_date" not in st.session_state:
        st.session_state["range_to_date"] = today
    if "range_from_clock" not in st.session_state:
        st.session_state["range_from_clock"] = dtime.min
    if "range_to_clock" not in st.session_state:
        st.session_state["range_to_clock"] = END_OF_DAY
    if "range_full_day" not in st.session_state:
        st.session_state["range_full_day"] = True


def _apply_preset(start: date, end: date, *, full_day: bool = True) -> None:
    st.session_state["range_from_date"] = start
    st.session_state["range_to_date"] = end
    st.session_state["range_full_day"] = full_day
    if full_day:
        st.session_state["range_from_clock"] = dtime.min
        st.session_state["range_to_clock"] = END_OF_DAY


def _build_range() -> tuple[str, str]:
    from_dt = datetime.combine(
        st.session_state["range_from_date"],
        dtime.min if st.session_state["range_full_day"] else st.session_state["range_from_clock"],
    )
    to_dt = datetime.combine(
        st.session_state["range_to_date"],
        END_OF_DAY if st.session_state["range_full_day"] else st.session_state["range_to_clock"],
    )
    return from_dt.strftime("%Y-%m-%d %H:%M:%S"), to_dt.strftime("%Y-%m-%d %H:%M:%S")


_init_range_state()

col1, col2 = st.columns([2, 1])
with col1:
    source_labels = {f"{s['name']} ({s['code']})": s for s in sources}
    selected_label = st.selectbox("Nguồn báo", list(source_labels.keys()))
    source = source_labels[selected_label]
with col2:
    delay = st.number_input("Delay (giây)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

st.subheader("Thời gian xuất bản")
preset_cols = st.columns(4)
presets = [
    ("Hôm nay", lambda: _apply_preset(today, today)),
    ("Hôm qua", lambda: _apply_preset(today - timedelta(days=1), today - timedelta(days=1))),
    (
        "7 ngày qua",
        lambda: _apply_preset(today - timedelta(days=6), today),
    ),
    (
        "Tháng này",
        lambda: _apply_preset(today.replace(day=1), today),
    ),
]
for col, (label, apply_fn) in zip(preset_cols, presets, strict=True):
    if col.button(label, use_container_width=True):
        apply_fn()
        st.rerun()

full_day = st.checkbox(
    "Cả ngày (00:00:00 → 23:59:59)",
    key="range_full_day",
    help="Bật để chỉ cần chọn ngày, không cần nhập giờ.",
)

date_cols = st.columns(2)
with date_cols[0]:
    st.date_input("Từ ngày", key="range_from_date")
with date_cols[1]:
    st.date_input("Đến ngày", key="range_to_date")

if not full_day:
    time_cols = st.columns(2)
    with time_cols[0]:
        st.time_input("Từ giờ", key="range_from_clock", step=timedelta(minutes=1))
    with time_cols[1]:
        st.time_input("Đến giờ", key="range_to_clock", step=timedelta(minutes=1))

from_time, to_time = _build_range()
if datetime.strptime(to_time, "%Y-%m-%d %H:%M:%S") < datetime.strptime(
    from_time, "%Y-%m-%d %H:%M:%S"
):
    st.error("Ngày/giờ kết thúc phải sau hoặc bằng thời điểm bắt đầu.")
else:
    st.info(f"**Khoảng scrape:** `{from_time}` → `{to_time}`")

st.caption(f"URL chuyên mục: {source['list_url']}")

log_box = st.empty()
run = st.button("🚀 Bắt đầu scrape", type="primary", use_container_width=False)

if run:
    if datetime.strptime(to_time, "%Y-%m-%d %H:%M:%S") < datetime.strptime(
        from_time, "%Y-%m-%d %H:%M:%S"
    ):
        st.error("Không thể scrape: khoảng thời gian không hợp lệ.")
        st.stop()

    logs: list[str] = []
    progress_bar = st.progress(0.0, text="Đang scrape...")

    def on_progress(p, msg: str) -> None:
        logs.append(msg)
        log_box.code("\n".join(logs[-80:]), language="text")
        progress_bar.progress(min(0.95, 0.05 + p.found * 0.02), text=msg[:80])

    try:
        with st.spinner("Đang thu thập..."):
            result = run_scrape_job(
                source_id=source["id"],
                from_time=from_time,
                to_time=to_time,
                request_delay=float(delay),
                on_progress=on_progress,
            )
        progress_bar.progress(1.0, text="Hoàn tất")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tìm thấy", result["total_found"])
        m2.metric("Thành công", result["success_count"])
        m3.metric("Thất bại", result["failed_count"])
        m4.metric("Bỏ qua", result["skipped_count"])
        st.success(
            f"Job #{result['job_id']} hoàn tất "
            f"({result['from_time']} → {result['to_time']}). "
            f"Đã xếp {result.get('index_queued', 0)} bài vào index Qdrant (nền). "
            "Xem trạng thái ở **Bài đã scrape** / **Lịch sử scrape**."
        )
    except Exception as exc:  # noqa: BLE001
        progress_bar.empty()
        st.error(f"Scrape thất bại: {exc}")
