"""Trang chính: kiểm chứng thông tin y tế."""

from __future__ import annotations

import html
import random
import time

import streamlit as st

from claim_cache import clear_cache
from guard import check_claim

EXAMPLES = [
    "Ăn nhiều muối gây cao huyết áp",
    "Vitamin C chữa khỏi cảm cúm",
    "Hút thuốc lá làm tăng nguy cơ bệnh mạch vành",
    "Uống aspirin hàng ngày ngăn ngừa đột quỵ",
]

# Thời gian loading tối thiểu khi cache hit (demo UX)
_CACHE_HIT_DELAY = (1.0, 2.0)


def _verdict_view(verdict: str, score):
    v = (verdict or "").upper()
    if "ĐÚNG" in v:
        return {
            "bg": "#dff7e8",
            "fg": "#166534",
            "border": "#86efac",
            "title": "Có thể đúng",
            "plain": "Thông tin này phù hợp với tài liệu y khoa tìm được.",
            "icon": "✓",
        }
    if "SAI" in v:
        return {
            "bg": "#fee4e2",
            "fg": "#991b1b",
            "border": "#fca5a5",
            "title": "Có thể sai",
            "plain": "Thông tin này mâu thuẫn với tài liệu y khoa tìm được.",
            "icon": "✕",
        }
    return {
        "bg": "#fff7d6",
        "fg": "#92400e",
        "border": "#fcd34d",
        "title": "Chưa đủ căn cứ",
        "plain": "Chưa đủ tài liệu để khẳng định đúng hay sai.",
        "icon": "!",
    }


@st.cache_resource(show_spinner="Đang chuẩn bị công cụ kiểm chứng...")
def _get_verify_claim():
    from pipeline import verify_claim

    return verify_claim


def _pick_example(example: str) -> None:
    st.session_state["claim_input"] = example


def _render_result(result: dict) -> None:
    view = _verdict_view(result["verdict"], result.get("verdict_score"))
    score = result.get("verdict_score")
    score_html = (
        f'<div class="verdict-meta">Mức độ chắc chắn: {score * 100:.0f}%</div>'
        if score is not None
        else ""
    )
    st.markdown(
        f"""
        <div class="verdict-card" style="background:{view['bg']};color:{view['fg']};border-color:{view['border']};">
          <div class="verdict-label">{view['icon']} Kết luận</div>
          <div class="verdict-main">{view['title']}</div>
          <div class="verdict-meta">{view['plain']}</div>
          {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Giải thích ngắn gọn</div>', unsafe_allow_html=True)
    explanation = (result.get("explanation") or "").strip()
    if explanation:
        st.markdown(
            f'<div class="explain-box">{html.escape(explanation)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="explain-box">Chưa có phần giải thích chi tiết cho lần kiểm chứng này.</div>',
            unsafe_allow_html=True,
        )

    sentences = result.get("key_sentences") or []
    if sentences:
        st.markdown(
            '<div class="section-title">Câu bằng chứng then chốt</div>',
            unsafe_allow_html=True,
        )
        tags = {"entailment": "ủng hộ", "contradiction": "mâu thuẫn", "neutral": "trung lập"}
        for sent in sentences[:3]:
            rel = max(tags, key=lambda k: sent.get(k, 0.0))
            st.markdown(
                f'<div class="evidence-box">{html.escape(sent["text"])}'
                f'<br><small>→ {tags[rel]} ({sent.get(rel, 0.0) * 100:.0f}%)</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Nguồn tham khảo</div>', unsafe_allow_html=True)
    docs = result.get("evidence_docs") or []
    if docs:
        for i, doc in enumerate(docs, start=1):
            rel = doc.get("score", 0.0)
            body = (doc.get("text") or "").strip()
            title = (doc.get("title") or "").strip()
            source = (doc.get("source") or "").strip()
            link = (doc.get("link") or "").strip()

            short = title if title and len(title) <= 72 else (title[:69] + "…" if title else f"Nguồn {i}")
            header = f"{short} · mức liên quan {float(rel):.2f}" if rel else short

            with st.expander(header, expanded=(i == 1)):
                meta = []
                if source:
                    meta.append(f"Nguồn: **{html.escape(source)}**")
                if link:
                    meta.append(f"[Đọc bài gốc]({link})")
                if meta:
                    st.markdown(" · ".join(meta))
                st.markdown(
                    f'<div class="evidence-box">{html.escape(body)}</div>',
                    unsafe_allow_html=True,
                )
    else:
        evidence = (result.get("evidence") or "").strip()
        if evidence:
            st.markdown(
                f'<div class="evidence-box">{html.escape(evidence)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Không tìm thấy đoạn tài liệu liên quan đủ rõ.")


def _run_verification(text: str) -> None:
    with st.spinner("Đang kiểm tra dạng câu..."):
        gate = check_claim(text)

    if not gate.valid:
        st.error("Câu này chưa phù hợp để kiểm chứng")
        st.info(gate.message)
        with st.expander("Gợi ý diễn đạt lại", expanded=True):
            st.markdown(
                "- Viết thành **khẳng định**, không hỏi\n"
                "- Tránh kể bệnh cá nhân (“tôi bị…”, “bác sĩ bảo tôi…”)\n"
                "- Tránh nhờ tư vấn (“giúp tôi…”, “hãy…”)\n"
                "- Nên có chủ đề y khoa rõ (bệnh, thuốc, thực phẩm, triệu chứng)"
            )
        return

    with st.spinner("Đang tìm tài liệu và phân tích..."):
        verify_claim = _get_verify_claim()
        result = verify_claim(text)
        if result.get("from_cache"):
            time.sleep(random.uniform(*_CACHE_HIT_DELAY))

    _render_result(result)


st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">🩺 Hỗ trợ kiểm chứng thông tin sức khỏe</div>
      <h1 class="hero-title">Thông tin này có đáng tin?</h1>
      <p class="hero-sub">
        Nhập một câu khẳng định về sức khỏe. Hệ thống sẽ đối chiếu với tài liệu y khoa
        và cho bạn biết thông tin đó có vẻ đúng, sai, hay chưa đủ căn cứ.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hint-card">
      <div class="hint-title">Cách viết câu dễ kiểm chứng</div>
      <p class="hint-body">
        Viết thành một khẳng định ngắn, rõ, không hỏi và không kể bệnh của riêng bạn.
        Ví dụ tốt: <em>“Ăn nhiều muối gây cao huyết áp”</em>.
        Ví dụ chưa phù hợp: <em>“Tôi bị cao huyết áp thì nên ăn gì?”</em>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "claim_input" not in st.session_state:
    st.session_state["claim_input"] = ""

st.text_area(
    "Nhập thông tin cần kiểm chứng",
    key="claim_input",
    placeholder="Ví dụ: Ăn nhiều muối gây cao huyết áp",
    height=120,
    label_visibility="collapsed",
)

st.caption("Gợi ý nhanh — bấm để điền sẵn:")
ex_cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    with ex_cols[i % 2]:
        st.button(
            example,
            key=f"ex_{i}",
            use_container_width=True,
            type="secondary",
            on_click=_pick_example,
            args=(example,),
        )

run = st.button("Kiểm chứng ngay", type="primary", use_container_width=True)

if run:
    text = (st.session_state.get("claim_input") or "").strip()
    if not text:
        st.warning("Bạn chưa nhập nội dung. Hãy viết một câu khẳng định về sức khỏe.")
    else:
        _run_verification(text)

with st.expander("Tuỳ chọn"):
    st.caption("Cache giúp demo nhanh hơn với câu đã kiểm chứng (tự hết hạn sau 2 giờ).")
    if st.button("Xóa cache", type="secondary"):
        removed = clear_cache()
        st.success(f"Đã xóa {removed} mục cache.")

st.markdown(
    """
    <div class="footer-note">
      Kết quả mang tính tham khảo, không thay thế tư vấn hoặc chẩn đoán của bác sĩ.
      Nếu bạn đang có triệu chứng, hãy đến cơ sở y tế để được khám trực tiếp.
    </div>
    """,
    unsafe_allow_html=True,
)
