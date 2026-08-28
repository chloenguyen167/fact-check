"""Điểm vào Streamlit — điều hướng & theme chung."""

import streamlit as st

from ui.theme import inject_theme, setup_sidebar_brand

st.set_page_config(
    page_title="MedFact Check",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
setup_sidebar_brand()

nav = st.navigation(
    {
        "Kiểm chứng": [
            st.Page(
                "pages/kiem_chung.py",
                title="Kiểm chứng thông tin",
                icon="🩺",
                default=True,
            ),
        ],
        "Quản lý dữ liệu": [
            st.Page(
                "pages/1_Thu_thap_du_lieu.py",
                title="Thu thập bài viết",
                icon="📰",
            ),
            st.Page(
                "pages/2_Quan_ly_nguon.py",
                title="Nguồn tin tức",
                icon="🗂️",
            ),
            st.Page(
                "pages/3_Lich_su_scrape.py",
                title="Lịch sử thu thập",
                icon="📋",
            ),
            st.Page(
                "pages/4_Bai_da_scrape.py",
                title="Kho bài viết",
                icon="📚",
            ),
        ],
    }
)

nav.run()
