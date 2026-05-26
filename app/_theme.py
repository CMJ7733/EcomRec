"""Streamlit 主题注入工具。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def apply_theme() -> None:
    """加载并注入统一主题 CSS。"""
    css_path = Path(__file__).parent / "static" / "theme.css"
    if not css_path.exists():
        return

    css_text = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
