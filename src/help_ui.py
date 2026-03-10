from __future__ import annotations

import streamlit as st

from src.help_texts import HELP_TEXT


PAGE_MAP = {
    "analysis": "analysis_page_intro",
    "screening": "screening_intro",
    "quant": "quant_intro",
    "macro": "macro_intro",
    "themes": "themes_intro",
    "discovery": "discovery_intro",
    "strategy": "strategy_intro",
    "portfolio": "portfolio_intro",
}


def page_intro(page_key: str) -> None:
    help_key = PAGE_MAP.get(page_key)
    if not help_key:
        return

    text = HELP_TEXT.get(help_key, "")
    if text:
        st.info(text)


def global_help_expander() -> None:
    with st.expander("Sådan bruges dashboardet"):
        intro = HELP_TEXT.get("dashboard_intro", "")
        tabs = HELP_TEXT.get("dashboard_tabs", "")

        if intro:
            st.markdown(intro)

        if tabs:
            st.markdown(tabs)


def render_dashboard_guide_sidebar() -> None:
    st.markdown("## Dashboard guide")

    intro = HELP_TEXT.get("dashboard_intro", "")
    if intro:
        st.markdown(intro)

    with st.expander("Sådan er dashboardet opbygget", expanded=False):
        tabs = HELP_TEXT.get("dashboard_tabs", "")
        if tabs:
            st.markdown(tabs)