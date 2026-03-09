from __future__ import annotations

import streamlit as st

from src.help_texts import GLOBAL_HELP, PAGE_HELP


def page_intro(page_key: str) -> None:
    text = PAGE_HELP.get(page_key)
    if text:
        st.info(text)


def global_help_expander() -> None:
    with st.expander("Sådan bruges dashboardet"):
        st.write(GLOBAL_HELP)