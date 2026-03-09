from __future__ import annotations

import pandas as pd
import streamlit as st

from src.compare_engine import build_compare_chart_df, build_compare_table


def build_normalized_compare_df(compare_chart_df: pd.DataFrame) -> pd.DataFrame:
    if compare_chart_df.empty or "Date" not in compare_chart_df.columns:
        return pd.DataFrame()

    work = compare_chart_df.copy()
    value_cols = [c for c in work.columns if c != "Date"]
    if not value_cols:
        return pd.DataFrame()

    for col in value_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    out = work[["Date"]].copy()

    for col in value_cols:
        s = work[col].dropna()
        if s.empty:
            continue
        first = s.iloc[0]
        if pd.isna(first) or first == 0:
            continue
        out[col] = (work[col] / first) * 100.0

    keep = ["Date"] + [c for c in out.columns if c != "Date"]
    return out[keep].dropna(how="all", subset=[c for c in out.columns if c != "Date"])


def render_compare_block(compare_tickers: list[str], years: int) -> None:
    compare_table_df = build_compare_table(compare_tickers, years=years)
    compare_chart_df = build_compare_chart_df(compare_tickers, years=min(years, 3))
    normalized_df = build_normalized_compare_df(compare_chart_df)

    view_mode = st.radio(
        "Compare-visning",
        ["Tabel", "Prisudvikling", "Normaliseret (100-base)"],
        horizontal=True,
        key="compare_view_mode",
    )

    if view_mode == "Tabel":
        if compare_table_df.empty:
            st.info("Ingen compare-data.")
        else:
            st.dataframe(compare_table_df, use_container_width=True, hide_index=True)

    elif view_mode == "Prisudvikling":
        if compare_chart_df.empty or "Date" not in compare_chart_df.columns:
            st.info("Ingen chart-data.")
        else:
            st.line_chart(compare_chart_df.set_index("Date"))

    else:
        if normalized_df.empty or "Date" not in normalized_df.columns:
            st.info("Ingen normaliseret chart-data.")
        else:
            st.line_chart(normalized_df.set_index("Date"))

    if not compare_table_df.empty:
        with st.expander("Vis compare-tabel"):
            st.dataframe(compare_table_df, use_container_width=True, hide_index=True)