from __future__ import annotations

import pandas as pd
import streamlit as st

from src.discovery_engine import (
    build_discovery_table,
    top_discovery_candidates,
    weakening_themes,
)
from src.help_ui import page_intro
from src.ui_style import apply_pro_style, render_badges, render_info_card


def _safe(v, fallback="—"):
    if v is None:
        return fallback
    try:
        if pd.isna(v):
            return fallback
    except Exception:
        pass
    txt = str(v).strip()
    return txt if txt else fallback


def _render_summary_cards(top_df: pd.DataFrame, weak_df: pd.DataFrame, disc_df: pd.DataFrame) -> None:
    st.markdown("### Discovery-overblik")

    top_theme = "—"
    if not top_df.empty and "Theme" in top_df.columns:
        top_theme = _safe(top_df.iloc[0].get("Theme"))

    top_stage = "—"
    if not top_df.empty and "Stage" in top_df.columns:
        top_stage = _safe(top_df.iloc[0].get("Stage"))

    weakening_theme = "—"
    if not weak_df.empty and "Theme" in weak_df.columns:
        weakening_theme = _safe(weak_df.iloc[0].get("Theme"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top discovery", top_theme)
    c2.metric("Top stage", top_stage)
    c3.metric("Weakening tema", weakening_theme)
    c4.metric("Discovery rows", len(disc_df))


def _render_badges(top_df: pd.DataFrame, weak_df: pd.DataFrame) -> None:
    badges: list[tuple[str, str]] = []

    if not top_df.empty and "Theme" in top_df.columns:
        badges.append((f"Stærkest: {_safe(top_df.iloc[0].get('Theme'))}", "buy"))

    if not top_df.empty and "Stage" in top_df.columns:
        badges.append((f"Stage: {_safe(top_df.iloc[0].get('Stage'))}", "hold"))

    if not weak_df.empty and "Theme" in weak_df.columns:
        badges.append((f"Svagere: {_safe(weak_df.iloc[0].get('Theme'))}", "sell"))

    if badges:
        render_badges(badges)


def _render_interpretation(top_df: pd.DataFrame, weak_df: pd.DataFrame) -> None:
    bullets = []

    if not top_df.empty:
        if "Theme" in top_df.columns:
            bullets.append(f"Stærkeste discovery-tema lige nu er {_safe(top_df.iloc[0].get('Theme'))}")
        if "Stage" in top_df.columns:
            bullets.append(f"Top discovery befinder sig i {_safe(top_df.iloc[0].get('Stage'))}-fase")

    if not weak_df.empty and "Theme" in weak_df.columns:
        bullets.append(f"Et svækkende tema lige nu er {_safe(weak_df.iloc[0].get('Theme'))}")

    if not bullets:
        bullets = [
            "Ingen tydelige discovery-signaler endnu",
            "Prøv at opdatere snapshot eller discovery-data",
        ]

    st.markdown("### Hvad betyder det?")
    st.info(" / ".join(bullets[:4]))


def _render_top_discovery(top_df: pd.DataFrame) -> None:
    st.markdown("### Top discovery candidates")

    if top_df.empty:
        st.info("Ingen discovery-kandidater endnu.")
        return

    cols = [c for c in ["Theme", "Discovery Score", "Stage"] if c in top_df.columns]
    if cols:
        st.dataframe(top_df[cols], use_container_width=True, hide_index=True)

    if "Theme" in top_df.columns:
        st.markdown("### Hurtigvalg")
        top_themes = top_df["Theme"].dropna().astype(str).head(6).tolist()
        if top_themes:
            btn_cols = st.columns(len(top_themes))
            for i, theme in enumerate(top_themes):
                if btn_cols[i].button(theme, key=f"disc_theme_{theme}"):
                    st.session_state["selected_theme"] = theme
                    st.success(f"Tema valgt: {theme}")


def _render_weakening_themes(weak_df: pd.DataFrame) -> None:
    st.markdown("### Weakening themes")

    if weak_df.empty:
        st.info("Ingen weakening themes endnu.")
        return

    cols = [c for c in ["Theme", "Discovery Score", "Stage"] if c in weak_df.columns]
    if cols:
        st.dataframe(weak_df[cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(weak_df, use_container_width=True, hide_index=True)


def _render_discovery_table(disc_df: pd.DataFrame) -> None:
    st.markdown("### Discovery table")

    if disc_df.empty:
        st.info("Ingen discovery-tabel endnu.")
        return

    st.dataframe(disc_df, use_container_width=True, hide_index=True)


def render_discovery_4() -> None:
    apply_pro_style()

    st.markdown("## Discovery 4.0 PRO")
    page_intro("discovery")

    top_df = top_discovery_candidates(10)
    weak_df = weakening_themes(10)
    disc_df = build_discovery_table()

    _render_summary_cards(top_df, weak_df, disc_df)
    _render_badges(top_df, weak_df)

    left, right = st.columns([2, 1])

    with left:
        _render_interpretation(top_df, weak_df)

    with right:
        render_info_card(
            "Discovery-fokus",
            _safe(top_df.iloc[0].get("Theme")) if not top_df.empty and "Theme" in top_df.columns else "—",
            "Brug discovery til at finde nye trends og stærke emerging temaer",
        )

    top_left, top_right = st.columns(2)

    with top_left:
        _render_top_discovery(top_df)

    with top_right:
        _render_weakening_themes(weak_df)

    _render_discovery_table(disc_df)