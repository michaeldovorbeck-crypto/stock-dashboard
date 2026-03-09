from __future__ import annotations

import pandas as pd
import streamlit as st

from src.help_ui import page_intro
from src.strategy_engine import top_etfs, top_leaders
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


def _top_value(df: pd.DataFrame, col: str, fallback="—"):
    if df.empty or col not in df.columns:
        return fallback
    return _safe(df.iloc[0].get(col), fallback)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        x = str(v).strip().upper()
        if not x or x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def _render_summary(etf_df: pd.DataFrame, leaders_df: pd.DataFrame) -> None:
    st.markdown("### Strategy-overblik")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top ETF", _top_value(etf_df, "Ticker"))
    c2.metric("Top ETF action", _top_value(etf_df, "Action"))
    c3.metric("Top leader", _top_value(leaders_df, "Ticker"))
    c4.metric("Top leader action", _top_value(leaders_df, "Action"))


def _render_badges(etf_df: pd.DataFrame, leaders_df: pd.DataFrame) -> None:
    badges: list[tuple[str, str]] = []

    if not etf_df.empty:
        etf_ticker = _top_value(etf_df, "Ticker")
        etf_action = str(_top_value(etf_df, "Action", "—")).upper()
        badges.append(
            (
                f"ETF: {etf_ticker} ({etf_action})",
                "buy" if etf_action == "BUY" else "sell" if etf_action == "SELL" else "hold",
            )
        )

    if not leaders_df.empty:
        leader_ticker = _top_value(leaders_df, "Ticker")
        leader_action = str(_top_value(leaders_df, "Action", "—")).upper()
        badges.append(
            (
                f"Leader: {leader_ticker} ({leader_action})",
                "buy" if leader_action == "BUY" else "sell" if leader_action == "SELL" else "hold",
            )
        )

    if badges:
        render_badges(badges)


def _render_interpretation(etf_df: pd.DataFrame, leaders_df: pd.DataFrame) -> None:
    bullets = []

    if not etf_df.empty:
        bullets.append(f"Stærkeste ETF lige nu er {_top_value(etf_df, 'Ticker')}")

    if not leaders_df.empty:
        bullets.append(f"Stærkeste leader lige nu er {_top_value(leaders_df, 'Ticker')}")

    top_etf_action = str(_top_value(etf_df, "Action", "")).upper()
    top_leader_action = str(_top_value(leaders_df, "Action", "")).upper()

    if top_etf_action == "BUY":
        bullets.append("ETF-signaler peger bullish")
    elif top_etf_action == "SELL":
        bullets.append("ETF-signaler peger defensivt")

    if top_leader_action == "BUY":
        bullets.append("Leaders viser fortsat relativ styrke")
    elif top_leader_action == "SELL":
        bullets.append("Leaders viser svaghed")

    if not bullets:
        bullets = ["Ingen tydelige strategy-signaler endnu"]

    st.markdown("### Hvad betyder det?")
    st.info(" / ".join(bullets[:4]))


def _render_top_etf_cards(etf_df: pd.DataFrame) -> None:
    st.markdown("### Top ETF'er")

    if etf_df.empty:
        st.info("Ingen ETF-data endnu.")
        return

    preview_cols = [c for c in ["Ticker", "Strategy Score", "Action"] if c in etf_df.columns]
    if preview_cols:
        st.dataframe(etf_df[preview_cols].head(10), use_container_width=True, hide_index=True)
    else:
        st.dataframe(etf_df.head(10), use_container_width=True, hide_index=True)

    if "Ticker" in etf_df.columns:
        tickers = _dedupe_preserve_order(etf_df["Ticker"].dropna().astype(str).head(10).tolist())[:6]

        if tickers:
            st.markdown("### Hurtigvalg ETF")
            cols = st.columns(len(tickers))
            for i, ticker in enumerate(tickers):
                if cols[i].button(ticker, key=f"strategy_etf_{i}_{ticker}"):
                    st.session_state["analysis_selected_ticker"] = ticker
                    st.success(f"Valgt til Analyse: {ticker}")


def _render_top_leader_cards(leaders_df: pd.DataFrame) -> None:
    st.markdown("### Top leaders")

    if leaders_df.empty:
        st.info("Ingen leader-data endnu.")
        return

    preview_cols = [c for c in ["Ticker", "Strategy Score", "Action"] if c in leaders_df.columns]
    if preview_cols:
        st.dataframe(leaders_df[preview_cols].head(10), use_container_width=True, hide_index=True)
    else:
        st.dataframe(leaders_df.head(10), use_container_width=True, hide_index=True)

    if "Ticker" in leaders_df.columns:
        tickers = _dedupe_preserve_order(leaders_df["Ticker"].dropna().astype(str).head(10).tolist())[:6]

        if tickers:
            st.markdown("### Hurtigvalg leaders")
            cols = st.columns(len(tickers))
            for i, ticker in enumerate(tickers):
                if cols[i].button(ticker, key=f"strategy_leader_{i}_{ticker}"):
                    st.session_state["analysis_selected_ticker"] = ticker
                    st.success(f"Valgt til Analyse: {ticker}")


def _render_detail_tables(etf_df: pd.DataFrame, leaders_df: pd.DataFrame) -> None:
    with st.expander("Vis alle strategy-tabeller"):
        st.markdown("**ETF'er**")
        if etf_df.empty:
            st.info("Ingen ETF-data.")
        else:
            st.dataframe(etf_df, use_container_width=True, hide_index=True)

        st.markdown("**Leaders**")
        if leaders_df.empty:
            st.info("Ingen leader-data.")
        else:
            st.dataframe(leaders_df, use_container_width=True, hide_index=True)


def render_strategy_4() -> None:
    apply_pro_style()

    st.markdown("## Strategy 4.0 PRO")
    page_intro("strategy")

    etf_df = top_etfs(15)
    leaders_df = top_leaders(20)

    _render_summary(etf_df, leaders_df)
    _render_badges(etf_df, leaders_df)

    left, right = st.columns([2, 1])

    with left:
        _render_interpretation(etf_df, leaders_df)

    with right:
        render_info_card(
            "Strategy-fokus",
            _top_value(etf_df, "Ticker"),
            "Brug strategy til at finde stærke ETF'er og leaders med relativ styrke",
        )

    c1, c2 = st.columns(2)

    with c1:
        _render_top_etf_cards(etf_df)

    with c2:
        _render_top_leader_cards(leaders_df)

    _render_detail_tables(etf_df, leaders_df)