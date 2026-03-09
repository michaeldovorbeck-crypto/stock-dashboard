from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analyse_v4_view import render_analysis_4
from src.discovery_v4_view import render_discovery_4
from src.macro_v4_view import render_macro_4
from src.strategy_v4_view import render_strategy_4

from src.cache_engine import load_snapshot
from src.data_sources import load_universe_csv
from src.learning_engine import build_learning_summary
from src.precompute_engine import build_quant_snapshot_for_universe
from src.screening_engine import run_screen_on_universe, summarize_screen, top_theme_hits
from src.storage_engine import load_portfolio_positions, load_watchlist, save_portfolio_positions

from src.portfolio_engine import (
    ACCOUNT_TYPES,
    analyze_portfolio_positions,
    build_account_summary,
    build_rebalance_suggestions,
    build_theme_exposure,
)

from src.theme_engine import build_theme_rankings, theme_deep_dive
from src.theme_definitions import THEMES

from src.help_ui import global_help_expander, page_intro
from src.help_texts import HELP_TEXT


st.set_page_config(
    page_title="Stock Dashboard V4 PRO",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Stock Dashboard V4 PRO")
global_help_expander()

if "portfolio_positions" not in st.session_state:
    st.session_state["portfolio_positions"] = load_portfolio_positions()

if "analysis_selected_ticker" not in st.session_state:
    st.session_state["analysis_selected_ticker"] = "AAPL"

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = load_watchlist()

with st.sidebar:
    st.markdown("## Dashboard guide")
    st.write(
        """
Dette dashboard hjælper dig med:

• aktieanalyse  
• kvantitativ screening  
• discovery af nye trends  
• tema-investering  
• porteføljestyring
"""
    )

(
    tab_analysis,
    tab_screening,
    tab_quant,
    tab_macro,
    tab_themes,
    tab_discovery,
    tab_strategy,
    tab_portfolio,
) = st.tabs(
    [
        "Analyse",
        "Screening",
        "Quant",
        "Macro",
        "Tema",
        "Discovery",
        "Strategy",
        "Portefølje",
    ]
)

with tab_analysis:
    render_analysis_4()

with tab_screening:
    st.subheader("Screening engine")
    page_intro("screening")

    universe_file = st.text_input(
        "Universe-fil",
        "global_all.csv",
        help=HELP_TEXT["universe"],
        key="screen_universe_file",
    )

    universe_df, universe_status = load_universe_csv(universe_file)

    if universe_df.empty:
        st.warning(universe_status)
    else:
        st.caption(universe_status)

        c1, c2, c3 = st.columns(3)

        with c1:
            screen_years = st.slider(
                "Historik (år)",
                1,
                10,
                3,
                key="screen_years",
            )

        with c2:
            max_tickers = st.number_input(
                "Max tickers",
                min_value=10,
                max_value=20000,
                value=500,
                step=100,
                key="screen_max_tickers",
            )

        with c3:
            min_timing = st.slider(
                "Min timing score",
                0,
                100,
                40,
                key="screen_min_timing",
            )

        if st.button("Kør screening", type="primary", key="screen_run_button"):
            with st.spinner("Kører screening..."):
                screen_df, screen_status = run_screen_on_universe(
                    filename=universe_file,
                    years=screen_years,
                    max_tickers=int(max_tickers),
                    min_timing_score=float(min_timing),
                )

            st.write(screen_status)

            if not screen_df.empty:
                summary = summarize_screen(screen_df)
                theme_hits_df = top_theme_hits(screen_df, top_n=10)

                s1, s2, s3 = st.columns(3)
                s1.metric("Kandidater", summary.get("count"))
                s2.metric("BUY ratio %", summary.get("buy_ratio_pct"))
                s3.metric("Bullish ratio %", summary.get("bullish_ratio_pct"))

                st.dataframe(screen_df, use_container_width=True, hide_index=True)

                st.markdown("### Theme hits")
                st.dataframe(theme_hits_df, use_container_width=True, hide_index=True)

with tab_quant:
    st.subheader("Quant engine")
    page_intro("quant")

    q1, q2, q3 = st.columns(3)

    with q1:
        snapshot_universe = st.text_input(
            "Snapshot universe",
            "global_all.csv",
            key="quant_snapshot_universe",
        )

    with q2:
        snapshot_years = st.slider(
            "Historik (år)",
            1,
            10,
            3,
            key="quant_snapshot_years",
        )

    with q3:
        snapshot_max = st.number_input(
            "Max tickers",
            min_value=100,
            max_value=50000,
            value=5000,
            step=100,
            key="quant_snapshot_max",
        )

    if st.button("Byg quant snapshot", type="primary", key="quant_build_snapshot_button"):
        with st.spinner("Bygger snapshot..."):
            snap_df, snap_status = build_quant_snapshot_for_universe(
                filename=snapshot_universe,
                years=int(snapshot_years),
                max_tickers=int(snapshot_max),
            )

        st.write(snap_status)

        if not snap_df.empty:
            st.dataframe(snap_df.head(100), use_container_width=True, hide_index=True)

    snapshot_df = load_snapshot()

    if not snapshot_df.empty:
        st.markdown("### Snapshot screener")

        top_n = st.slider(
            "Top N",
            10,
            200,
            50,
            key="quant_top_n",
        )

        work = snapshot_df.copy()

        if "Quant Score" in work.columns and "Timing Score" in work.columns:
            work = work.sort_values(
                ["Quant Score", "Timing Score"],
                ascending=[False, False],
            )

        work = work.head(top_n)
        st.dataframe(work, use_container_width=True, hide_index=True)

    st.markdown("### Learning")

    learn = build_learning_summary()

    l1, l2, l3 = st.columns(3)
    l1.metric("Logs", learn.get("logs"))
    l2.metric("BUY logs", learn.get("buy_logs"))
    l3.metric("SELL logs", learn.get("sell_logs"))

with tab_macro:
    render_macro_4()

with tab_themes:
    st.subheader("Temaer")
    page_intro("themes")

    ranking_df = build_theme_rankings()
    if not ranking_df.empty:
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    selected_theme = st.selectbox(
        "Vælg tema",
        list(THEMES.keys()),
        key="themes_selected_theme",
    )

    deep = theme_deep_dive(selected_theme)

    if deep:
        st.write(deep.get("description", ""))

        members_df = deep.get("members_df", pd.DataFrame())
        if not members_df.empty:
            st.dataframe(members_df, use_container_width=True, hide_index=True)

with tab_discovery:
    render_discovery_4()

with tab_strategy:
    render_strategy_4()

with tab_portfolio:
    st.subheader("Portefølje")

    c1, c2, c3 = st.columns(3)

    with c1:
        pf_ticker = st.text_input(
            "Ticker",
            "MSFT",
            key="portfolio_ticker",
        )

    with c2:
        pf_amount = st.number_input(
            "Antal",
            min_value=0.0,
            max_value=100000.0,
            value=10.0,
            step=1.0,
            key="portfolio_amount",
        )

    with c3:
        pf_account = st.selectbox(
            "Konto",
            ACCOUNT_TYPES,
            key="portfolio_account",
        )

    if st.button("Tilføj position", key="portfolio_add_button"):
        if pf_ticker.strip():
            st.session_state["portfolio_positions"].append(
                {
                    "Ticker": pf_ticker.strip().upper(),
                    "Antal": float(pf_amount),
                    "Konto": pf_account,
                }
            )
            save_portfolio_positions(st.session_state["portfolio_positions"])
            st.success(f"{pf_ticker.strip().upper()} tilføjet")

    positions_df = pd.DataFrame(st.session_state["portfolio_positions"])

    if positions_df.empty:
        st.info("Porteføljen er tom.")
    else:
        analyzed_df = analyze_portfolio_positions(positions_df)

        st.dataframe(analyzed_df, use_container_width=True, hide_index=True)

        st.markdown("### Konto-overblik")
        account_summary_df = build_account_summary(analyzed_df)
        if not account_summary_df.empty:
            st.dataframe(account_summary_df, use_container_width=True, hide_index=True)

        theme_expo = build_theme_exposure(analyzed_df)

        st.markdown("### Theme exposure")
        if not theme_expo.empty:
            st.dataframe(theme_expo, use_container_width=True, hide_index=True)

        st.markdown("### Rebalance forslag")
        rebalance_df = build_rebalance_suggestions(analyzed_df, theme_expo)
        if not rebalance_df.empty:
            st.dataframe(rebalance_df, use_container_width=True, hide_index=True)