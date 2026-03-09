import pandas as pd
import streamlit as st

from src.analysis_engine import build_asset_analysis
from src.cache_engine import load_snapshot
from src.compare_engine import build_compare_chart_df, build_compare_table
from src.data_sources import load_universe_csv
from src.discovery_engine import (
    build_discovery_table,
    discovery_deep_dive,
    top_discovery_candidates,
    weakening_themes,
)
from src.history_engine import recent_assets_df, register_recent_view
from src.learning_engine import build_learning_summary
from src.macro_engine import macro_snapshot
from src.news_engine import build_asset_news_links
from src.portfolio_context_engine import build_portfolio_context
from src.portfolio_engine import (
    ACCOUNT_TYPES,
    analyze_portfolio_positions,
    build_account_summary,
    build_rebalance_suggestions,
    build_theme_exposure,
)
from src.precompute_engine import build_quant_snapshot_for_universe
from src.screening_engine import run_screen_on_universe, summarize_screen, top_theme_hits
from src.search_engine import search_assets
from src.signal_log_engine import append_signal_log, read_signal_log, signal_summary
from src.storage_engine import (
    load_portfolio_positions,
    load_watchlist,
    save_portfolio_positions,
    save_watchlist,
)
from src.strategy_engine import top_etfs, top_leaders, top_strategy_by_theme
from src.technical_view_engine import build_technical_view
from src.theme_definitions import THEMES
from src.theme_engine import build_theme_rankings, theme_deep_dive
from src.watchlist_engine import add_to_watchlist, remove_from_watchlist, watchlist_to_df

st.set_page_config(
    page_title="Stock Dashboard V3",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Stock Dashboard V3")

if "portfolio_positions" not in st.session_state:
    st.session_state["portfolio_positions"] = load_portfolio_positions()

if "analysis_selected_ticker" not in st.session_state:
    st.session_state["analysis_selected_ticker"] = "AAPL"

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = load_watchlist()

tab_analysis, tab_screening, tab_quant, tab_macro, tab_themes, tab_discovery, tab_strategy, tab_portfolio = st.tabs(
    ["Analyse", "Screening", "Quant", "Macro", "Tema", "Discovery", "Strategy", "Portefølje"]
)

with tab_analysis:
    st.subheader("Analyse 2.0")

    q1, q2 = st.columns([3, 1])
    with q1:
        search_query = st.text_input(
            "Global search (ticker, navn, ETF, tema, sektor)",
            value=st.session_state["analysis_selected_ticker"],
        )
    with q2:
        analysis_years = st.slider("Historik (år)", 1, 10, 5, 1, key="analysis_years")

    search_df = search_assets(search_query, limit=20)

    st.markdown("### Søgeresultater")
    if search_df.empty:
        st.info("Ingen søgeresultater.")
        selected_ticker = search_query.strip().upper()
    else:
        display_options = []
        option_map = {}
        for _, row in search_df.iterrows():
            label = f"{row['ticker']} — {row.get('name', '')} | {row.get('type', '')}"
            if str(row.get("themes", "")).strip():
                label += f" | Themes: {row.get('themes', '')}"
            option_map[label] = row["ticker"]
            display_options.append(label)

        selected_label = st.selectbox("Vælg papir", display_options, index=0)
        selected_ticker = option_map[selected_label]
        st.session_state["analysis_selected_ticker"] = selected_ticker

    analysis = build_asset_analysis(selected_ticker, years=analysis_years)

    if not analysis or not analysis.get("has_data"):
        st.warning(analysis.get("message", "Ingen data fundet for ticker."))
    else:
        register_recent_view(analysis["ticker"])
        record = analysis.get("record", {})
        timing = analysis.get("timing", {})
        macro = analysis.get("macro", {})
        returns = analysis.get("returns", {})
        tech_df = build_technical_view(analysis["df"])
        news_links = build_asset_news_links(
            analysis["ticker"],
            name=record.get("name", ""),
            themes=record.get("themes", ""),
        )

        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric("Ticker", analysis["ticker"])
        a2.metric("Seneste kurs", analysis["last"])
        a3.metric("Signal", timing.get("action"))
        a4.metric("Trend", timing.get("trend"))
        a5.metric("Timing score", timing.get("timing_score"))
        a6.metric("RSI", timing.get("rsi"))

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("1D %", returns.get("1D"))
        b2.metric("1M %", returns.get("1M"))
        b3.metric("3M %", returns.get("3M"))
        b4.metric("6M %", returns.get("6M"))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Navn", record.get("name", ""))
        c2.metric("Type", record.get("type", ""))
        c3.metric("Land", record.get("country", ""))
        c4.metric("Sektor", record.get("sector", ""))

        st.write(f"**Themes:** {record.get('themes', '') or 'Ingen tema-match endnu'}")

        add1, add2, add3, add4 = st.columns(4)
        with add1:
            if st.button("Tilføj til watchlist", key="add_watchlist_analysis"):
                st.session_state["watchlist"] = add_to_watchlist(st.session_state["watchlist"], analysis["ticker"])
                save_watchlist(st.session_state["watchlist"])
        with add2:
            if st.button("Fjern fra watchlist", key="remove_watchlist_analysis"):
                st.session_state["watchlist"] = remove_from_watchlist(st.session_state["watchlist"], analysis["ticker"])
                save_watchlist(st.session_state["watchlist"])
        with add3:
            if st.button("Tilføj til portefølje", key="add_portfolio_analysis"):
                st.session_state["portfolio_positions"].append(
                    {"Ticker": analysis["ticker"], "Antal": 1.0, "Konto": ACCOUNT_TYPES[0]}
                )
                save_portfolio_positions(st.session_state["portfolio_positions"])
        with add4:
            if st.button("Log signal", key="log_signal_analysis"):
                append_signal_log(
                    source="analysis",
                    ticker=analysis["ticker"],
                    action=timing.get("action", ""),
                    timing_score=timing.get("timing_score"),
                    theme=record.get("themes", ""),
                    note="Manual analysis log",
                )

        tc1, tc2 = st.columns(2)
        with tc1:
            if not tech_df.empty:
                st.line_chart(tech_df.set_index("Date")[["Close", "EMA20", "EMA50", "EMA200"]].dropna(how="all"))
        with tc2:
            if not tech_df.empty:
                st.line_chart(tech_df.set_index("Date")[["RSI14"]].dropna(how="all"))

        st.markdown("### Nyheder og catalysts")
        n1, n2, n3 = st.columns(3)
        n1.markdown(f"[Ticker news]({news_links['ticker_news']})")
        n2.markdown(f"[Company news]({news_links['company_news']})")
        if news_links["theme_news"]:
            n3.markdown(f"[Theme news]({news_links['theme_news']})")

        st.markdown("### Senest sete")
        st.dataframe(recent_assets_df(), use_container_width=True, hide_index=True)

        st.markdown("### Watchlist")
        st.dataframe(watchlist_to_df(st.session_state["watchlist"]), use_container_width=True, hide_index=True)

with tab_screening:
    st.subheader("Screening engine")

    universe_default = "global_all.csv"
    universe_file = st.text_input("Universe-fil", universe_default)

    universe_preview_df, universe_status = load_universe_csv(universe_file)

    if universe_preview_df.empty:
        st.warning(universe_status)
    else:
        st.caption(universe_status)

        available_countries = sorted([x for x in universe_preview_df["country"].dropna().astype(str).unique().tolist() if x])
        available_sectors = sorted([x for x in universe_preview_df["sector"].dropna().astype(str).unique().tolist() if x])

        c1, c2, c3 = st.columns(3)
        with c1:
            screen_years = st.slider("Screen historik (år)", 1, 10, 3, 1, key="screen_years")
        with c2:
            max_tickers = st.number_input("Max tickers", min_value=10, max_value=20000, value=500, step=100)
        with c3:
            min_timing = st.slider("Min timing score", 0, 100, 40, 1)

        c4, c5 = st.columns(2)
        with c4:
            country_filter = st.multiselect("Land filter", available_countries, default=[])
        with c5:
            sector_filter = st.multiselect("Sektor filter", available_sectors, default=[])

        action_filter = st.multiselect("Action filter", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD"])

        if st.button("Kør live screening", type="primary"):
            with st.spinner("Kører live screening..."):
                screen_df, screen_status = run_screen_on_universe(
                    filename=universe_file,
                    years=screen_years,
                    max_tickers=int(max_tickers),
                    min_timing_score=float(min_timing),
                    allowed_actions=action_filter,
                    country_filter=country_filter,
                    sector_filter=sector_filter,
                )

            st.write(screen_status)
            if not screen_df.empty:
                summary = summarize_screen(screen_df)
                theme_hits_df = top_theme_hits(screen_df, top_n=10)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kandidater", summary["count"])
                m2.metric("Gns screen score", summary["avg_screen_score"])
                m3.metric("BUY ratio %", summary["buy_ratio_pct"])
                m4.metric("Bullish ratio %", summary["bullish_ratio_pct"])

                st.dataframe(screen_df, use_container_width=True, hide_index=True)
                st.dataframe(theme_hits_df, use_container_width=True, hide_index=True)

with tab_quant:
    st.subheader("Quant upgrade")

    q1, q2, q3 = st.columns(3)
    with q1:
        snapshot_universe = st.text_input("Snapshot universe-fil", "global_all.csv", key="snapshot_universe")
    with q2:
        snapshot_years = st.slider("Snapshot historik (år)", 1, 10, 3, 1, key="snapshot_years")
    with q3:
        snapshot_max = st.number_input("Max tickers i snapshot", min_value=100, max_value=50000, value=5000, step=500)

    if st.button("Byg quant snapshot", type="primary"):
        with st.spinner("Bygger quant snapshot..."):
            snap_df, snap_status = build_quant_snapshot_for_universe(
                filename=snapshot_universe,
                years=int(snapshot_years),
                max_tickers=int(snapshot_max),
            )
        st.write(snap_status)
        if not snap_df.empty:
            st.dataframe(snap_df.head(100), use_container_width=True, hide_index=True)

    snapshot_df = load_snapshot()

    st.markdown("### Snapshot screener")
    if snapshot_df.empty:
        st.info("Intet snapshot endnu. Byg et snapshot først.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            min_quant = st.slider("Min Quant Score", 0, 100, 50, 1)
        with f2:
            min_timing = st.slider("Min Timing Score (snapshot)", 0, 100, 40, 1, key="min_timing_snapshot")
        with f3:
            only_actions = st.multiselect("Action filter (snapshot)", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD"], key="action_snapshot")

        work = snapshot_df.copy()
        work = work[pd.to_numeric(work["Quant Score"], errors="coerce") >= min_quant]
        work = work[pd.to_numeric(work["Timing Score"], errors="coerce") >= min_timing]
        work = work[work["Action"].astype(str).str.upper().isin([x.upper() for x in only_actions])]

        top_n = st.slider("Top N snapshot candidates", 10, 500, 50, 10)
        work = work.sort_values(["Quant Score", "Timing Score"], ascending=[False, False]).head(top_n)

        m1, m2, m3 = st.columns(3)
        m1.metric("Snapshot rows", len(snapshot_df))
        m2.metric("Filtered rows", len(work))
        m3.metric("Top Quant Score", None if work.empty else round(pd.to_numeric(work["Quant Score"], errors="coerce").max(), 2))

        st.dataframe(work, use_container_width=True, hide_index=True)

    st.markdown("### Learning overview")
    learn = build_learning_summary()
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Logs", learn["logs"])
    l2.metric("BUY logs", learn["buy_logs"])
    l3.metric("HOLD logs", learn["hold_logs"])
    l4.metric("SELL logs", learn["sell_logs"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top sources")
        st.dataframe(learn["top_sources"], use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Top tickers")
        st.dataframe(learn["top_tickers"], use_container_width=True, hide_index=True)

with tab_macro:
    macro = macro_snapshot()
    m1, m2, m3 = st.columns(3)
    m1.metric("Regime", macro["regime"])
    m2.metric("Inflation YoY %", macro["inflation_yoy_pct"])
    m3.metric("Industrial Prod. YoY %", macro["industrial_production_yoy_pct"])

with tab_themes:
    ranking_df = build_theme_rankings()
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    selected_theme = st.selectbox("Vælg tema", list(THEMES.keys()))
    deep = theme_deep_dive(selected_theme)
    if deep:
        st.write(deep.get("description", ""))
        st.dataframe(deep["members_df"], use_container_width=True, hide_index=True)

with tab_discovery:
    st.dataframe(top_discovery_candidates(10), use_container_width=True, hide_index=True)
    st.dataframe(weakening_themes(10), use_container_width=True, hide_index=True)
    st.dataframe(build_discovery_table(), use_container_width=True, hide_index=True)

with tab_strategy:
    st.dataframe(top_etfs(15), use_container_width=True, hide_index=True)
    st.dataframe(top_leaders(20), use_container_width=True, hide_index=True)

with tab_portfolio:
    c1, c2, c3 = st.columns(3)
    with c1:
        pf_ticker = st.text_input("Ticker", "MSFT", key="pf_ticker")
    with c2:
        pf_amount = st.number_input("Antal", min_value=0.0, value=10.0, step=1.0, key="pf_amount")
    with c3:
        pf_account = st.selectbox("Konto", ACCOUNT_TYPES, index=0, key="pf_account")

    if st.button("Tilføj position"):
        if pf_ticker.strip():
            st.session_state["portfolio_positions"].append(
                {"Ticker": pf_ticker.strip().upper(), "Antal": float(pf_amount), "Konto": pf_account}
            )
            save_portfolio_positions(st.session_state["portfolio_positions"])

    if st.button("Ryd portefølje"):
        st.session_state["portfolio_positions"] = []
        save_portfolio_positions(st.session_state["portfolio_positions"])

    positions_df = pd.DataFrame(st.session_state["portfolio_positions"])
    if positions_df.empty:
        st.info("Porteføljen er tom.")
    else:
        analyzed_df = analyze_portfolio_positions(positions_df)
        st.dataframe(analyzed_df, use_container_width=True, hide_index=True)
        st.dataframe(build_account_summary(analyzed_df), use_container_width=True, hide_index=True)
        st.dataframe(build_theme_exposure(analyzed_df), use_container_width=True, hide_index=True)
        st.dataframe(build_rebalance_suggestions(analyzed_df, build_theme_exposure(analyzed_df)), use_container_width=True, hide_index=True)