import pandas as pd
import streamlit as st

from src.data_sources import fetch_history, load_universe_csv
from src.discovery_engine import (
    build_discovery_table,
    discovery_deep_dive,
    top_discovery_candidates,
    weakening_themes,
)
from src.macro_engine import macro_snapshot
from src.portfolio_engine import (
    ACCOUNT_TYPES,
    analyze_portfolio_positions,
    build_account_summary,
    build_rebalance_suggestions,
    build_theme_exposure,
)
from src.screening_engine import run_screen_on_universe, summarize_screen, top_theme_hits
from src.strategy_engine import top_etfs, top_leaders, top_strategy_by_theme
from src.theme_definitions import THEMES
from src.theme_engine import build_theme_rankings, theme_deep_dive
from src.timing_engine import build_timing_snapshot

st.set_page_config(
    page_title="Stock Dashboard V3",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Stock Dashboard V3")

if "portfolio_positions" not in st.session_state:
    st.session_state["portfolio_positions"] = []

tab_analysis, tab_screening, tab_macro, tab_themes, tab_discovery, tab_strategy, tab_portfolio = st.tabs(
    ["Analyse", "Screening", "Macro", "Tema", "Discovery", "Strategy", "Portefølje"]
)

with tab_analysis:
    st.subheader("Analyse")

    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("Ticker", "AAPL")
    with c2:
        years = st.slider("Historik (år)", 1, 10, 5, 1)

    df = fetch_history(ticker, years=years)

    if df.empty:
        st.warning("Ingen data fundet for ticker.")
    else:
        timing = build_timing_snapshot(df)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Timing score", timing["timing_score"])
        m2.metric("Action", timing["action"])
        m3.metric("Trend", timing["trend"])
        m4.metric("RSI", timing["rsi"])
        m5.metric("1M momentum %", timing["momentum_1m"])
        m6.metric("ATR %", timing["atr_pct"])

        st.markdown("### Kurs")
        st.line_chart(df.set_index("Date")["Close"])

        with st.expander("Vis rå kursdata"):
            st.dataframe(df.tail(100), use_container_width=True, hide_index=True)

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
            max_tickers = st.number_input("Max tickers", min_value=10, max_value=5000, value=100, step=10)
        with c3:
            min_timing = st.slider("Min timing score", 0, 100, 40, 1)

        c4, c5 = st.columns(2)
        with c4:
            country_filter = st.multiselect("Land filter", available_countries, default=[])
        with c5:
            sector_filter = st.multiselect("Sektor filter", available_sectors, default=[])

        action_filter = st.multiselect("Action filter", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD"])

        if st.button("Kør screening", type="primary"):
            with st.spinner("Kører screening..."):
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

            if screen_df.empty:
                st.info("Ingen kandidater fundet.")
            else:
                summary = summarize_screen(screen_df)
                theme_hits_df = top_theme_hits(screen_df, top_n=10)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Kandidater", summary["count"])
                m2.metric("Gns screen score", summary["avg_screen_score"])
                m3.metric("BUY ratio %", summary["buy_ratio_pct"])
                m4.metric("Bullish ratio %", summary["bullish_ratio_pct"])

                st.markdown("### Resultater")
                st.dataframe(screen_df, use_container_width=True, hide_index=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### Top theme hits")
                    if theme_hits_df.empty:
                        st.info("Ingen theme hits endnu.")
                    else:
                        st.dataframe(theme_hits_df, use_container_width=True, hide_index=True)

                with c2:
                    st.markdown("### Universe preview")
                    st.dataframe(universe_preview_df.head(20), use_container_width=True, hide_index=True)

with tab_macro:
    st.subheader("Macro")

    macro = macro_snapshot()

    if not macro["fred_key_loaded"]:
        st.warning("FRED_API_KEY blev ikke fundet i secrets eller environment.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Regime", macro["regime"])
    m2.metric("Inflation YoY %", macro["inflation_yoy_pct"])
    m3.metric("Industrial Prod. YoY %", macro["industrial_production_yoy_pct"])

    m4, m5, m6 = st.columns(3)
    m4.metric("US 10Y", macro["us_10y"])
    m5.metric("US 2Y", macro["us_2y"])
    m6.metric("Arbejdsløshed", macro["unemployment"])

    m7, m8 = st.columns(2)
    m7.metric("10Y - 2Y", macro["rate_curve_10y_minus_2y"])
    m8.metric("Oil YoY %", macro["oil_yoy_pct"])

    with st.expander("Vis macro snapshot"):
        st.json(macro)

with tab_themes:
    st.subheader("Megatrends og temaer")

    ranking_df = build_theme_rankings()

    st.markdown("### Tema-ranking")
    if ranking_df.empty:
        st.warning("Kunne ikke beregne tema-ranking endnu.")
    else:
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    selected_theme = st.selectbox("Vælg tema", list(THEMES.keys()))
    deep = theme_deep_dive(selected_theme)

    if not deep:
        st.warning("Ingen data for valgt tema.")
    else:
        st.markdown(f"## {deep['theme']}")
        if deep.get("description"):
            st.write(deep["description"])
        if deep.get("summary_text"):
            st.info(deep["summary_text"])

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Proxy", deep["proxy"])
        p2.metric("RS 3M %", deep["rs_3m"])
        p3.metric("RS 6M %", deep["rs_6m"])
        p4.metric("Proxy Action", deep["proxy_timing"].get("action"))
        p5.metric("Proxy Timing", deep["proxy_timing"].get("timing_score"))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Drivere")
            for d in deep["drivers"]:
                st.write(f"- {d}")

            st.markdown("### ETF-kandidater")
            st.write(", ".join(deep["etfs"]) if deep["etfs"] else "Ingen ETF-kandidater endnu.")

        with c2:
            st.markdown("### Modvinde")
            for h in deep["headwinds"]:
                st.write(f"- {h}")

            st.markdown("### Ledende aktier")
            st.write(", ".join(deep["leaders"]) if deep["leaders"] else "Ingen leaders endnu.")

        st.markdown("### Medlemmer")
        members_df = deep["members_df"]
        if members_df is not None and not members_df.empty:
            st.dataframe(members_df, use_container_width=True, hide_index=True)
        else:
            st.info("Ingen medlemsdata endnu.")

with tab_discovery:
    st.subheader("Discovery engine")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("### Top discovery-kandidater")
        top_disc = top_discovery_candidates(10)
        if top_disc.empty:
            st.info("Ingen discovery-data endnu.")
        else:
            st.dataframe(top_disc, use_container_width=True, hide_index=True)

    with d2:
        st.markdown("### Svækkende temaer")
        weak_df = weakening_themes(10)
        if weak_df.empty:
            st.info("Ingen weakening-data endnu.")
        else:
            st.dataframe(weak_df, use_container_width=True, hide_index=True)

    st.markdown("### Komplet discovery-tabel")
    discovery_df = build_discovery_table()
    if discovery_df.empty:
        st.warning("Kunne ikke beregne discovery endnu.")
    else:
        st.dataframe(discovery_df, use_container_width=True, hide_index=True)

    selected_discovery_theme = st.selectbox(
        "Vælg tema til discovery deep dive",
        list(THEMES.keys()),
        key="discovery_theme_select",
    )

    disc = discovery_deep_dive(selected_discovery_theme)
    if disc:
        st.markdown(f"## {disc['theme']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Discovery Score", disc["discovery_score"])
        c2.metric("Stage", disc["stage"])
        c3.metric("Acceleration", disc["acceleration"])

        st.info(disc["why_now"])

        x1, x2 = st.columns(2)
        with x1:
            st.markdown("### Drivere")
            for d in disc["drivers"]:
                st.write(f"- {d}")

            st.markdown("### ETF'er")
            st.write(", ".join(disc["etfs"]) if disc["etfs"] else "Ingen ETF'er endnu.")

        with x2:
            st.markdown("### Modvinde")
            for h in disc["headwinds"]:
                st.write(f"- {h}")

            st.markdown("### Leaders")
            st.write(", ".join(disc["leaders"]) if disc["leaders"] else "Ingen leaders endnu.")

with tab_strategy:
    st.subheader("Strategy engine")

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("### Top ETF-kandidater")
        etf_df = top_etfs(15)
        if etf_df.empty:
            st.info("Ingen ETF-strategy data endnu.")
        else:
            st.dataframe(etf_df, use_container_width=True, hide_index=True)

    with s2:
        st.markdown("### Top ledende aktier")
        leader_df = top_leaders(20)
        if leader_df.empty:
            st.info("Ingen leader-strategy data endnu.")
        else:
            st.dataframe(leader_df, use_container_width=True, hide_index=True)

    st.markdown("### Strategy pr. tema")
    selected_strategy_theme = st.selectbox(
        "Vælg tema til strategy view",
        list(THEMES.keys()),
        key="strategy_theme_select",
    )

    theme_strategy_df = top_strategy_by_theme(selected_strategy_theme, top_n=12)
    if theme_strategy_df.empty:
        st.info("Ingen strategy-kandidater for valgt tema endnu.")
    else:
        st.dataframe(theme_strategy_df, use_container_width=True, hide_index=True)

with tab_portfolio:
    st.subheader("Portefølje")

    c1, c2, c3 = st.columns(3)
    with c1:
        pf_ticker = st.text_input("Ticker", "MSFT", key="pf_ticker")
    with c2:
        pf_amount = st.number_input("Antal", min_value=0.0, value=10.0, step=1.0, key="pf_amount")
    with c3:
        pf_account = st.selectbox("Konto", ACCOUNT_TYPES, index=0, key="pf_account")

    add_col, clear_col = st.columns(2)

    with add_col:
        if st.button("Tilføj position"):
            if pf_ticker.strip():
                st.session_state["portfolio_positions"].append(
                    {
                        "Ticker": pf_ticker.strip().upper(),
                        "Antal": float(pf_amount),
                        "Konto": pf_account,
                    }
                )

    with clear_col:
        if st.button("Ryd portefølje"):
            st.session_state["portfolio_positions"] = []

    positions_df = pd.DataFrame(st.session_state["portfolio_positions"])

    if positions_df.empty:
        st.info("Porteføljen er tom.")
    else:
        st.markdown("### Indtastede positioner")
        st.dataframe(positions_df, use_container_width=True, hide_index=True)

        analyzed_df = analyze_portfolio_positions(positions_df)
        account_summary_df = build_account_summary(analyzed_df)
        theme_exposure_df = build_theme_exposure(analyzed_df)
        rebalance_df = build_rebalance_suggestions(analyzed_df, theme_exposure_df)

        st.markdown("### Positionanalyse")
        st.dataframe(analyzed_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Kontosammendrag")
            if account_summary_df.empty:
                st.info("Ingen kontodata endnu.")
            else:
                st.dataframe(account_summary_df, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("### Temaeksponering")
            if theme_exposure_df.empty:
                st.info("Ingen temaeksponering endnu.")
            else:
                st.dataframe(theme_exposure_df, use_container_width=True, hide_index=True)

        st.markdown("### Rebalanceringsidéer")
        if rebalance_df.empty:
            st.info("Ingen forslag endnu.")
        else:
            st.dataframe(rebalance_df, use_container_width=True, hide_index=True)