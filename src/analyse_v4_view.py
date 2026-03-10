from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis_engine import build_asset_analysis
from src.analysis_ui_components import (
    render_factor_view,
    render_hero_panel,
    render_market_now_cards,
    render_quick_stats_block,
    render_recommendation_panel,
    render_score_gauge_block,
    render_signal_summary_card,
    render_why_this_matters,
    safe_metric_value,
)
from src.chart_ui import render_candlestick_chart, render_volume_panel
from src.compare_ui import render_compare_block
from src.diagnostics_engine import (
    get_ticker_diagnostics,
    render_alternative_ticker_buttons,
    render_data_status_banner,
    render_diagnostics_tab,
)
from src.help_texts import HELP_TEXT
from src.help_ui import page_intro
from src.history_engine import recent_assets_df, register_recent_view
from src.news_bias_engine import build_news_bias_snapshot
from src.news_engine import build_asset_news_links
from src.overview_engine import build_market_overview, build_quick_picks
from src.peer_engine import build_peer_group
from src.portfolio_context_engine import build_portfolio_context
from src.portfolio_engine import ACCOUNT_TYPES
from src.search_engine import search_assets
from src.signal_duration_engine import build_signal_duration_snapshot
from src.signal_log_engine import append_signal_log
from src.storage_engine import (
    load_portfolio_positions,
    load_watchlist,
    save_portfolio_positions,
    save_watchlist,
)
from src.technical_view_engine import build_technical_view
from src.ui_style import apply_pro_style, render_badges, render_info_card
from src.unified_signal_engine import build_unified_signal_snapshot
from src.watchlist_engine import add_to_watchlist, remove_from_watchlist, watchlist_to_df


def _ensure_analysis_state() -> None:
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist()
    if "portfolio_positions" not in st.session_state:
        st.session_state["portfolio_positions"] = load_portfolio_positions()
    if "analysis_selected_ticker" not in st.session_state:
        st.session_state["analysis_selected_ticker"] = "AAPL"


def _get_watchlist_safe() -> list:
    return st.session_state.get("watchlist", [])


def _get_portfolio_positions_safe() -> list:
    return st.session_state.get("portfolio_positions", [])


def _render_market_header(market: dict) -> None:
    st.markdown("## Analyse 4.0 PRO")
    page_intro("analysis")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Makroregime", market.get("macro", {}).get("regime"), help=HELP_TEXT["macro_regime"])
    m2.metric("Top tema", market.get("top_theme") or "—")
    m3.metric("Top discovery", market.get("top_discovery") or "—", help=HELP_TEXT["discovery_score"])
    m4.metric("Top ETF", market.get("top_etf") or "—")
    m5.metric("Top leader", market.get("top_leader") or "—")
    m6.metric("BUY i snapshot", market.get("snapshot_buy_count", 0))


def _render_quick_picks() -> None:
    quick_picks = build_quick_picks()
    if not quick_picks:
        return

    st.markdown("### Quick picks")
    cols = st.columns(len(quick_picks))
    for i, ticker in enumerate(quick_picks):
        if cols[i].button(ticker, key=f"analysis4_qp_{i}_{ticker}"):
            st.session_state["analysis_selected_ticker"] = ticker
            st.rerun()


def _render_search_block() -> tuple[str, int, pd.DataFrame]:
    left, right = st.columns([3, 1])

    with left:
        search_query = st.text_input(
            "Global search (ticker, navn, ETF, tema, sektor)",
            value=st.session_state.get("analysis_selected_ticker", "AAPL"),
            key="analysis4_search_query",
        )

    with right:
        analysis_years = st.slider("Historik (år)", 1, 10, 5, 1, key="analysis4_years")

    search_df = search_assets(search_query, limit=20)

    st.markdown("### Søgeresultater")
    if search_df.empty:
        st.info("Ingen søgeresultater.")
        selected_ticker = search_query.strip().upper()
    else:
        options = []
        option_map = {}

        for _, row in search_df.iterrows():
            label = f"{row['ticker']} — {row.get('name', '')} | {row.get('type', '')}"
            if str(row.get("themes", "")).strip():
                label += f" | Themes: {row.get('themes', '')}"
            options.append(label)
            option_map[label] = row["ticker"]

        selected_label = st.selectbox("Vælg papir", options, index=0, key="analysis4_selected_label")
        selected_ticker = option_map[selected_label]
        st.session_state["analysis_selected_ticker"] = selected_ticker

        with st.expander("Vis søgeresultater"):
            st.dataframe(search_df, use_container_width=True, hide_index=True)

    return selected_ticker, analysis_years, search_df


def _render_right_sidebar() -> None:
    watchlist = _get_watchlist_safe()

    st.markdown("### Watchlist")
    st.caption(HELP_TEXT["watchlist"])
    st.dataframe(watchlist_to_df(watchlist), use_container_width=True, hide_index=True)

    st.markdown("### Senest sete")
    st.caption(HELP_TEXT["recent_views"])
    st.dataframe(recent_assets_df(), use_container_width=True, hide_index=True)


def _render_actions(analysis: dict) -> None:
    watchlist = _get_watchlist_safe()
    portfolio_positions = _get_portfolio_positions_safe()

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        if st.button("Tilføj til watchlist", key=f"analysis4_add_watch_{analysis['ticker']}"):
            st.session_state["watchlist"] = add_to_watchlist(watchlist, analysis["ticker"])
            save_watchlist(st.session_state["watchlist"])
            st.success(f"{analysis['ticker']} tilføjet til watchlist")

    with a2:
        if st.button("Fjern fra watchlist", key=f"analysis4_remove_watch_{analysis['ticker']}"):
            st.session_state["watchlist"] = remove_from_watchlist(watchlist, analysis["ticker"])
            save_watchlist(st.session_state["watchlist"])
            st.success(f"{analysis['ticker']} fjernet fra watchlist")

    with a3:
        if st.button("Tilføj til portefølje", key=f"analysis4_add_pf_{analysis['ticker']}"):
            portfolio_positions = list(portfolio_positions)
            portfolio_positions.append({"Ticker": analysis["ticker"], "Antal": 1.0, "Konto": ACCOUNT_TYPES[0]})
            st.session_state["portfolio_positions"] = portfolio_positions
            save_portfolio_positions(st.session_state["portfolio_positions"])
            st.success(f"{analysis['ticker']} tilføjet til porteføljen")

    with a4:
        if st.button("Log signal", key=f"analysis4_log_{analysis['ticker']}"):
            unified = analysis.get("unified_signal", {})
            append_signal_log(
                source="analysis_4_pro",
                ticker=analysis["ticker"],
                action=unified.get("overall_signal", ""),
                timing_score=unified.get("technical_score"),
                theme=analysis.get("record", {}).get("themes", ""),
                note="Unified signal log",
            )
            st.success("Signal logget")


def _render_signal_duration_block(analysis: dict) -> None:
    snapshot = build_signal_duration_snapshot(analysis["df"])

    st.markdown("### Teknisk signal over tid")
    st.caption(
        "Denne sektion viser hvor længe aktivet historisk har stået i KØB, HOLD eller SÆLG "
        "baseret på den samme fælles tekniske signalmotor."
    )

    left, right = st.columns([1, 1.4])

    with left:
        st.metric("Nuværende teknisk signal", snapshot.get("current_signal", "Ukendt"))
        st.metric("I nuværende signal", snapshot.get("streak_trading_days", 0))
        st.metric("Kalenderdage i streak", snapshot.get("streak_calendar_days", 0))

        last_switch = snapshot.get("last_switch_date")
        if last_switch is None:
            st.metric("Sidste skifte", "—")
        else:
            try:
                st.metric("Sidste skifte", pd.to_datetime(last_switch).date().isoformat())
            except Exception:
                st.metric("Sidste skifte", str(last_switch))

        st.metric("Forrige signal", snapshot.get("previous_signal", "Ukendt"))

    with right:
        dist_df = snapshot.get("distribution_df", pd.DataFrame())
        if dist_df is None or dist_df.empty:
            st.info("Ingen signalhistorik endnu.")
        else:
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

    recent_df = snapshot.get("recent_df", pd.DataFrame())
    if recent_df is not None and not recent_df.empty:
        with st.expander("Vis seneste signalhistorik"):
            show_cols = [c for c in ["Date", "Technical Signal", "Technical Score", "Trend State", "Close", "RSI14"] if c in recent_df.columns]
            st.dataframe(recent_df[show_cols], use_container_width=True, hide_index=True)


def _render_news_bias_block(analysis: dict) -> None:
    news_bias = analysis.get("news_bias_snapshot", {})
    score = news_bias.get("score", 0.0)
    bucket = news_bias.get("bucket", "Neutral")
    count = news_bias.get("headline_count", 0)

    st.markdown("### Nyhedsbias")
    st.caption("Nyhedsbias vurderer om de seneste overskrifter omkring aktivet er positive, neutrale eller negative.")

    n1, n2, n3 = st.columns(3)
    n1.metric("News bias", f"{score:+.1f}")
    n2.metric("Retning", bucket)
    n3.metric("Headlines", count)

    top_positive = news_bias.get("top_positive", [])
    top_negative = news_bias.get("top_negative", [])

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Mest positive headlines**")
        if top_positive:
            for x in top_positive:
                st.write(f"• {x}")
        else:
            st.info("Ingen tydeligt positive headlines.")

    with c2:
        st.markdown("**Mest negative headlines**")
        if top_negative:
            for x in top_negative:
                st.write(f"• {x}")
        else:
            st.info("Ingen tydeligt negative headlines.")

    headlines_df = news_bias.get("headlines_df", pd.DataFrame())
    if headlines_df is not None and not headlines_df.empty:
        with st.expander("Vis analyserede headlines"):
            cols = [c for c in ["title", "source", "pub_date", "headline_score"] if c in headlines_df.columns]
            st.dataframe(headlines_df[cols], use_container_width=True, hide_index=True)


def _render_overview_tab(analysis: dict, diag: dict, market: dict) -> None:
    tech_df = build_technical_view(analysis["df"])
    unified = analysis.get("unified_signal", {})

    st.info(HELP_TEXT["analysis_overview_box"])
    st.caption(HELP_TEXT["analysis_hero_box"])
    render_hero_panel(analysis, diag)

    technical_signal = str(unified.get("technical_signal", "HOLD"))
    overall_signal = str(unified.get("overall_signal", "HOLD"))

    badge_items = [
        (f"Teknisk: {technical_signal}", "buy" if technical_signal == "KØB" else "sell" if technical_signal == "SÆLG" else "hold"),
        (f"Samlet: {overall_signal}", "buy" if overall_signal == "KØB" else "sell" if overall_signal == "SÆLG" else "hold"),
        (f"Trend: {safe_metric_value(unified.get('trend_state'))}", "neutral"),
        (f"RSI: {safe_metric_value(analysis.get('timing', {}).get('rsi'))}", "neutral"),
    ]
    render_badges(badge_items)

    _render_actions(analysis)

    top_block_left, top_block_right = st.columns([1.5, 1])

    with top_block_left:
        render_recommendation_panel(unified)

    with top_block_right:
        st.markdown("### Niveau-overblik")
        st.caption("Her omsættes centrale målepunkter til lavt, mellem eller højt niveau.")
        render_score_gauge_block("Teknisk score", unified.get("technical_score"), HELP_TEXT["timing_score"])
        render_score_gauge_block("Samlet score", unified.get("overall_score"), HELP_TEXT["quant_score"])
        render_score_gauge_block("RSI", analysis.get("timing", {}).get("rsi"), HELP_TEXT["rsi"])
        data_score = 85 if unified.get("data_bucket") == "Høj" else 55 if unified.get("data_bucket") == "Mellem" else 20
        render_score_gauge_block("Datakvalitet", data_score, HELP_TEXT["data_source"])

    _render_news_bias_block(analysis)
    _render_signal_duration_block(analysis)

    top_left, top_right = st.columns([2, 1])

    with top_left:
        st.caption(HELP_TEXT["analysis_signal_summary_box"])
        render_signal_summary_card(analysis)

        st.caption(HELP_TEXT["analysis_why_this_matters_box"])
        render_why_this_matters(analysis)

    with top_right:
        render_factor_view(analysis)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("### Pris og glidende gennemsnit")
        st.caption("Her ser du prisudviklingen sammen med EMA20, EMA50 og EMA200 for at vurdere trendretning og styrke.")
        if not tech_df.empty:
            cols = [c for c in ["Close", "EMA20", "EMA50", "EMA200"] if c in tech_df.columns]
            if cols and "Date" in tech_df.columns:
                st.line_chart(tech_df.set_index("Date")[cols].dropna(how="all"))

    with c2:
        st.markdown("### Quick stats")
        st.caption("Quick stats opsummerer de vigtigste tekniske nøgletal i kompakt form.")
        render_quick_stats_block(analysis)
        render_info_card("Datakvalitet", str(len(diag.get("df", pd.DataFrame()))), "Antal datapunkter i analysen")

    st.caption(HELP_TEXT["analysis_market_now_box"])
    render_market_now_cards(market)


def _render_technicals_tab(analysis: dict) -> None:
    tech_df = build_technical_view(analysis["df"])

    st.subheader("Technicals")
    st.info(HELP_TEXT["analysis_technicals_box"])

    if tech_df.empty or "Date" not in tech_df.columns:
        st.info("Ingen technical data endnu.")
        return

    view_mode = st.radio(
        "Chart-visning",
        ["Candlestick + volume", "Candlestick", "Line chart"],
        horizontal=True,
        key="technical_chart_mode",
    )

    if view_mode == "Candlestick + volume":
        render_candlestick_chart(
            tech_df,
            title=f"{analysis['ticker']} — Candlestick + volume",
            show_volume=True,
            show_ema=True,
            height=700,
        )
    elif view_mode == "Candlestick":
        render_candlestick_chart(
            tech_df,
            title=f"{analysis['ticker']} — Candlestick",
            show_volume=False,
            show_ema=True,
            height=620,
        )
    else:
        cols = [c for c in ["Close", "EMA20", "EMA50", "EMA200"] if c in tech_df.columns]
        if cols:
            st.line_chart(tech_df.set_index("Date")[cols].dropna(how="all"))
        if "Volume" in tech_df.columns:
            st.markdown("### Volume")
            render_volume_panel(tech_df)

    c1, c2 = st.columns(2)

    with c1:
        if "Date" in tech_df.columns and "RSI14" in tech_df.columns:
            st.markdown("### RSI")
            st.caption("RSI bruges til at vurdere om aktivet er overkøbt, oversolgt eller neutralt.")
            st.line_chart(tech_df.set_index("Date")[["RSI14"]].dropna(how="all"))

    with c2:
        stats_rows = []
        latest = tech_df.iloc[-1] if not tech_df.empty else None
        if latest is not None:
            for col in ["Open", "High", "Low", "Close", "Volume", "EMA20", "EMA50", "EMA200", "RSI14"]:
                if col in tech_df.columns:
                    stats_rows.append({"Metric": col, "Latest": latest.get(col)})

        if stats_rows:
            st.markdown("### Seneste technicals")
            st.caption("Denne tabel viser de nyeste tekniske niveauer og indikatorer.")
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    raw_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "EMA20", "EMA50", "EMA200", "RSI14"] if c in tech_df.columns]
    if raw_cols:
        with st.expander("Vis technical table"):
            st.dataframe(tech_df[raw_cols].tail(100), use_container_width=True, hide_index=True)


def _render_context_tab(analysis: dict) -> None:
    record = analysis.get("record", {})
    macro = analysis.get("macro", {})
    portfolio_positions = _get_portfolio_positions_safe()

    st.subheader("Context")
    st.info(HELP_TEXT["analysis_context_box"])

    left, right = st.columns(2)

    with left:
        st.markdown("### Theme context")
        st.caption("Her ser du hvordan aktivet er knyttet til større investeringstemaer og strukturelle trends.")
        theme_context_df = analysis.get("theme_context_df", pd.DataFrame())
        if theme_context_df.empty:
            st.info("Ingen direkte theme context endnu.")
        else:
            st.dataframe(theme_context_df, use_container_width=True, hide_index=True)

        st.markdown("### Strategy context")
        st.caption("Her ser du hvordan aktivet relaterer sig til ETF-strategier, leaders eller bredere markedsstyrke.")
        strategy_context_df = analysis.get("strategy_context_df", pd.DataFrame())
        if strategy_context_df.empty:
            st.info("Ingen strategy context endnu.")
        else:
            st.dataframe(strategy_context_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Macro context")
        st.caption("Makrodata giver baggrunden for om miljøet understøtter risiko, vækst eller defensiv positionering.")
        st.metric("Regime", safe_metric_value(macro.get("regime")), help=HELP_TEXT["macro_regime"])
        st.metric("Inflation YoY %", safe_metric_value(macro.get("inflation_yoy_pct")))
        st.metric("Industrial YoY %", safe_metric_value(macro.get("industrial_production_yoy_pct")))
        st.metric("US 10Y", safe_metric_value(macro.get("us_10y")))
        st.metric("US 2Y", safe_metric_value(macro.get("us_2y")))
        st.metric("Arbejdsløshed", safe_metric_value(macro.get("unemployment")))

        st.markdown("### Asset identity")
        st.caption("Identity-sektionen viser hvad aktivet er, og hvilken rolle det kan have i en bredere analyse.")
        st.metric("Navn", safe_metric_value(record.get("name")))
        st.metric("Type", safe_metric_value(record.get("type")))
        st.metric("Land", safe_metric_value(record.get("country")))
        st.metric("Sektor", safe_metric_value(record.get("sector")))

        st.markdown("### Portfolio context")
        st.caption("Her ser du om aktivet allerede findes i porteføljen, og hvordan det er placeret på tværs af konti.")
        pf_ctx = build_portfolio_context(analysis["ticker"], portfolio_positions)
        if not pf_ctx["owned"]:
            st.info("Ikke i porteføljen endnu.")
        else:
            st.metric("Samlet antal", pf_ctx["total_shares"])
            st.metric("Konti", pf_ctx["accounts"])
            st.dataframe(pf_ctx["positions_df"], use_container_width=True, hide_index=True)


def _render_compare_tab(analysis: dict, years: int) -> None:
    record = analysis.get("record", {})

    st.subheader("Compare")
    st.info(HELP_TEXT["analysis_compare_box"])

    auto_peers = build_peer_group(record)

    if auto_peers:
        st.markdown("### Auto peers")
        st.caption("Auto peers er foreslåede sammenligningsaktiver baseret på tema, strategi eller beslægtet markedsrolle.")
        cols = st.columns(len(auto_peers))
        for i, peer in enumerate(auto_peers):
            if cols[i].button(peer, key=f"peer_{analysis['ticker']}_{i}_{peer}"):
                st.session_state["analysis_selected_ticker"] = peer
                st.rerun()

    compare_defaults = [analysis["ticker"]]
    if analysis["ticker"] != "SPY":
        compare_defaults.append("SPY")
    compare_defaults += auto_peers[:2]

    deduped_defaults = []
    seen = set()
    for item in compare_defaults:
        sym = str(item).strip().upper()
        if sym and sym not in seen:
            deduped_defaults.append(sym)
            seen.add(sym)

    compare_input = st.text_input(
        "Sammenlign tickers (komma-separeret)",
        value=",".join(deduped_defaults[:4]),
        key="analysis4_compare_input",
    )

    compare_tickers = [x.strip().upper() for x in compare_input.split(",") if x.strip()]
    render_compare_block(compare_tickers, years)


def _render_news_block(analysis: dict) -> None:
    record = analysis.get("record", {})
    news_links = build_asset_news_links(
        analysis["ticker"],
        name=record.get("name", ""),
        themes=record.get("themes", ""),
    )

    st.markdown("### News & catalysts")
    st.caption("Nyheder og catalysts hjælper med at forstå om der er aktuelle begivenheder, som kan forklare eller drive prisudviklingen.")
    n1, n2, n3 = st.columns(3)

    if news_links.get("ticker_news"):
        n1.markdown(f"[Ticker news]({news_links['ticker_news']})")
    if news_links.get("company_news"):
        n2.markdown(f"[Company news]({news_links['company_news']})")
    if news_links.get("theme_news"):
        n3.markdown(f"[Theme news]({news_links['theme_news']})")


def render_analysis_4() -> None:
    _ensure_analysis_state()
    apply_pro_style()
    market = build_market_overview()

    _render_market_header(market)
    _render_quick_picks()

    selected_ticker, analysis_years, search_df = _render_search_block()
    diag = get_ticker_diagnostics(selected_ticker, years=analysis_years)

    left, right = st.columns([3, 1])

    with right:
        _render_right_sidebar()

    with left:
        render_data_status_banner(diag)

        if not diag["ok"]:
            st.warning("Ingen data fundet for valgt ticker.")
            render_alternative_ticker_buttons(
                original_ticker=selected_ticker,
                alternatives=diag.get("alternatives", []),
                used_symbol=diag.get("used_symbol", ""),
                state_key="analysis_selected_ticker",
            )
            if not search_df.empty:
                st.markdown("### Alternative forslag fra søgning")
                cols_to_show = [c for c in ["ticker", "name", "type", "themes"] if c in search_df.columns]
                if cols_to_show:
                    st.dataframe(search_df[cols_to_show], use_container_width=True, hide_index=True)
            render_diagnostics_tab(diag)
            return

        analysis = build_asset_analysis(selected_ticker, years=analysis_years)

        if not analysis or not analysis.get("has_data"):
            st.warning(analysis.get("message", "Ingen data fundet."))
            render_alternative_ticker_buttons(
                original_ticker=selected_ticker,
                alternatives=diag.get("alternatives", []),
                used_symbol=diag.get("used_symbol", ""),
                state_key="analysis_selected_ticker",
            )
            render_diagnostics_tab(diag)
            return

        record = analysis.get("record", {})
        news_bias_snapshot = build_news_bias_snapshot(
            ticker=analysis.get("ticker", ""),
            company_name=record.get("name", ""),
            themes=record.get("themes", ""),
            limit=12,
        )
        analysis["news_bias_snapshot"] = news_bias_snapshot
        analysis["unified_signal"] = build_unified_signal_snapshot(
            analysis,
            diag,
            news_bias=news_bias_snapshot.get("score", 0.0),
        )

        register_recent_view(analysis["ticker"])

        tabs = st.tabs(["Overblik", "Technicals", "Context", "Compare", "Diagnostics"])

        with tabs[0]:
            _render_overview_tab(analysis, diag, market)
            _render_news_block(analysis)

        with tabs[1]:
            _render_technicals_tab(analysis)

        with tabs[2]:
            _render_context_tab(analysis)

        with tabs[3]:
            _render_compare_tab(analysis, analysis_years)

        with tabs[4]:
            st.info(HELP_TEXT["analysis_diagnostics_box"])
            render_diagnostics_tab(diag)