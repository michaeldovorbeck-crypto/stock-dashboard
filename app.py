from __future__ import annotations

import hmac

import pandas as pd
import streamlit as st

from src.analyse_v4_view import render_analysis_4
from src.discovery_v4_view import render_discovery_4
from src.macro_v4_view import render_macro_4
from src.strategy_v4_view import render_strategy_4
from src.portfolio_view import render_portfolio_view

from src.cache_engine import load_snapshot
from src.data_sources import load_universe_csv
from src.learning_engine import build_learning_summary
from src.precompute_engine import build_quant_snapshot_for_universe
from src.screening_engine import run_screen_on_universe, summarize_screen, top_theme_hits
from src.storage_engine import (
    load_portfolio_positions,
    load_portfolio_transactions,
    load_watchlist,
    save_portfolio_positions,
    save_portfolio_transactions,
)

from src.portfolio_engine import (
    ACCOUNT_TYPES,
    analyze_portfolio_positions,
    build_account_summary,
    build_rebalance_suggestions,
    build_theme_exposure,
)

from src.portfolio_transactions_engine import (
    add_transaction,
    build_positions_from_transactions,
    normalize_transactions_df,
    remove_transaction_by_index,
    transaction_display_df,
)

from src.portfolio_upload_engine import (
    load_transactions_from_upload,
    merge_transactions,
)

from src.portfolio_signal_engine import (
    enrich_positions_with_signals,
    build_portfolio_signal_summary,
)

from src.depot_positions_engine import (
    load_positions_from_depot_uploads,
    normalize_uploaded_positions_df,
    portfolio_positions_display_df,
)

from src.theme_engine import build_theme_rankings, theme_deep_dive
from src.theme_definitions import THEMES

from src.help_ui import global_help_expander, page_intro, render_dashboard_guide_sidebar
from src.help_texts import HELP_TEXT
from src.onboarding_ui import render_onboarding_guide
from src.portfolio_app_helpers import (
    build_discovery_df_from_themes,
    build_macro_df_from_session,
)


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Stock Dashboard V4 PRO",
    page_icon="📊",
    layout="wide",
)


# -----------------------------------------------------------------------------
# SIMPLE LOGIN
# -----------------------------------------------------------------------------

APP_PASSWORD = "Mosevej3"


def check_password(password: str) -> bool:
    return hmac.compare_digest(str(password), APP_PASSWORD)


def require_login() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return

    st.title("🔐 Stock Dashboard V4 PRO")
    st.write("Privat adgang kræver login.")

    with st.form("login_form", clear_on_submit=False):
        password = st.text_input("Adgangskode", type="password")
        submitted = st.form_submit_button("Log ind", type="primary")

    if submitted:
        if check_password(password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Forkert adgangskode")

    st.stop()


def render_logout_button() -> None:
    if st.button("Log ud", key="logout_button"):
        st.session_state["authenticated"] = False
        st.rerun()


require_login()


# -----------------------------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------------------------

if "portfolio_positions" not in st.session_state:
    st.session_state["portfolio_positions"] = load_portfolio_positions()

if "portfolio_transactions" not in st.session_state:
    st.session_state["portfolio_transactions"] = load_portfolio_transactions()

if "portfolio_uploaded_positions" not in st.session_state:
    st.session_state["portfolio_uploaded_positions"] = []

if "analysis_selected_ticker" not in st.session_state:
    st.session_state["analysis_selected_ticker"] = "AAPL"

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = load_watchlist()

if "macro_regime" not in st.session_state:
    st.session_state["macro_regime"] = "NEUTRAL"

if "macro_risk_modifier" not in st.session_state:
    st.session_state["macro_risk_modifier"] = 1.0


# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------

st.title("📊 Stock Dashboard V4 PRO")
global_help_expander()
render_onboarding_guide()

with st.sidebar:
    render_dashboard_guide_sidebar()
    st.markdown("---")
    render_logout_button()


# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------

with tab_analysis:
    render_analysis_4()


# -----------------------------------------------------------------------------
# SCREENING
# -----------------------------------------------------------------------------

with tab_screening:
    st.subheader("Screening engine")
    page_intro("screening")

    universe_file = st.text_input(
        "Universe-fil",
        "global_all.csv",
        help=HELP_TEXT.get("universe", ""),
        key="screen_universe_file",
    )

    universe_df, universe_status = load_universe_csv(universe_file)

    if universe_df.empty:
        st.warning(universe_status)
    else:
        st.caption(universe_status)

        c1, c2, c3 = st.columns(3)

        with c1:
            screen_years = st.slider("Historik (år)", 1, 10, 3, key="screen_years")

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


# -----------------------------------------------------------------------------
# QUANT
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# MACRO
# -----------------------------------------------------------------------------

with tab_macro:
    render_macro_4()


# -----------------------------------------------------------------------------
# THEMES
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# DISCOVERY
# -----------------------------------------------------------------------------

with tab_discovery:
    render_discovery_4()


# -----------------------------------------------------------------------------
# STRATEGY
# -----------------------------------------------------------------------------

with tab_strategy:
    render_strategy_4()


# -----------------------------------------------------------------------------
# PORTFOLIO
# -----------------------------------------------------------------------------

with tab_portfolio:
    st.subheader("Portefølje")
    page_intro("portfolio")

    st.info(
        "Portefølje 3.1 – Du kan nu uploade både handelslog og depotoversigter. "
        "Depotoversigter kobles automatisk til dine 3 depoter: "
        "AKT 11005683, RPP 11005956 og AOP 47994264."
    )

    tx_df = normalize_transactions_df(pd.DataFrame(st.session_state["portfolio_transactions"]))
    uploaded_positions_df = normalize_uploaded_positions_df(
        pd.DataFrame(st.session_state["portfolio_uploaded_positions"])
    )

    # -------------------------------------------------------------------------
    # DEPOTOVERSIGT UPLOAD
    # -------------------------------------------------------------------------

    st.markdown("### Upload depotoversigter (aktuelle beholdninger)")
    depot_files = st.file_uploader(
        "Upload en eller flere depotoversigter fra dine 3 depoter",
        type=["csv"],
        accept_multiple_files=True,
        key="portfolio_depot_upload",
    )

    if depot_files:
        imported_positions_df, depot_msg = load_positions_from_depot_uploads(depot_files)
        st.write(depot_msg)

        if not imported_positions_df.empty:
            st.markdown("#### Preview af depotbeholdninger")
            st.caption(
                "Ticker kan redigeres før import. Konto kobles automatisk ud fra kontonummer i filnavnet."
            )

            editable_cols = [
                c for c in [
                    "Account Code",
                    "Account",
                    "Ticker",
                    "Asset Name",
                    "Currency",
                    "Net Shares",
                    "Avg Cost",
                    "Last Price",
                    "Market Value DKK",
                    "Return %",
                    "Return DKK",
                ]
                if c in imported_positions_df.columns
            ]

            edited_positions_df = st.data_editor(
                imported_positions_df[editable_cols],
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="portfolio_depot_editor",
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Account Code": st.column_config.TextColumn("Depot"),
                    "Account": st.column_config.TextColumn("Konto"),
                    "Asset Name": st.column_config.TextColumn("Navn"),
                },
            )

            merge_keys = ["Account", "Asset Name", "Currency", "Net Shares"]
            editable_merge_cols = [c for c in merge_keys + ["Ticker"] if c in edited_positions_df.columns]

            edited_merge = edited_positions_df[editable_merge_cols].copy()

            imported_positions_df = imported_positions_df.drop(columns=["Ticker"], errors="ignore").merge(
                edited_merge,
                on=[c for c in merge_keys if c in imported_positions_df.columns and c in edited_merge.columns],
                how="left",
            )

            d1, d2 = st.columns(2)

            with d1:
                if st.button("Indlæs depotoversigter som aktuelle beholdninger", key="portfolio_import_depots"):
                    clean_positions = normalize_uploaded_positions_df(imported_positions_df)
                    st.session_state["portfolio_uploaded_positions"] = clean_positions.to_dict(orient="records")
                    st.success(f"{len(clean_positions)} aktuelle positioner indlæst fra depotoversigter.")

            with d2:
                if st.button("Ryd uploaded depotbeholdninger", key="portfolio_clear_uploaded_positions"):
                    st.session_state["portfolio_uploaded_positions"] = []
                    st.warning("Uploaded depotbeholdninger er ryddet.")

    # -------------------------------------------------------------------------
    # HANDELSLOG UPLOAD
    # -------------------------------------------------------------------------

    st.markdown("### Upload handler fra CSV")
    uploaded_file = st.file_uploader(
        "Upload CSV med handler",
        type=["csv"],
        key="portfolio_csv_upload",
    )

    if uploaded_file is not None:
        imported_df, import_msg = load_transactions_from_upload(uploaded_file)
        st.write(import_msg)

        if not imported_df.empty:
            st.markdown("#### Import preview")
            st.dataframe(transaction_display_df(imported_df), use_container_width=True, hide_index=True)

            u1, u2 = st.columns(2)

            with u1:
                if st.button("Tilføj import til eksisterende handelslog", key="portfolio_merge_upload"):
                    merged = merge_transactions(tx_df, imported_df)
                    st.session_state["portfolio_transactions"] = merged.to_dict(orient="records")
                    save_portfolio_transactions(st.session_state["portfolio_transactions"])
                    st.success(f"Import tilføjet. Handelslog har nu {len(merged)} handler.")

            with u2:
                if st.button("Erstat eksisterende handelslog med import", key="portfolio_replace_upload"):
                    clean_import = normalize_transactions_df(imported_df)
                    st.session_state["portfolio_transactions"] = clean_import.to_dict(orient="records")
                    save_portfolio_transactions(st.session_state["portfolio_transactions"])
                    st.success(f"Handelslog erstattet med {len(clean_import)} importerede handler.")

    # -------------------------------------------------------------------------
    # MANUEL HANDEL
    # -------------------------------------------------------------------------

    st.markdown("### Tilføj handel manuelt")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        tx_date = st.date_input("Dato", key="pf_tx_date")
    with c2:
        tx_ticker = st.text_input("Ticker", "MSFT", key="pf_tx_ticker")
    with c3:
        tx_account = st.selectbox("Konto", ACCOUNT_TYPES, key="pf_tx_account")
    with c4:
        tx_side = st.selectbox("Side", ["BUY", "SELL"], key="pf_tx_side")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        tx_shares = st.number_input("Antal", min_value=0.0, value=10.0, step=1.0, key="pf_tx_shares")
    with c6:
        tx_price = st.number_input("Kurs", min_value=0.0, value=100.0, step=0.01, key="pf_tx_price")
    with c7:
        tx_fee = st.number_input("Gebyr", min_value=0.0, value=0.0, step=0.01, key="pf_tx_fee")
    with c8:
        tx_note = st.text_input("Note", "", key="pf_tx_note")

    a1, a2 = st.columns(2)
    with a1:
        if st.button("Tilføj handel", key="pf_add_transaction"):
            new_df = add_transaction(
                tx_df,
                date_value=tx_date,
                ticker=tx_ticker,
                account=tx_account,
                side=tx_side,
                shares=float(tx_shares),
                price=float(tx_price),
                fee=float(tx_fee),
                note=tx_note,
            )
            st.session_state["portfolio_transactions"] = new_df.to_dict(orient="records")
            save_portfolio_transactions(st.session_state["portfolio_transactions"])
            st.success(f"Handel tilføjet: {tx_side} {tx_ticker.upper()}")

    with a2:
        if st.button("Ryd hele handelslog", key="pf_clear_transactions"):
            st.session_state["portfolio_transactions"] = []
            save_portfolio_transactions([])
            st.warning("Hele handelsloggen er ryddet")

    tx_df = normalize_transactions_df(pd.DataFrame(st.session_state["portfolio_transactions"]))
    uploaded_positions_df = normalize_uploaded_positions_df(
        pd.DataFrame(st.session_state["portfolio_uploaded_positions"])
    )

    # -------------------------------------------------------------------------
    # AKTIVE POSITIONER
    # -------------------------------------------------------------------------

    st.markdown("### Åbne positioner")

    tx_positions_df = build_positions_from_transactions(tx_df)

    if not uploaded_positions_df.empty:
        positions_df = uploaded_positions_df.copy()
        st.success("Aktuelle beholdninger kommer fra uploaded depotoversigter.")
        st.dataframe(
            portfolio_positions_display_df(positions_df),
            use_container_width=True,
            hide_index=True,
        )
    else:
        positions_df = tx_positions_df.copy()
        if positions_df.empty:
            st.info("Ingen åbne positioner endnu.")
        else:
            st.caption("Aktuelle beholdninger beregnes fra handelsloggen.")

    positions_signal_df = pd.DataFrame()
    signal_history_df = pd.DataFrame()
    analysis_df = pd.DataFrame()
    signal_df = pd.DataFrame()
    news_df = pd.DataFrame()
    discovery_df = pd.DataFrame()
    macro_df = pd.DataFrame()

    if not positions_df.empty:
        positions_for_signals = positions_df.copy()
        if "Ticker" in positions_for_signals.columns:
            positions_for_signals = positions_for_signals[
                positions_for_signals["Ticker"].fillna("").astype(str).str.strip() != ""
            ].copy()

        if positions_for_signals.empty:
            st.warning("Ingen positioner har ticker endnu. Udfyld ticker i depot-upload preview for at få signaler.")
        else:
            with st.spinner("Beriger positioner med signaler..."):
                positions_signal_df = enrich_positions_with_signals(positions_for_signals)

            summary = build_portfolio_signal_summary(positions_signal_df)

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Positioner", summary.get("positions", 0))
            s2.metric("KØB", summary.get("buy_count", 0))
            s3.metric("HOLD", summary.get("hold_count", 0))
            s4.metric("SÆLG", summary.get("sell_count", 0))
            s5.metric("Gns samlet score", summary.get("avg_overall_score"))

            st.dataframe(positions_signal_df, use_container_width=True, hide_index=True)

            st.markdown("### Handlingsliste")
            action_cols = [
                c for c in [
                    "Ticker",
                    "Account",
                    "Net Shares",
                    "Avg Cost",
                    "Last Price",
                    "Unrealized P/L %",
                    "Technical Signal",
                    "Overall Signal",
                    "Overall Score",
                    "News Bias",
                    "Signal Days",
                    "Action Flag",
                ] if c in positions_signal_df.columns
            ]
            if action_cols:
                st.dataframe(
                    positions_signal_df[action_cols].copy(),
                    use_container_width=True,
                    hide_index=True,
                )

            analysis_df = positions_signal_df.copy().rename(
                columns={
                    "Ticker": "ticker",
                    "Last Price": "current_price",
                    "Technical Signal": "trend",
                    "Overall Signal": "signal",
                    "Overall Score": "timing_score",
                    "News Bias": "news_sentiment",
                }
            )

            signal_df = positions_signal_df.copy().rename(
                columns={
                    "Ticker": "ticker",
                    "Overall Signal": "signal",
                    "Overall Score": "signal_score",
                    "Signal Days": "signal_streak",
                }
            )
            keep_cols = [c for c in ["ticker", "signal", "signal_score", "signal_streak"] if c in signal_df.columns]
            signal_df = signal_df[keep_cols] if keep_cols else pd.DataFrame()

            signal_history_df = positions_signal_df.copy().rename(
                columns={
                    "Ticker": "ticker",
                    "Overall Signal": "signal",
                }
            )
            if "ticker" in signal_history_df.columns and "signal" in signal_history_df.columns:
                signal_history_df = signal_history_df[["ticker", "signal"]].copy()
                signal_history_df["date"] = pd.Timestamp.today().normalize()

            if "News Bias" in positions_signal_df.columns:
                news_df = positions_signal_df[["Ticker", "News Bias"]].copy().rename(
                    columns={
                        "Ticker": "ticker",
                        "News Bias": "news_sentiment",
                    }
                )

            discovery_df = build_discovery_df_from_themes(positions_signal_df, THEMES)
            macro_df = build_macro_df_from_session()

            st.markdown("---")
            render_portfolio_view(
                portfolio_df=positions_for_signals,
                analysis_df=analysis_df,
                signal_df=signal_df,
                signal_history_df=signal_history_df,
                news_df=news_df,
                discovery_df=discovery_df,
                macro_df=macro_df,
            )

    st.markdown("### Handelslog")
    if tx_df.empty:
        st.info("Ingen handler registreret endnu.")
    else:
        tx_display = transaction_display_df(tx_df).reset_index(drop=True)
        tx_display_show = tx_display.copy()
        tx_display_show.insert(0, "Idx", tx_display_show.index)

        st.dataframe(tx_display_show, use_container_width=True, hide_index=True)

        remove_idx = st.number_input(
            "Slet handel med indeks (se kolonnen Idx ovenfor)",
            min_value=0,
            max_value=max(0, len(tx_display_show) - 1),
            value=0,
            step=1,
            key="pf_remove_tx_idx",
        )

        if st.button("Slet valgt handel", key="pf_remove_transaction"):
            updated = remove_transaction_by_index(tx_df, int(remove_idx))
            st.session_state["portfolio_transactions"] = updated.to_dict(orient="records")
            save_portfolio_transactions(st.session_state["portfolio_transactions"])
            st.success("Valgt handel er slettet")

    # -------------------------------------------------------------------------
    # LEGACY SECTION
    # -------------------------------------------------------------------------

    st.markdown("### Legacy porteføljevisning")
    st.caption("Denne sektion bevares midlertidigt for bagudkompatibilitet med den gamle positionsmodel.")

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
            "Legacy konto",
            ACCOUNT_TYPES,
            key="portfolio_account",
        )

    if st.button("Tilføj legacy position", key="portfolio_add_button"):
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

    legacy_positions_df = pd.DataFrame(st.session_state["portfolio_positions"])

    if legacy_positions_df.empty:
        st.info("Legacy porteføljen er tom.")
    else:
        analyzed_df = analyze_portfolio_positions(legacy_positions_df)

        st.dataframe(analyzed_df, use_container_width=True, hide_index=True)

        st.markdown("### Legacy konto-overblik")
        account_summary_df = build_account_summary(analyzed_df)
        if not account_summary_df.empty:
            st.dataframe(account_summary_df, use_container_width=True, hide_index=True)

        theme_expo = build_theme_exposure(analyzed_df)

        st.markdown("### Legacy theme exposure")
        if not theme_expo.empty:
            st.dataframe(theme_expo, use_container_width=True, hide_index=True)

        st.markdown("### Legacy rebalance forslag")
        rebalance_df = build_rebalance_suggestions(analyzed_df, theme_expo)
        if not rebalance_df.empty:
            st.dataframe(rebalance_df, use_container_width=True, hide_index=True)