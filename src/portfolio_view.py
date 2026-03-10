# portfolio_view.py

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.portfolio_intelligence_engine import build_portfolio_intelligence
from src.portfolio_risk_engine import build_portfolio_risk


# -----------------------------------------------------------------------------
# CACHED BUILDERS
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=900)
def cached_build_portfolio_intelligence(
    portfolio_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    signal_history_df: pd.DataFrame,
    news_df: pd.DataFrame,
    discovery_df: pd.DataFrame,
    macro_df: pd.DataFrame,
):
    return build_portfolio_intelligence(
        portfolio_df=portfolio_df,
        analysis_df=analysis_df,
        signal_df=signal_df,
        signal_history_df=signal_history_df,
        news_df=news_df,
        discovery_df=discovery_df,
        macro_df=macro_df,
    )


@st.cache_data(show_spinner=False, ttl=900)
def cached_build_portfolio_risk(
    portfolio_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    discovery_df: pd.DataFrame,
    macro_df: pd.DataFrame,
):
    return build_portfolio_risk(
        portfolio_df=portfolio_df,
        analysis_df=analysis_df,
        discovery_df=discovery_df,
        macro_df=macro_df,
    )


# -----------------------------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------------------------

def _format_pct(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "-"


def _severity_icon(severity: str) -> str:
    sev = str(severity).upper()
    if sev == "HIGH":
        return "🔴"
    if sev == "MEDIUM":
        return "🟠"
    if sev == "LOW":
        return "🟡"
    return "🔵"


def _render_alerts(alerts_df: pd.DataFrame, title: str):
    st.subheader(title)

    if alerts_df is None or alerts_df.empty:
        st.info("Ingen alerts.")
        return

    for _, row in alerts_df.iterrows():
        icon = _severity_icon(row.get("severity", "INFO"))
        message = row.get("message", "")
        severity = row.get("severity", "INFO")
        st.markdown(f"**{icon} {severity}** — {message}")


def _render_metric_row(
    health_summary_df: pd.DataFrame,
    risk_summary_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    volatility_df: pd.DataFrame,
):
    c1, c2, c3, c4 = st.columns(4)

    if not health_summary_df.empty:
        row = health_summary_df.iloc[0]
        c1.metric(
            "Portfolio Health",
            f"{row.get('portfolio_health_score', 0):.1f}/100",
            row.get("health_status", "Neutral"),
        )

    if not risk_summary_df.empty:
        row = risk_summary_df.iloc[0]
        delta = f"{row.get('risk_label', 'Moderate')} | {row.get('macro_regime', 'NEUTRAL')}"
        c2.metric(
            "Risk Score",
            f"{row.get('risk_score', 0):.1f}/100",
            delta,
        )

    if not concentration_df.empty:
        row = concentration_df.iloc[0]
        c3.metric(
            "Top Position",
            f"{row.get('top_position_ticker', '-')}",
            f"{row.get('top_position_weight_pct', 0):.2f}%",
        )
        c4.metric(
            "Top 3 Weight",
            f"{row.get('top_3_weight_pct', 0):.2f}%",
            volatility_df.iloc[0].get("volatility_bucket", "Unknown") if not volatility_df.empty else "Unknown",
        )


def _render_signal_distribution(signal_dist_df: pd.DataFrame):
    st.subheader("Signal Distribution")

    if signal_dist_df.empty:
        st.info("Ingen signaldata.")
        return

    chart_df = signal_dist_df.set_index("signal")[["count"]]
    st.bar_chart(chart_df)
    st.dataframe(signal_dist_df, use_container_width=True)


def _render_sector_exposure(sector_exposure_df: pd.DataFrame):
    st.subheader("Sector Exposure")

    if sector_exposure_df.empty:
        st.info("Ingen sektordata.")
        return

    chart_df = sector_exposure_df.set_index("sector")[["weight_pct"]]
    st.bar_chart(chart_df)
    st.dataframe(sector_exposure_df, use_container_width=True)


def _render_theme_exposure(theme_exposure_df: pd.DataFrame):
    st.subheader("Theme Exposure")

    if theme_exposure_df.empty:
        st.info("Ingen temadata.")
        return

    top_df = theme_exposure_df.head(12).copy()
    chart_df = top_df.set_index("theme")[["weight_pct"]]
    st.bar_chart(chart_df)
    st.dataframe(top_df, use_container_width=True)


def _render_signal_drift(signal_drift_df: pd.DataFrame):
    st.subheader("Signal Drift")

    if signal_drift_df.empty:
        st.info("Ingen signalhistorik tilgængelig.")
        return

    drift_only = signal_drift_df[signal_drift_df["drift_flag"] == True].copy()
    if drift_only.empty:
        st.success("Ingen nye signalændringer registreret.")
        st.dataframe(signal_drift_df, use_container_width=True)
    else:
        st.dataframe(drift_only, use_container_width=True)


def _render_rebalance(rebalance_df: pd.DataFrame):
    st.subheader("Smart Rebalance Suggestions")

    if rebalance_df.empty:
        st.info("Ingen rebalance-forslag.")
        return

    st.dataframe(rebalance_df, use_container_width=True)


def _render_concentration(concentration_df: pd.DataFrame, diversification_df: pd.DataFrame):
    st.subheader("Risk Concentration")

    c1, c2, c3, c4 = st.columns(4)

    if not concentration_df.empty:
        row = concentration_df.iloc[0]
        c1.metric("Top Position", row.get("top_position_ticker", "-"))
        c2.metric("Top Position Weight", f"{row.get('top_position_weight_pct', 0):.2f}%")
        c3.metric("Top 3 Weight", f"{row.get('top_3_weight_pct', 0):.2f}%")

    if not diversification_df.empty:
        row = diversification_df.iloc[0]
        c4.metric("Effective N", f"{row.get('effective_n_positions', 0):.2f}")

    if not diversification_df.empty:
        st.dataframe(diversification_df, use_container_width=True)


# -----------------------------------------------------------------------------
# MAIN VIEW
# -----------------------------------------------------------------------------

def render_portfolio_view(
    portfolio_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None = None,
    signal_df: pd.DataFrame | None = None,
    signal_history_df: pd.DataFrame | None = None,
    news_df: pd.DataFrame | None = None,
    discovery_df: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None,
):
    st.title("Portfolio Intelligence & Risk")

    portfolio_df = portfolio_df if isinstance(portfolio_df, pd.DataFrame) else pd.DataFrame()
    analysis_df = analysis_df if isinstance(analysis_df, pd.DataFrame) else pd.DataFrame()
    signal_df = signal_df if isinstance(signal_df, pd.DataFrame) else pd.DataFrame()
    signal_history_df = signal_history_df if isinstance(signal_history_df, pd.DataFrame) else pd.DataFrame()
    news_df = news_df if isinstance(news_df, pd.DataFrame) else pd.DataFrame()
    discovery_df = discovery_df if isinstance(discovery_df, pd.DataFrame) else pd.DataFrame()
    macro_df = macro_df if isinstance(macro_df, pd.DataFrame) else pd.DataFrame()

    if portfolio_df.empty:
        st.warning("Porteføljen er tom. Tilføj eller upload positioner først.")
        return

    intel = cached_build_portfolio_intelligence(
        portfolio_df=portfolio_df,
        analysis_df=analysis_df,
        signal_df=signal_df,
        signal_history_df=signal_history_df,
        news_df=news_df,
        discovery_df=discovery_df,
        macro_df=macro_df,
    )

    risk = cached_build_portfolio_risk(
        portfolio_df=portfolio_df,
        analysis_df=analysis_df,
        discovery_df=discovery_df,
        macro_df=macro_df,
    )

    health_summary_df = intel["health_summary"]
    signal_distribution_df = intel["signal_distribution"]
    signal_drift_df = intel["signal_drift"]
    alerts_df = intel["alerts"]
    rebalance_df = intel["rebalance"]

    concentration_df = risk["concentration"]
    sector_exposure_df = risk["sector_exposure"]
    theme_exposure_df = risk["theme_exposure"]
    volatility_df = risk["volatility"]
    diversification_df = risk["diversification"]
    risk_summary_df = risk["risk_summary"]
    risk_alerts_df = risk["risk_alerts"]

    _render_metric_row(
        health_summary_df=health_summary_df,
        risk_summary_df=risk_summary_df,
        concentration_df=concentration_df,
        volatility_df=volatility_df,
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Signals & Rebalance",
        "Risk",
        "Raw Data",
    ])

    with tab1:
        left, right = st.columns(2)

        with left:
            _render_signal_distribution(signal_distribution_df)
            _render_alerts(alerts_df, "Portfolio Alerts")

        with right:
            _render_alerts(risk_alerts_df, "Risk Alerts")
            _render_concentration(concentration_df, diversification_df)

    with tab2:
        left, right = st.columns(2)

        with left:
            _render_signal_drift(signal_drift_df)

        with right:
            _render_rebalance(rebalance_df)

    with tab3:
        left, right = st.columns(2)

        with left:
            _render_sector_exposure(sector_exposure_df)

        with right:
            _render_theme_exposure(theme_exposure_df)

        st.subheader("Volatility")
        st.dataframe(volatility_df, use_container_width=True)

        st.subheader("Risk Summary")
        st.dataframe(risk_summary_df, use_container_width=True)

    with tab4:
        st.subheader("Portfolio Snapshot")
        st.dataframe(intel["snapshot"], use_container_width=True)

        st.subheader("Risk Snapshot")
        st.dataframe(risk["snapshot"], use_container_width=True)