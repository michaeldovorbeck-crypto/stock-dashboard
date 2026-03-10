from __future__ import annotations

import pandas as pd
import streamlit as st


def build_discovery_df_from_themes(
    positions_signal_df: pd.DataFrame,
    themes_dict: dict,
) -> pd.DataFrame:
    if positions_signal_df.empty or "Ticker" not in positions_signal_df.columns:
        return pd.DataFrame(columns=["ticker", "themes", "discovery_score"])

    rows = []

    unique_tickers = positions_signal_df["Ticker"].astype(str).str.upper().dropna().unique()

    for ticker in unique_tickers:
        matched_themes = []

        for theme_name, theme_data in themes_dict.items():
            members = theme_data.get("members", [])
            members_upper = [str(member).upper() for member in members]
            if ticker in members_upper:
                matched_themes.append(theme_name)

        if not matched_themes:
            matched_themes = ["Unclassified"]

        discovery_score = 70.0 if matched_themes != ["Unclassified"] else 50.0

        rows.append(
            {
                "ticker": ticker,
                "themes": "|".join(matched_themes),
                "discovery_score": discovery_score,
            }
        )

    return pd.DataFrame(rows)


def build_macro_df_from_session() -> pd.DataFrame:
    regime = str(st.session_state.get("macro_regime", "NEUTRAL")).upper()
    modifier = float(st.session_state.get("macro_risk_modifier", 1.0))

    return pd.DataFrame(
        [
            {
                "macro_regime": regime,
                "macro_risk_modifier": modifier,
            }
        ]
    )