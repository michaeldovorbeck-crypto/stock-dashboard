from __future__ import annotations

import pandas as pd

from src.analysis_engine import build_asset_analysis


def build_portfolio_signals(positions_df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """
    Bygger BUY/HOLD/SELL signaler for hele porteføljen
    """

    if positions_df is None or positions_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in positions_df.iterrows():

        ticker = str(row.get("Ticker", "")).upper()
        shares = row.get("Antal", 0)
        account = row.get("Konto", "")

        if not ticker:
            continue

        analysis = build_asset_analysis(ticker, years=years)

        if not analysis or not analysis.get("has_data"):
            continue

        timing = analysis.get("timing", {})

        rows.append(
            {
                "Ticker": ticker,
                "Konto": account,
                "Antal": shares,
                "Signal": timing.get("action"),
                "Timing Score": timing.get("timing_score"),
                "Trend": timing.get("trend"),
                "RSI": timing.get("rsi"),
                "1M Momentum": timing.get("momentum_1m"),
                "3M Momentum": timing.get("momentum_3m"),
                "ATR %": timing.get("atr_pct"),
            }
        )

    return pd.DataFrame(rows)


def build_portfolio_signal_summary(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Opsummerer hvor mange BUY/HOLD/SELL der er i porteføljen
    """

    if signal_df is None or signal_df.empty:
        return pd.DataFrame()

    summary = (
        signal_df.groupby("Signal")
        .size()
        .reset_index(name="Antal")
        .sort_values("Antal", ascending=False)
    )

    return summary


def build_signal_alerts(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finder kritiske ændringer i porteføljen
    """

    if signal_df is None or signal_df.empty:
        return pd.DataFrame()

    alerts = []

    for _, row in signal_df.iterrows():

        ticker = row["Ticker"]
        signal = row["Signal"]
        score = row["Timing Score"]

        if signal == "SELL":
            alerts.append(
                {
                    "Ticker": ticker,
                    "Alert": "Overvej at reducere position",
                    "Score": score,
                }
            )

        elif signal == "BUY" and score and score > 70:
            alerts.append(
                {
                    "Ticker": ticker,
                    "Alert": "Stærkt købssignal",
                    "Score": score,
                }
            )

    return pd.DataFrame(alerts)

def enrich_positions_with_signals(positions_df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """
    Tilføjer BUY/HOLD/SELL signaler direkte til portefølje positioner
    """

    if positions_df is None or positions_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in positions_df.iterrows():

        ticker = str(row.get("Ticker", "")).upper()
        account = row.get("Konto", "")
        shares = row.get("Antal", 0)

        if not ticker:
            continue

        analysis = build_asset_analysis(ticker, years=years)

        if not analysis or not analysis.get("has_data"):
            rows.append(row.to_dict())
            continue

        timing = analysis.get("timing", {})

        new_row = row.to_dict()

        new_row["Signal"] = timing.get("action")
        new_row["Timing Score"] = timing.get("timing_score")
        new_row["Trend"] = timing.get("trend")
        new_row["RSI"] = timing.get("rsi")
        new_row["1M Momentum"] = timing.get("momentum_1m")
        new_row["3M Momentum"] = timing.get("momentum_3m")

        rows.append(new_row)

    return pd.DataFrame(rows)