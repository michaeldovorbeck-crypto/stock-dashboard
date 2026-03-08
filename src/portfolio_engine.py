# src/portfolio_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_sources import fetch_history
from src.theme_definitions import THEMES
from src.timing_engine import build_timing_snapshot


ACCOUNT_TYPES = ["Månedsopsparing", "Aldersopsparing", "Ratepension"]


def _theme_lookup_for_ticker(ticker: str) -> list[str]:
    ticker = (ticker or "").strip().upper()
    matches = []

    for theme_name, cfg in THEMES.items():
        members = [str(x).strip().upper() for x in cfg.get("members", [])]
        leaders = [str(x).strip().upper() for x in cfg.get("leaders", [])]
        etfs = [str(x).strip().upper() for x in cfg.get("etfs", [])]
        proxy = str(cfg.get("proxy", "")).strip().upper()

        all_refs = set(members + leaders + etfs + ([proxy] if proxy else []))
        if ticker in all_refs:
            matches.append(theme_name)

    return matches


def analyze_portfolio_positions(positions_df: pd.DataFrame) -> pd.DataFrame:
    if positions_df is None or positions_df.empty:
        return pd.DataFrame()

    work = positions_df.copy()

    required_cols = {"Ticker", "Antal", "Konto"}
    missing = required_cols - set(work.columns)
    if missing:
        raise ValueError(f"Mangler kolonner i portefølje: {sorted(missing)}")

    rows = []

    for _, row in work.iterrows():
        ticker = str(row["Ticker"]).strip()
        antal = float(row["Antal"]) if pd.notna(row["Antal"]) else 0.0
        konto = str(row["Konto"]).strip()

        df = fetch_history(ticker, years=5)

        if df.empty:
            rows.append(
                {
                    "Ticker": ticker,
                    "Konto": konto,
                    "Antal": antal,
                    "Seneste kurs": np.nan,
                    "Værdi": np.nan,
                    "Timing Score": np.nan,
                    "Action": "NO DATA",
                    "Trend": "No data",
                    "RSI": np.nan,
                    "1M Momentum %": np.nan,
                    "ATR %": np.nan,
                    "Temaer": "",
                    "Antal temaer": 0,
                }
            )
            continue

        timing = build_timing_snapshot(df)
        last_price = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
        value = antal * last_price

        themes = _theme_lookup_for_ticker(ticker)

        rows.append(
            {
                "Ticker": ticker,
                "Konto": konto,
                "Antal": antal,
                "Seneste kurs": round(last_price, 2),
                "Værdi": round(value, 2),
                "Timing Score": timing.get("timing_score", np.nan),
                "Action": timing.get("action", "NO DATA"),
                "Trend": timing.get("trend", "No data"),
                "RSI": timing.get("rsi", np.nan),
                "1M Momentum %": timing.get("momentum_1m", np.nan),
                "ATR %": timing.get("atr_pct", np.nan),
                "Temaer": ", ".join(themes),
                "Antal temaer": len(themes),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values("Værdi", ascending=False, na_position="last").reset_index(drop=True)

    return out


def build_account_summary(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    if analyzed_df is None or analyzed_df.empty:
        return pd.DataFrame()

    work = analyzed_df.copy()
    work["Værdi"] = pd.to_numeric(work["Værdi"], errors="coerce")
    work["Timing Score"] = pd.to_numeric(work["Timing Score"], errors="coerce")

    grouped = (
        work.groupby("Konto", dropna=False)
        .agg(
            Positioner=("Ticker", "count"),
            Samlet_værdi=("Værdi", "sum"),
            Gns_timing=("Timing Score", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "Samlet_værdi": "Samlet værdi",
            "Gns_timing": "Gns timing",
        }
    )

    grouped["Samlet værdi"] = grouped["Samlet værdi"].round(2)
    grouped["Gns timing"] = grouped["Gns timing"].round(2)

    return grouped.sort_values("Samlet værdi", ascending=False).reset_index(drop=True)


def build_theme_exposure(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    if analyzed_df is None or analyzed_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in analyzed_df.iterrows():
        value = pd.to_numeric(row.get("Værdi"), errors="coerce")
        theme_text = str(row.get("Temaer", "")).strip()

        if pd.isna(value) or value <= 0 or not theme_text:
            continue

        themes = [x.strip() for x in theme_text.split(",") if x.strip()]
        if not themes:
            continue

        split_value = float(value) / len(themes)

        for theme in themes:
            rows.append({"Theme": theme, "Value": split_value})

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).groupby("Theme", as_index=False)["Value"].sum()
    total = out["Value"].sum()

    if total > 0:
        out["Weight %"] = out["Value"] / total * 100.0
    else:
        out["Weight %"] = np.nan

    out["Value"] = out["Value"].round(2)
    out["Weight %"] = out["Weight %"].round(2)

    return out.sort_values("Weight %", ascending=False).reset_index(drop=True)


def build_rebalance_suggestions(analyzed_df: pd.DataFrame, theme_exposure_df: pd.DataFrame) -> pd.DataFrame:
    suggestions = []

    if analyzed_df is None or analyzed_df.empty:
        return pd.DataFrame()

    work = analyzed_df.copy()
    work["Værdi"] = pd.to_numeric(work["Værdi"], errors="coerce")
    work["Timing Score"] = pd.to_numeric(work["Timing Score"], errors="coerce")

    # Position-based suggestions
    for _, row in work.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        value = pd.to_numeric(row.get("Værdi"), errors="coerce")
        action = str(row.get("Action", "")).strip()
        timing = pd.to_numeric(row.get("Timing Score"), errors="coerce")

        if pd.notna(value) and value > 0:
            if action == "SELL":
                suggestions.append(
                    {
                        "Type": "Position",
                        "Target": ticker,
                        "Suggestion": "Review / trim",
                        "Reason": "Position har SELL-signal i timing engine.",
                    }
                )
            elif pd.notna(timing) and timing >= 70:
                suggestions.append(
                    {
                        "Type": "Position",
                        "Target": ticker,
                        "Suggestion": "Keep / add on weakness",
                        "Reason": "Position har høj timing score.",
                    }
                )

    # Theme-based suggestions
    if theme_exposure_df is not None and not theme_exposure_df.empty:
        for _, row in theme_exposure_df.iterrows():
            theme = str(row.get("Theme", "")).strip()
            weight = pd.to_numeric(row.get("Weight %"), errors="coerce")

            if pd.notna(weight) and weight > 35:
                suggestions.append(
                    {
                        "Type": "Theme",
                        "Target": theme,
                        "Suggestion": "Trim concentration",
                        "Reason": "Tema-vægt er over 35% og kan være for koncentreret.",
                    }
                )
            elif pd.notna(weight) and weight < 5:
                suggestions.append(
                    {
                        "Type": "Theme",
                        "Target": theme,
                        "Suggestion": "Low exposure",
                        "Reason": "Tema-vægt er lav og kan være underrepræsenteret.",
                    }
                )

    if not suggestions:
        return pd.DataFrame(
            columns=["Type", "Target", "Suggestion", "Reason"]
        )

    return pd.DataFrame(suggestions).drop_duplicates().reset_index(drop=True)