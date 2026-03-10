from __future__ import annotations

import pandas as pd

from src.analysis_engine import build_asset_analysis


ACCOUNT_TYPES = [
    "Månedsopsparing",
    "Aktiedepot",
    "Ratepension",
    "Aldersopsparing",
    "ASK",
    "Frie midler",
]


def _safe_float(value):
    try:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def analyze_portfolio_positions(positions_df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Konto",
                "Antal",
                "Seneste kurs",
                "Værdi",
                "Timing Score",
                "Action",
                "Trend",
                "RSI",
                "1M Momentum %",
                "ATR %",
                "Temaer",
                "Antal temaer",
            ]
        )

    work = positions_df.copy()

    if "Ticker" not in work.columns:
        work["Ticker"] = ""
    if "Antal" not in work.columns:
        work["Antal"] = 0.0
    if "Konto" not in work.columns:
        work["Konto"] = ""

    work["Ticker"] = work["Ticker"].astype(str).str.strip().str.upper()
    work["Antal"] = pd.to_numeric(work["Antal"], errors="coerce").fillna(0.0)
    work["Konto"] = work["Konto"].astype(str).fillna("")

    rows = []

    for _, row in work.iterrows():
        ticker = row["Ticker"]
        antal = float(row["Antal"])
        konto = row["Konto"]

        if not ticker or antal <= 0:
            continue

        analysis = build_asset_analysis(ticker, years=years)

        last_price = None
        value = None
        timing_score = None
        action = ""
        trend = ""
        rsi = None
        momentum_1m = None
        atr_pct = None
        themes = ""

        if analysis and analysis.get("has_data"):
            timing = analysis.get("timing", {})
            record = analysis.get("record", {})

            last_price = _safe_float(analysis.get("last"))
            value = None if last_price is None else round(last_price * antal, 2)
            timing_score = _safe_float(timing.get("timing_score"))
            action = str(timing.get("action", "") or "")
            trend = str(timing.get("trend", "") or "")
            rsi = _safe_float(timing.get("rsi"))
            momentum_1m = _safe_float(timing.get("momentum_1m"))
            atr_pct = _safe_float(timing.get("atr_pct"))
            themes = str(record.get("themes", "") or "")

        theme_count = len([x for x in str(themes).split(",") if str(x).strip()])

        rows.append(
            {
                "Ticker": ticker,
                "Konto": konto,
                "Antal": round(antal, 4),
                "Seneste kurs": None if last_price is None else round(last_price, 4),
                "Værdi": value,
                "Timing Score": None if timing_score is None else round(timing_score, 2),
                "Action": action,
                "Trend": trend,
                "RSI": None if rsi is None else round(rsi, 2),
                "1M Momentum %": None if momentum_1m is None else round(momentum_1m, 2),
                "ATR %": None if atr_pct is None else round(atr_pct, 2),
                "Temaer": themes,
                "Antal temaer": theme_count,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if "Værdi" in out.columns:
        out = out.sort_values("Værdi", ascending=False, na_position="last")

    return out.reset_index(drop=True)


def build_account_summary(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    if analyzed_df is None or analyzed_df.empty:
        return pd.DataFrame(columns=["Konto", "Positioner", "Samlet værdi", "Gns timing"])

    work = analyzed_df.copy()

    grouped = (
        work.groupby("Konto", dropna=False)
        .agg(
            Positioner=("Ticker", "count"),
            **{
                "Samlet værdi": ("Værdi", "sum"),
                "Gns timing": ("Timing Score", "mean"),
            },
        )
        .reset_index()
    )

    grouped["Samlet værdi"] = pd.to_numeric(grouped["Samlet værdi"], errors="coerce").round(2)
    grouped["Gns timing"] = pd.to_numeric(grouped["Gns timing"], errors="coerce").round(2)

    return grouped.sort_values("Samlet værdi", ascending=False, na_position="last").reset_index(drop=True)


def build_theme_exposure(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    if analyzed_df is None or analyzed_df.empty or "Temaer" not in analyzed_df.columns:
        return pd.DataFrame(columns=["Tema", "Eksponering", "Antal positioner"])

    rows = []

    for _, row in analyzed_df.iterrows():
        ticker = row.get("Ticker", "")
        value = _safe_float(row.get("Værdi")) or 0.0
        themes = [x.strip() for x in str(row.get("Temaer", "")).split(",") if x.strip()]

        for theme in themes:
            rows.append(
                {
                    "Tema": theme,
                    "Ticker": ticker,
                    "Eksponering": value,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Tema", "Eksponering", "Antal positioner"])

    work = pd.DataFrame(rows)

    grouped = (
        work.groupby("Tema", dropna=False)
        .agg(
            Eksponering=("Eksponering", "sum"),
            **{"Antal positioner": ("Ticker", "nunique")},
        )
        .reset_index()
    )

    grouped["Eksponering"] = pd.to_numeric(grouped["Eksponering"], errors="coerce").round(2)
    return grouped.sort_values("Eksponering", ascending=False, na_position="last").reset_index(drop=True)


def build_rebalance_suggestions(analyzed_df: pd.DataFrame, theme_expo_df: pd.DataFrame) -> pd.DataFrame:
    if analyzed_df is None or analyzed_df.empty:
        return pd.DataFrame(columns=["Ticker", "Konto", "Forslag", "Begrundelse"])

    rows = []

    for _, row in analyzed_df.iterrows():
        ticker = str(row.get("Ticker", "") or "")
        konto = str(row.get("Konto", "") or "")
        action = str(row.get("Action", "") or "").upper()
        timing_score = _safe_float(row.get("Timing Score"))
        value = _safe_float(row.get("Værdi")) or 0.0

        forslag = "Hold"
        begrundelse = "Ingen tydelig ændring nødvendig."

        if action == "SELL":
            forslag = "Reducer"
            begrundelse = "Aktuelt signal er SELL."
        elif action == "BUY" and timing_score is not None and timing_score > 65:
            forslag = "Overvej køb"
            begrundelse = "Stærkt BUY-signal og høj timing score."
        elif timing_score is not None and timing_score < 40:
            forslag = "Overvåg"
            begrundelse = "Lav timing score."

        if value == 0:
            forslag = "Ingen data"
            begrundelse = "Kunne ikke beregne positionens værdi."

        rows.append(
            {
                "Ticker": ticker,
                "Konto": konto,
                "Forslag": forslag,
                "Begrundelse": begrundelse,
            }
        )

    return pd.DataFrame(rows)