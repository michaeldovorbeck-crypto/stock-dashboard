# src/compare_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_sources import fetch_history
from src.timing_engine import build_timing_snapshot


def _safe_return(close: pd.Series, lookback: int) -> float:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < lookback:
        return np.nan
    base = float(c.iloc[-lookback])
    last = float(c.iloc[-1])
    if base == 0:
        return np.nan
    return (last / base - 1.0) * 100.0


def build_compare_table(tickers: list[str], years: int = 5) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        t = str(ticker).strip().upper()
        if not t:
            continue

        df = fetch_history(t, years=years)
        if df.empty:
            rows.append(
                {
                    "Ticker": t,
                    "Last": np.nan,
                    "Timing Score": np.nan,
                    "Action": "NO DATA",
                    "Trend": "No data",
                    "RSI": np.nan,
                    "ATR %": np.nan,
                    "1M %": np.nan,
                    "3M %": np.nan,
                    "6M %": np.nan,
                }
            )
            continue

        timing = build_timing_snapshot(df)
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        last = float(close.iloc[-1]) if not close.empty else np.nan

        rows.append(
            {
                "Ticker": t,
                "Last": round(last, 2) if pd.notna(last) else np.nan,
                "Timing Score": timing.get("timing_score", np.nan),
                "Action": timing.get("action", "NO DATA"),
                "Trend": timing.get("trend", "No data"),
                "RSI": timing.get("rsi", np.nan),
                "ATR %": timing.get("atr_pct", np.nan),
                "1M %": round(_safe_return(close, 21), 2) if pd.notna(_safe_return(close, 21)) else np.nan,
                "3M %": round(_safe_return(close, 63), 2) if pd.notna(_safe_return(close, 63)) else np.nan,
                "6M %": round(_safe_return(close, 126), 2) if pd.notna(_safe_return(close, 126)) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Timing Score", ascending=False, na_position="last").reset_index(drop=True)


def build_compare_chart_df(tickers: list[str], years: int = 3) -> pd.DataFrame:
    merged = None

    for ticker in tickers:
        t = str(ticker).strip().upper()
        if not t:
            continue

        df = fetch_history(t, years=years)
        if df.empty:
            continue

        work = df[["Date", "Close"]].copy()
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work["Close"] = pd.to_numeric(work["Close"], errors="coerce")
        work = work.dropna().sort_values("Date").reset_index(drop=True)

        if work.empty:
            continue

        base = float(work["Close"].iloc[0])
        if base == 0:
            continue

        work[t] = work["Close"] / base * 100.0
        work = work[["Date", t]]

        if merged is None:
            merged = work
        else:
            merged = merged.merge(work, on="Date", how="outer")

    if merged is None:
        return pd.DataFrame()

    return merged.sort_values("Date").reset_index(drop=True)