# src/strategy_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_sources import fetch_history
from src.theme_engine import build_theme_rankings
from src.theme_definitions import THEMES
from src.timing_engine import build_timing_snapshot


def _safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _instrument_snapshot(ticker: str, years: int = 5) -> dict:
    df = fetch_history(ticker, years=years)

    if df.empty:
        return {
            "Ticker": ticker,
            "Timing Score": np.nan,
            "Action": "NO DATA",
            "Trend": "No data",
            "1M Momentum %": np.nan,
            "3M Momentum %": np.nan,
            "RSI": np.nan,
            "ATR %": np.nan,
        }

    timing = build_timing_snapshot(df)

    return {
        "Ticker": ticker,
        "Timing Score": timing.get("timing_score", np.nan),
        "Action": timing.get("action", "NO DATA"),
        "Trend": timing.get("trend", "No data"),
        "1M Momentum %": timing.get("momentum_1m", np.nan),
        "3M Momentum %": timing.get("momentum_3m", np.nan),
        "RSI": timing.get("rsi", np.nan),
        "ATR %": timing.get("atr_pct", np.nan),
    }


def build_strategy_candidates() -> pd.DataFrame:
    theme_rankings = build_theme_rankings()
    if theme_rankings.empty:
        return pd.DataFrame()

    theme_score_map = {
        row["Theme"]: row["Theme Score"]
        for _, row in theme_rankings.iterrows()
        if "Theme" in row and "Theme Score" in row
    }

    rows = []

    for theme_name, cfg in THEMES.items():
        theme_score = _safe_float(theme_score_map.get(theme_name, np.nan))

        etfs = cfg.get("etfs", [])
        leaders = cfg.get("leaders", [])

        for ticker in etfs:
            snap = _instrument_snapshot(ticker, years=5)
            timing_score = _safe_float(snap["Timing Score"])

            strategy_score = 0.0
            if pd.notna(theme_score):
                strategy_score += theme_score * 0.55
            if pd.notna(timing_score):
                strategy_score += timing_score * 0.45

            rows.append(
                {
                    "Theme": theme_name,
                    "Type": "ETF",
                    "Ticker": ticker,
                    "Theme Score": round(theme_score, 2) if pd.notna(theme_score) else np.nan,
                    "Timing Score": round(timing_score, 2) if pd.notna(timing_score) else np.nan,
                    "Strategy Score": round(strategy_score, 2) if pd.notna(strategy_score) else np.nan,
                    "Action": snap["Action"],
                    "Trend": snap["Trend"],
                    "1M Momentum %": snap["1M Momentum %"],
                    "3M Momentum %": snap["3M Momentum %"],
                    "RSI": snap["RSI"],
                    "ATR %": snap["ATR %"],
                }
            )

        for ticker in leaders:
            snap = _instrument_snapshot(ticker, years=5)
            timing_score = _safe_float(snap["Timing Score"])

            strategy_score = 0.0
            if pd.notna(theme_score):
                strategy_score += theme_score * 0.45
            if pd.notna(timing_score):
                strategy_score += timing_score * 0.55

            rows.append(
                {
                    "Theme": theme_name,
                    "Type": "Leader",
                    "Ticker": ticker,
                    "Theme Score": round(theme_score, 2) if pd.notna(theme_score) else np.nan,
                    "Timing Score": round(timing_score, 2) if pd.notna(timing_score) else np.nan,
                    "Strategy Score": round(strategy_score, 2) if pd.notna(strategy_score) else np.nan,
                    "Action": snap["Action"],
                    "Trend": snap["Trend"],
                    "1M Momentum %": snap["1M Momentum %"],
                    "3M Momentum %": snap["3M Momentum %"],
                    "RSI": snap["RSI"],
                    "ATR %": snap["ATR %"],
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["Strategy Score", "Theme Score", "Timing Score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return out


def top_strategy_by_theme(theme_name: str, top_n: int = 10) -> pd.DataFrame:
    df = build_strategy_candidates()
    if df.empty:
        return pd.DataFrame()

    df = df[df["Theme"] == theme_name].copy()
    if df.empty:
        return pd.DataFrame()

    return df.head(top_n).reset_index(drop=True)


def top_etfs(top_n: int = 15) -> pd.DataFrame:
    df = build_strategy_candidates()
    if df.empty:
        return pd.DataFrame()

    df = df[df["Type"] == "ETF"].copy()
    if df.empty:
        return pd.DataFrame()

    return df.head(top_n).reset_index(drop=True)


def top_leaders(top_n: int = 20) -> pd.DataFrame:
    df = build_strategy_candidates()
    if df.empty:
        return pd.DataFrame()

    df = df[df["Type"] == "Leader"].copy()
    if df.empty:
        return pd.DataFrame()

    return df.head(top_n).reset_index(drop=True)