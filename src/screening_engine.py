# src/screening_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_sources import fetch_history, load_universe_csv
from src.theme_definitions import THEMES
from src.timing_engine import build_timing_snapshot


def _theme_lookup_for_ticker(ticker: str) -> list[str]:
    ticker = (ticker or "").strip().upper()
    matches = []

    for theme_name, cfg in THEMES.items():
        members = [str(x).strip().upper() for x in cfg.get("members", [])]
        leaders = [str(x).strip().upper() for x in cfg.get("leaders", [])]
        etfs = [str(x).strip().upper() for x in cfg.get("etfs", [])]
        proxy = str(cfg.get("proxy", "")).strip().upper()

        refs = set(members + leaders + etfs + ([proxy] if proxy else []))
        if ticker in refs:
            matches.append(theme_name)

    return matches


def _build_screen_row(row: pd.Series, years: int = 3) -> dict | None:
    ticker = str(row.get("ticker", "")).strip()
    name = str(row.get("name", "")).strip()
    sector = str(row.get("sector", "")).strip()
    country = str(row.get("country", "")).strip()

    if not ticker:
        return None

    df = fetch_history(ticker, years=years)
    if df.empty:
        return None

    timing = build_timing_snapshot(df)

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return None

    last = float(close.iloc[-1])

    ret_1m = np.nan
    ret_3m = np.nan
    ret_6m = np.nan

    if len(close) >= 21:
        ret_1m = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1.0) * 100.0
    if len(close) >= 63:
        ret_3m = (float(close.iloc[-1]) / float(close.iloc[-63]) - 1.0) * 100.0
    if len(close) >= 126:
        ret_6m = (float(close.iloc[-1]) / float(close.iloc[-126]) - 1.0) * 100.0

    themes = _theme_lookup_for_ticker(ticker)

    screen_score = 0.0

    timing_score = pd.to_numeric(timing.get("timing_score"), errors="coerce")
    if pd.notna(timing_score):
        screen_score += float(timing_score) * 0.60

    mom_1m = pd.to_numeric(timing.get("momentum_1m"), errors="coerce")
    mom_3m = pd.to_numeric(timing.get("momentum_3m"), errors="coerce")
    rsi = pd.to_numeric(timing.get("rsi"), errors="coerce")
    atr_pct = pd.to_numeric(timing.get("atr_pct"), errors="coerce")

    if pd.notna(mom_1m):
        screen_score += max(-10.0, min(10.0, float(mom_1m) / 2.0))
    if pd.notna(mom_3m):
        screen_score += max(-10.0, min(10.0, float(mom_3m) / 4.0))

    if pd.notna(rsi) and 50 <= rsi <= 70:
        screen_score += 6.0

    if pd.notna(atr_pct):
        if atr_pct > 6:
            screen_score -= 8.0
        elif atr_pct > 4:
            screen_score -= 4.0

    if themes:
        screen_score += min(8.0, len(themes) * 2.0)

    screen_score = max(0.0, min(100.0, screen_score))

    return {
        "Ticker": ticker,
        "Name": name,
        "Sector": sector,
        "Country": country,
        "Last": round(last, 2),
        "Timing Score": round(float(timing_score), 2) if pd.notna(timing_score) else np.nan,
        "Action": timing.get("action", "NO DATA"),
        "Trend": timing.get("trend", "No data"),
        "RSI": round(float(rsi), 2) if pd.notna(rsi) else np.nan,
        "ATR %": round(float(atr_pct), 2) if pd.notna(atr_pct) else np.nan,
        "1M Momentum %": round(float(mom_1m), 2) if pd.notna(mom_1m) else np.nan,
        "3M Momentum %": round(float(mom_3m), 2) if pd.notna(mom_3m) else np.nan,
        "1M Return %": round(ret_1m, 2) if pd.notna(ret_1m) else np.nan,
        "3M Return %": round(ret_3m, 2) if pd.notna(ret_3m) else np.nan,
        "6M Return %": round(ret_6m, 2) if pd.notna(ret_6m) else np.nan,
        "Themes": ", ".join(themes),
        "Theme Count": len(themes),
        "Screen Score": round(screen_score, 2),
    }


def run_screen_on_universe(
    filename: str,
    years: int = 3,
    max_tickers: int = 100,
    min_timing_score: float = 0.0,
    allowed_actions: list[str] | None = None,
    country_filter: list[str] | None = None,
    sector_filter: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    uni_df, status = load_universe_csv(filename)
    if uni_df.empty:
        return pd.DataFrame(), status

    work = uni_df.copy()

    if country_filter:
        cf = {str(x).strip() for x in country_filter if str(x).strip()}
        if cf:
            work = work[work["country"].astype(str).isin(cf)]

    if sector_filter:
        sf = {str(x).strip() for x in sector_filter if str(x).strip()}
        if sf:
            work = work[work["sector"].astype(str).isin(sf)]

    work = work.head(max_tickers).reset_index(drop=True)

    rows = []
    for _, row in work.iterrows():
        out = _build_screen_row(row, years=years)
        if out is not None:
            rows.append(out)

    if not rows:
        return pd.DataFrame(), "Ingen brugbare resultater i screeningen."

    df = pd.DataFrame(rows)

    if min_timing_score > 0:
        df = df[pd.to_numeric(df["Timing Score"], errors="coerce") >= float(min_timing_score)]

    if allowed_actions:
        aa = {str(x).strip().upper() for x in allowed_actions}
        df = df[df["Action"].astype(str).str.upper().isin(aa)]

    if df.empty:
        return pd.DataFrame(), "Ingen kandidater matchede filtrene."

    df = df.sort_values(
        ["Screen Score", "Timing Score", "3M Momentum %"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return df, f"Screening færdig: {len(df)} kandidater fra {filename}"


def summarize_screen(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "count": 0,
            "avg_screen_score": None,
            "avg_timing_score": None,
            "buy_ratio_pct": None,
            "bullish_ratio_pct": None,
        }

    work = df.copy()

    screen = pd.to_numeric(work["Screen Score"], errors="coerce")
    timing = pd.to_numeric(work["Timing Score"], errors="coerce")

    buy_ratio = (work["Action"].astype(str).str.upper() == "BUY").mean() * 100.0
    bullish_ratio = work["Trend"].astype(str).isin(["Bullish", "Positive"]).mean() * 100.0

    return {
        "count": int(len(work)),
        "avg_screen_score": round(float(screen.mean()), 2) if not screen.dropna().empty else None,
        "avg_timing_score": round(float(timing.mean()), 2) if not timing.dropna().empty else None,
        "buy_ratio_pct": round(float(buy_ratio), 2),
        "bullish_ratio_pct": round(float(bullish_ratio), 2),
    }


def top_theme_hits(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df is None or df.empty or "Themes" not in df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        score = pd.to_numeric(row.get("Screen Score"), errors="coerce")
        themes = str(row.get("Themes", "")).strip()
        if not themes:
            continue

        for theme in [x.strip() for x in themes.split(",") if x.strip()]:
            rows.append({"Theme": theme, "Screen Score": score})

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).groupby("Theme", as_index=False).agg(
        Hits=("Theme", "count"),
        Avg_Screen_Score=("Screen Score", "mean"),
    )

    out["Avg_Screen_Score"] = out["Avg_Screen_Score"].round(2)

    return out.sort_values(
        ["Hits", "Avg_Screen_Score"],
        ascending=[False, False]
    ).head(top_n).reset_index(drop=True)