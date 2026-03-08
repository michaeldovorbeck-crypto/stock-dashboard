# src/theme_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_sources import fetch_history
from src.theme_definitions import THEMES
from src.timing_engine import build_timing_snapshot


def _pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a / b - 1.0) * 100.0


def _relative_strength(proxy_df: pd.DataFrame, bench_df: pd.DataFrame, lookback_days: int) -> float:
    if proxy_df.empty or bench_df.empty:
        return np.nan

    p = proxy_df[["Date", "Close"]].dropna().copy()
    b = bench_df[["Date", "Close"]].dropna().copy()

    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce")

    p["Close"] = pd.to_numeric(p["Close"], errors="coerce")
    b["Close"] = pd.to_numeric(b["Close"], errors="coerce")

    p = p.dropna().sort_values("Date").reset_index(drop=True)
    b = b.dropna().sort_values("Date").reset_index(drop=True)

    if len(p) < lookback_days or len(b) < lookback_days:
        return np.nan

    p_last = float(p["Close"].iloc[-1])
    p_prev = float(p["Close"].iloc[-lookback_days])
    b_last = float(b["Close"].iloc[-1])
    b_prev = float(b["Close"].iloc[-lookback_days])

    p_ret = _pct_change(p_last, p_prev)
    b_ret = _pct_change(b_last, b_prev)

    if pd.isna(p_ret) or pd.isna(b_ret):
        return np.nan

    return p_ret - b_ret


def _member_snapshot(ticker: str) -> dict:
    df = fetch_history(ticker, years=3)

    if df.empty:
        return {
            "Ticker": ticker,
            "Timing Score": np.nan,
            "Action": "NO DATA",
            "Trend": "No data",
            "1M Momentum %": np.nan,
            "3M Momentum %": np.nan,
        }

    timing = build_timing_snapshot(df)

    return {
        "Ticker": ticker,
        "Timing Score": timing.get("timing_score", np.nan),
        "Action": timing.get("action", "NO DATA"),
        "Trend": timing.get("trend", "No data"),
        "1M Momentum %": timing.get("momentum_1m", np.nan),
        "3M Momentum %": timing.get("momentum_3m", np.nan),
    }


def build_theme_rankings() -> pd.DataFrame:
    rows = []

    for theme_name, cfg in THEMES.items():
        proxy = cfg.get("proxy", "")
        benchmark = cfg.get("benchmark", "SPY")
        members = cfg.get("members", [])

        proxy_df = fetch_history(proxy, years=5)
        bench_df = fetch_history(benchmark, years=5)

        if proxy_df.empty:
            rows.append(
                {
                    "Theme": theme_name,
                    "Proxy": proxy,
                    "Theme Score": np.nan,
                    "Stage": "No data",
                    "Proxy Action": "NO DATA",
                    "Proxy Trend": "No data",
                    "Proxy Timing": np.nan,
                    "1M Momentum %": np.nan,
                    "3M Momentum %": np.nan,
                    "RS 3M %": np.nan,
                    "RS 6M %": np.nan,
                    "Breadth %": np.nan,
                    "BUY Ratio %": np.nan,
                    "Avg Member Timing": np.nan,
                }
            )
            continue

        proxy_timing = build_timing_snapshot(proxy_df)

        rs_3m = _relative_strength(proxy_df, bench_df, 63)
        rs_6m = _relative_strength(proxy_df, bench_df, 126)

        member_rows = [_member_snapshot(m) for m in members]
        member_df = pd.DataFrame(member_rows)

        avg_member_score = np.nan
        breadth = np.nan
        buy_ratio = np.nan

        if not member_df.empty:
            score_series = pd.to_numeric(member_df["Timing Score"], errors="coerce")
            if not score_series.dropna().empty:
                avg_member_score = float(score_series.dropna().mean())

            trend_series = member_df["Trend"].astype(str)
            if len(trend_series) > 0:
                breadth = float(trend_series.isin(["Bullish", "Positive"]).mean() * 100.0)

            action_series = member_df["Action"].astype(str)
            if len(action_series) > 0:
                buy_ratio = float((action_series == "BUY").mean() * 100.0)

        theme_score = 0.0

        proxy_timing_score = proxy_timing.get("timing_score", np.nan)
        if pd.notna(proxy_timing_score):
            theme_score += float(proxy_timing_score) * 0.45

        if pd.notna(avg_member_score):
            theme_score += float(avg_member_score) * 0.30

        if pd.notna(rs_3m):
            theme_score += max(-15.0, min(15.0, float(rs_3m))) * 0.80

        if pd.notna(rs_6m):
            theme_score += max(-15.0, min(15.0, float(rs_6m))) * 0.50

        if pd.notna(breadth):
            theme_score += (float(breadth) - 50.0) * 0.15

        theme_score = max(0.0, min(100.0, theme_score))

        if theme_score >= 70:
            stage = "Leading"
        elif theme_score >= 55:
            stage = "Emerging"
        elif theme_score >= 40:
            stage = "Neutral"
        else:
            stage = "Weakening"

        rows.append(
            {
                "Theme": theme_name,
                "Proxy": proxy,
                "Theme Score": round(theme_score, 2),
                "Stage": stage,
                "Proxy Action": proxy_timing.get("action", "NO DATA"),
                "Proxy Trend": proxy_timing.get("trend", "No data"),
                "Proxy Timing": proxy_timing.get("timing_score", np.nan),
                "1M Momentum %": proxy_timing.get("momentum_1m", np.nan),
                "3M Momentum %": proxy_timing.get("momentum_3m", np.nan),
                "RS 3M %": round(rs_3m, 2) if pd.notna(rs_3m) else np.nan,
                "RS 6M %": round(rs_6m, 2) if pd.notna(rs_6m) else np.nan,
                "Breadth %": round(breadth, 2) if pd.notna(breadth) else np.nan,
                "BUY Ratio %": round(buy_ratio, 2) if pd.notna(buy_ratio) else np.nan,
                "Avg Member Timing": round(avg_member_score, 2) if pd.notna(avg_member_score) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if "Theme Score" in out.columns:
        out = out.sort_values("Theme Score", ascending=False, na_position="last").reset_index(drop=True)
    return out


def theme_deep_dive(theme_name: str) -> dict:
    cfg = THEMES.get(theme_name, {})
    if not cfg:
        return {}

    proxy = cfg.get("proxy", "")
    benchmark = cfg.get("benchmark", "SPY")
    members = cfg.get("members", [])
    etfs = cfg.get("etfs", [])
    leaders = cfg.get("leaders", [])
    drivers = cfg.get("drivers", [])
    headwinds = cfg.get("headwinds", [])
    description = cfg.get("description", "")

    proxy_df = fetch_history(proxy, years=5)
    bench_df = fetch_history(benchmark, years=5)

    proxy_timing = build_timing_snapshot(proxy_df) if not proxy_df.empty else {}

    rs_3m = _relative_strength(proxy_df, bench_df, 63)
    rs_6m = _relative_strength(proxy_df, bench_df, 126)

    member_rows = [_member_snapshot(m) for m in members]
    members_df = pd.DataFrame(member_rows)

    if not members_df.empty and "Timing Score" in members_df.columns:
        members_df = members_df.sort_values("Timing Score", ascending=False, na_position="last").reset_index(drop=True)

    summary_text = []
    if proxy_timing:
        action = proxy_timing.get("action", "NO DATA")
        trend = proxy_timing.get("trend", "No data")
        summary_text.append(f"Proxy {proxy} er aktuelt i trend '{trend}' med signal '{action}'.")
    if pd.notna(rs_3m):
        summary_text.append(f"Relativ styrke mod benchmark over 3 måneder er {rs_3m:.2f}%.")
    if pd.notna(rs_6m):
        summary_text.append(f"Relativ styrke mod benchmark over 6 måneder er {rs_6m:.2f}%.")

    return {
        "theme": theme_name,
        "proxy": proxy,
        "benchmark": benchmark,
        "description": description,
        "proxy_timing": proxy_timing,
        "rs_3m": round(rs_3m, 2) if pd.notna(rs_3m) else None,
        "rs_6m": round(rs_6m, 2) if pd.notna(rs_6m) else None,
        "drivers": drivers,
        "headwinds": headwinds,
        "etfs": etfs,
        "leaders": leaders,
        "members_df": members_df,
        "summary_text": " ".join(summary_text),
    }