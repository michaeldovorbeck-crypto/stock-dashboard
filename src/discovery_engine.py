# src/discovery_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.theme_definitions import THEMES
from src.theme_engine import build_theme_rankings


def build_discovery_table() -> pd.DataFrame:
    ranking_df = build_theme_rankings()
    if ranking_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in ranking_df.iterrows():
        theme = row.get("Theme")
        proxy_timing = pd.to_numeric(row.get("Proxy Timing"), errors="coerce")
        mom_1m = pd.to_numeric(row.get("1M Momentum %"), errors="coerce")
        mom_3m = pd.to_numeric(row.get("3M Momentum %"), errors="coerce")
        rs_3m = pd.to_numeric(row.get("RS 3M %"), errors="coerce")
        rs_6m = pd.to_numeric(row.get("RS 6M %"), errors="coerce")
        breadth = pd.to_numeric(row.get("Breadth %"), errors="coerce")
        buy_ratio = pd.to_numeric(row.get("BUY Ratio %"), errors="coerce")
        avg_member_timing = pd.to_numeric(row.get("Avg Member Timing"), errors="coerce")

        discovery_score = 0.0

        if pd.notna(proxy_timing):
            discovery_score += proxy_timing * 0.30

        if pd.notna(avg_member_timing):
            discovery_score += avg_member_timing * 0.20

        if pd.notna(rs_3m):
            discovery_score += max(-15.0, min(15.0, float(rs_3m))) * 1.2

        if pd.notna(rs_6m):
            discovery_score += max(-15.0, min(15.0, float(rs_6m))) * 0.7

        if pd.notna(mom_1m):
            discovery_score += max(-10.0, min(10.0, float(mom_1m) / 2.0))

        if pd.notna(mom_3m):
            discovery_score += max(-10.0, min(10.0, float(mom_3m) / 4.0))

        if pd.notna(breadth):
            discovery_score += (float(breadth) - 50.0) * 0.18

        if pd.notna(buy_ratio):
            discovery_score += (float(buy_ratio) - 50.0) * 0.12

        discovery_score = max(0.0, min(100.0, discovery_score))

        acceleration = "Neutral"
        if pd.notna(mom_1m) and pd.notna(mom_3m):
            if mom_1m > mom_3m and mom_1m > 0:
                acceleration = "Accelerating"
            elif mom_1m < mom_3m and mom_3m > 0:
                acceleration = "Decelerating"
            elif mom_1m < 0 and mom_3m < 0:
                acceleration = "Weakening"

        stage = "Neutral"
        if discovery_score >= 72:
            stage = "Leading"
        elif discovery_score >= 58:
            stage = "Emerging"
        elif discovery_score < 40:
            stage = "Weakening"

        why_now = []

        if pd.notna(rs_3m) and rs_3m > 3:
            why_now.append("stærk relativ styrke")
        if pd.notna(mom_1m) and mom_1m > 5:
            why_now.append("positiv kort momentum")
        if pd.notna(breadth) and breadth > 60:
            why_now.append("forbedret breadth")
        if pd.notna(buy_ratio) and buy_ratio > 50:
            why_now.append("flere BUY-signaler")
        if pd.notna(proxy_timing) and proxy_timing > 65:
            why_now.append("stærk proxy timing")

        if not why_now:
            why_now.append("ingen tydelig acceleration endnu")

        rows.append(
            {
                "Theme": theme,
                "Discovery Score": round(discovery_score, 2),
                "Stage": stage,
                "Acceleration": acceleration,
                "Why now": ", ".join(why_now[:3]),
                "Proxy Timing": round(proxy_timing, 2) if pd.notna(proxy_timing) else np.nan,
                "1M Momentum %": round(mom_1m, 2) if pd.notna(mom_1m) else np.nan,
                "3M Momentum %": round(mom_3m, 2) if pd.notna(mom_3m) else np.nan,
                "RS 3M %": round(rs_3m, 2) if pd.notna(rs_3m) else np.nan,
                "RS 6M %": round(rs_6m, 2) if pd.notna(rs_6m) else np.nan,
                "Breadth %": round(breadth, 2) if pd.notna(breadth) else np.nan,
                "BUY Ratio %": round(buy_ratio, 2) if pd.notna(buy_ratio) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("Discovery Score", ascending=False, na_position="last").reset_index(drop=True)


def discovery_deep_dive(theme_name: str) -> dict:
    table = build_discovery_table()
    if table.empty:
        return {}

    row = table[table["Theme"] == theme_name]
    if row.empty:
        return {}

    row = row.iloc[0]

    cfg = THEMES.get(theme_name, {})

    return {
        "theme": theme_name,
        "discovery_score": row.get("Discovery Score"),
        "stage": row.get("Stage"),
        "acceleration": row.get("Acceleration"),
        "why_now": row.get("Why now"),
        "drivers": cfg.get("drivers", []),
        "headwinds": cfg.get("headwinds", []),
        "proxy": cfg.get("proxy"),
        "leaders": cfg.get("leaders", []),
        "etfs": cfg.get("etfs", []),
    }


def top_discovery_candidates(top_n: int = 10) -> pd.DataFrame:
    df = build_discovery_table()
    if df.empty:
        return pd.DataFrame()
    return df.head(top_n).reset_index(drop=True)


def weakening_themes(top_n: int = 10) -> pd.DataFrame:
    df = build_discovery_table()
    if df.empty:
        return pd.DataFrame()

    weak = df[df["Stage"].isin(["Weakening"])].copy()
    if weak.empty:
        weak = df.sort_values("Discovery Score", ascending=True, na_position="last").head(top_n)

    return weak.head(top_n).reset_index(drop=True)