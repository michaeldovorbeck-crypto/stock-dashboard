# src/overview_engine.py
from __future__ import annotations

import pandas as pd

from src.cache_engine import load_snapshot
from src.discovery_engine import top_discovery_candidates
from src.macro_engine import macro_snapshot
from src.strategy_engine import top_etfs, top_leaders
from src.theme_engine import build_theme_rankings


def build_market_overview() -> dict:
    macro = macro_snapshot()
    theme_df = build_theme_rankings()
    discovery_df = top_discovery_candidates(5)
    etf_df = top_etfs(5)
    leader_df = top_leaders(5)
    snapshot_df = load_snapshot()

    top_theme = ""
    if not theme_df.empty and "Theme" in theme_df.columns:
        top_theme = str(theme_df.iloc[0]["Theme"])

    top_discovery = ""
    if not discovery_df.empty and "Theme" in discovery_df.columns:
        top_discovery = str(discovery_df.iloc[0]["Theme"])

    top_etf = ""
    if not etf_df.empty and "Ticker" in etf_df.columns:
        top_etf = str(etf_df.iloc[0]["Ticker"])

    top_leader = ""
    if not leader_df.empty and "Ticker" in leader_df.columns:
        top_leader = str(leader_df.iloc[0]["Ticker"])

    buy_count = 0
    if not snapshot_df.empty and "Action" in snapshot_df.columns:
        buy_count = int((snapshot_df["Action"].astype(str).str.upper() == "BUY").sum())

    return {
        "macro": macro,
        "top_theme": top_theme,
        "top_discovery": top_discovery,
        "top_etf": top_etf,
        "top_leader": top_leader,
        "snapshot_buy_count": buy_count,
        "theme_df": theme_df,
        "discovery_df": discovery_df,
        "etf_df": etf_df,
        "leader_df": leader_df,
        "snapshot_df": snapshot_df,
    }


def build_quick_picks() -> list[str]:
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "SPY", "QQQ", "NOVO-B.CO", "ASML", "TSM"]