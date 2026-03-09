# src/history_engine.py
from __future__ import annotations

import pandas as pd

from src.storage_engine import add_recent_asset, load_recent_assets


def register_recent_view(ticker: str) -> None:
    add_recent_asset(ticker)


def recent_assets_df() -> pd.DataFrame:
    items = load_recent_assets()
    if not items:
        return pd.DataFrame(columns=["Ticker"])
    return pd.DataFrame({"Ticker": items})