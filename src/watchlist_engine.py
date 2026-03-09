# src/watchlist_engine.py
from __future__ import annotations

import pandas as pd


def add_to_watchlist(watchlist: list[str], ticker: str) -> list[str]:
    t = str(ticker).strip().upper()
    if not t:
        return watchlist

    out = [str(x).strip().upper() for x in watchlist if str(x).strip()]
    if t not in out:
        out.append(t)
    return out


def remove_from_watchlist(watchlist: list[str], ticker: str) -> list[str]:
    t = str(ticker).strip().upper()
    return [str(x).strip().upper() for x in watchlist if str(x).strip().upper() != t]


def watchlist_to_df(watchlist: list[str]) -> pd.DataFrame:
    if not watchlist:
        return pd.DataFrame(columns=["Ticker"])

    rows = [{"Ticker": str(x).strip().upper()} for x in watchlist if str(x).strip()]
    if not rows:
        return pd.DataFrame(columns=["Ticker"])

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)