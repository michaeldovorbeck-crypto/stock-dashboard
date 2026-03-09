from __future__ import annotations

import pandas as pd

from src.strategy_engine import top_leaders
from src.strategy_engine import top_etfs
from src.theme_definitions import THEMES


def peers_from_theme(theme_text: str, max_peers: int = 5) -> list[str]:
    """
    Finder peers ud fra tema.
    """
    if not theme_text:
        return []

    peers = []

    for theme, data in THEMES.items():
        if theme.lower() in theme_text.lower():
            peers.extend(data.get("tickers", []))

    return list(dict.fromkeys(peers))[:max_peers]


def peers_from_sector(df: pd.DataFrame, sector: str, ticker: str, max_peers: int = 5) -> list[str]:
    """
    Finder peers fra samme sektor i universe dataframe.
    """
    if df.empty or not sector:
        return []

    work = df.copy()
    work = work[work["sector"] == sector]

    peers = work["ticker"].tolist()

    peers = [x for x in peers if x != ticker]

    return peers[:max_peers]


def peers_from_leaders(max_peers: int = 5) -> list[str]:
    """
    Markedsledere.
    """
    leaders_df = top_leaders(20)

    if leaders_df.empty:
        return []

    return leaders_df["Ticker"].head(max_peers).tolist()


def peers_from_strategy_etfs(max_peers: int = 5) -> list[str]:
    """
    Stærkeste ETF'er.
    """
    etf_df = top_etfs(20)

    if etf_df.empty:
        return []

    return etf_df["Ticker"].head(max_peers).tolist()


def build_peer_group(record: dict, universe_df: pd.DataFrame | None = None) -> list[str]:
    """
    Samler peers fra flere kilder.
    """

    ticker = record.get("ticker")
    sector = record.get("sector", "")
    themes = record.get("themes", "")

    peers = []

    peers += peers_from_theme(themes)
    peers += peers_from_leaders()
    peers += peers_from_strategy_etfs()

    if universe_df is not None:
        peers += peers_from_sector(universe_df, sector, ticker)

    # fjern duplicates
    peers = list(dict.fromkeys(peers))

    if ticker in peers:
        peers.remove(ticker)

    return peers[:6]