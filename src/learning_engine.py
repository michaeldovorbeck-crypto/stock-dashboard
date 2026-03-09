# src/learning_engine.py
from __future__ import annotations

import pandas as pd

from src.signal_log_engine import read_signal_log


def build_learning_summary() -> dict:
    df = read_signal_log(limit=5000)
    if df.empty:
        return {
            "logs": 0,
            "buy_logs": 0,
            "hold_logs": 0,
            "sell_logs": 0,
            "top_sources": pd.DataFrame(),
            "top_tickers": pd.DataFrame(),
        }

    actions = df["action"].astype(str).str.upper()

    top_sources = (
        df.groupby("source", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    top_tickers = (
        df.groupby("ticker", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "logs": int(len(df)),
        "buy_logs": int((actions == "BUY").sum()),
        "hold_logs": int((actions == "HOLD").sum()),
        "sell_logs": int((actions == "SELL").sum()),
        "top_sources": top_sources,
        "top_tickers": top_tickers,
    }