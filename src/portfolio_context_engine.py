# src/portfolio_context_engine.py
from __future__ import annotations

import pandas as pd


def build_portfolio_context(ticker: str, portfolio_positions: list[dict]) -> dict:
    if not portfolio_positions:
        return {
            "owned": False,
            "positions_df": pd.DataFrame(),
            "total_shares": 0.0,
            "accounts": "",
        }

    df = pd.DataFrame(portfolio_positions)
    if df.empty or "Ticker" not in df.columns:
        return {
            "owned": False,
            "positions_df": pd.DataFrame(),
            "total_shares": 0.0,
            "accounts": "",
        }

    work = df.copy()
    work["Ticker"] = work["Ticker"].astype(str).str.upper().str.strip()

    t = str(ticker).strip().upper()
    match = work[work["Ticker"] == t].copy()

    if match.empty:
        return {
            "owned": False,
            "positions_df": pd.DataFrame(),
            "total_shares": 0.0,
            "accounts": "",
        }

    total_shares = pd.to_numeric(match["Antal"], errors="coerce").fillna(0).sum()
    accounts = ", ".join(sorted(match["Konto"].astype(str).dropna().unique().tolist()))

    return {
        "owned": True,
        "positions_df": match.reset_index(drop=True),
        "total_shares": round(float(total_shares), 2),
        "accounts": accounts,
    }