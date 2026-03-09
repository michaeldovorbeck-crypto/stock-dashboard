# src/precompute_engine.py
from __future__ import annotations

import pandas as pd

from src.cache_engine import save_snapshot
from src.data_sources import fetch_history, load_universe_csv
from src.quant_engine import build_quant_snapshot
from src.screening_engine import _theme_lookup_for_ticker
from src.timing_engine import build_timing_snapshot


def build_quant_snapshot_for_universe(
    filename: str,
    years: int = 3,
    max_tickers: int | None = None,
) -> tuple[pd.DataFrame, str]:
    uni_df, status = load_universe_csv(filename)
    if uni_df.empty:
        return pd.DataFrame(), status

    work = uni_df.copy()
    if max_tickers is not None:
        work = work.head(max_tickers)

    rows = []

    for _, row in work.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        name = str(row.get("name", "")).strip()
        sector = str(row.get("sector", "")).strip()
        country = str(row.get("country", "")).strip()

        if not ticker:
            continue

        df = fetch_history(ticker, years=years)
        if df.empty:
            continue

        timing = build_timing_snapshot(df)
        quant = build_quant_snapshot(df)
        themes = _theme_lookup_for_ticker(ticker)

        rows.append(
            {
                "Ticker": ticker,
                "Name": name,
                "Sector": sector,
                "Country": country,
                "Themes": ", ".join(themes),
                "Theme Count": len(themes),
                "Last": quant["last"],
                "Timing Score": timing.get("timing_score"),
                "Action": timing.get("action"),
                "Trend": timing.get("trend"),
                "RSI": timing.get("rsi"),
                "ATR %": timing.get("atr_pct"),
                "1M Momentum %": timing.get("momentum_1m"),
                "3M Momentum %": timing.get("momentum_3m"),
                "Q Ret 1M %": quant["ret_1m"],
                "Q Ret 3M %": quant["ret_3m"],
                "Q Ret 6M %": quant["ret_6m"],
                "Q Vol 3M %": quant["vol_3m"],
                "Q EMA50 Gap %": quant["ema50_gap_pct"],
                "Q EMA200 Gap %": quant["ema200_gap_pct"],
                "Q DD 6M %": quant["drawdown_6m_pct"],
                "Momentum Score": quant["momentum_score"],
                "Trend Score": quant["trend_score"],
                "Low Vol Score": quant["low_vol_score"],
                "Quant Score": quant["quant_score"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, "Ingen snapshot-data bygget."

    out = out.sort_values(
        ["Quant Score", "Timing Score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    save_snapshot(out)

    return out, f"Snapshot bygget: {len(out)} rækker fra {filename}"