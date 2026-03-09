# src/analysis_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_sources import fetch_history
from src.macro_engine import macro_snapshot
from src.search_engine import find_asset_record
from src.strategy_engine import build_strategy_candidates
from src.theme_engine import build_theme_rankings
from src.timing_engine import build_timing_snapshot


def _safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _returns(close: pd.Series) -> dict:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if c.empty:
        return {"1D": np.nan, "1M": np.nan, "3M": np.nan, "6M": np.nan}

    out = {"1D": np.nan, "1M": np.nan, "3M": np.nan, "6M": np.nan}

    if len(c) >= 2:
        out["1D"] = (float(c.iloc[-1]) / float(c.iloc[-2]) - 1.0) * 100.0
    if len(c) >= 21:
        out["1M"] = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1.0) * 100.0
    if len(c) >= 63:
        out["3M"] = (float(c.iloc[-1]) / float(c.iloc[-63]) - 1.0) * 100.0
    if len(c) >= 126:
        out["6M"] = (float(c.iloc[-1]) / float(c.iloc[-126]) - 1.0) * 100.0

    return out


def _theme_context(themes_text: str) -> pd.DataFrame:
    ranking = build_theme_rankings()
    if ranking.empty:
        return pd.DataFrame()

    themes = [x.strip() for x in str(themes_text).split(",") if x.strip()]
    if not themes:
        return pd.DataFrame()

    out = ranking[ranking["Theme"].isin(themes)].copy()
    if out.empty:
        return pd.DataFrame()

    return out.reset_index(drop=True)


def _strategy_context(ticker: str) -> pd.DataFrame:
    df = build_strategy_candidates()
    if df.empty:
        return pd.DataFrame()

    out = df[df["Ticker"].astype(str).str.upper() == str(ticker).strip().upper()].copy()
    if out.empty:
        return pd.DataFrame()

    return out.reset_index(drop=True)


def build_asset_analysis(ticker: str, years: int = 5) -> dict:
    t = (ticker or "").strip()
    if not t:
        return {}

    record = find_asset_record(t)
    df = fetch_history(t, years=years)

    if df.empty:
        return {
            "ticker": t,
            "record": record,
            "has_data": False,
            "message": "Ingen data fundet.",
        }

    timing = build_timing_snapshot(df)
    macro = macro_snapshot()

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    last = float(close.iloc[-1]) if not close.empty else np.nan
    rets = _returns(close)

    theme_context_df = _theme_context(record.get("themes", ""))
    strategy_context_df = _strategy_context(t)

    return {
        "ticker": t.upper(),
        "record": record,
        "has_data": True,
        "df": df,
        "last": round(last, 2) if pd.notna(last) else np.nan,
        "returns": {k: (round(v, 2) if pd.notna(v) else np.nan) for k, v in rets.items()},
        "timing": timing,
        "macro": macro,
        "theme_context_df": theme_context_df,
        "strategy_context_df": strategy_context_df,
    }