# src/technical_view_engine.py
from __future__ import annotations

import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_technical_view(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d["High"] = pd.to_numeric(d.get("High"), errors="coerce")
    d["Low"] = pd.to_numeric(d.get("Low"), errors="coerce")
    d = d.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()
    d["RSI14"] = rsi(d["Close"], 14)

    if "High" in d.columns and "Low" in d.columns:
        prev_close = d["Close"].shift(1)
        tr1 = d["High"] - d["Low"]
        tr2 = (d["High"] - prev_close).abs()
        tr3 = (d["Low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
        d["ATR_PCT"] = d["ATR14"] / d["Close"] * 100.0
    else:
        d["ATR14"] = np.nan
        d["ATR_PCT"] = np.nan

    return d