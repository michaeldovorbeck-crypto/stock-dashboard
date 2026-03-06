from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False).mean()


def drawdown(close: pd.Series, window: int = 252) -> pd.Series:
    c = close.astype(float)
    rolling_max = c.rolling(window).max()
    dd = (c / rolling_max) - 1.0
    return dd


def compute_signals(ohlcv: pd.DataFrame) -> dict:
    """
    Input: DataFrame med kolonner mindst: Date, Close
    Output: dict med nøgletal + simple signal-flags
    """
    if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
        return {}

    df = ohlcv.copy()

    # Sørg for sortering og numeric
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if len(df) < 60:
        return {}

    close = df["Close"]
    last = float(close.iloc[-1])

    # Indikatorer
    r = rsi(close, 14)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200) if len(df) >= 200 else pd.Series([np.nan] * len(df), index=df.index)

    trend_up = bool(ema50.iloc[-1] > ema200.iloc[-1]) if len(df) >= 200 else bool(ema50.iloc[-1] > ema50.iloc[-20])
    rsi_now = float(r.iloc[-1]) if not np.isnan(r.iloc[-1]) else np.nan
    rsi_prev = float(r.iloc[-6]) if len(r) > 6 and not np.isnan(r.iloc[-6]) else np.nan

    # Volatilitet (20d)
    ret = close.pct_change()
    vol20 = float(ret.rolling(20).std().iloc[-1] * 100) if len(df) >= 21 else np.nan

    # Drawdown sidste 3 mdr ~ 63 dage
    dd63 = drawdown(close, 63)
    dd3m = float(dd63.iloc[-1]) if not np.isnan(dd63.iloc[-1]) else np.nan

    # Score (simpel heuristik)
    score = 0.0
    if trend_up:
        score += 50
    if not np.isnan(rsi_now):
        # “sund” RSI zone 45-60
        score += max(0, 30 - abs(rsi_now - 52))
    if not np.isnan(rsi_prev) and not np.isnan(rsi_now) and rsi_now > rsi_prev:
        score += 10
    if not np.isnan(dd3m):
        # mindre drawdown = bedre
        score += max(0, 10 - abs(dd3m) * 100)

    # Købs-zone flag (tidlig)
    buy_early = bool(trend_up and (not np.isnan(rsi_now)) and 40 <= rsi_now <= 60 and (np.isnan(rsi_prev) or rsi_now >= rsi_prev))

    # Risiko flag
    risk = "OK"
    if not np.isnan(vol20) and vol20 > 4.5:
        risk = "Høj"
    if not np.isnan(dd3m) and dd3m < -0.15:
        risk = "Høj"

    return {
        "last": last,
        "rsi": rsi_now,
        "vol20": vol20,
        "drawdown3m": dd3m,
        "trend_up": trend_up,
        "buy_early": buy_early,
        "score": float(round(score, 2)),
        "risk": risk,
    }


def trade_label(sig: dict) -> str:
    """
    Dansk label til UI: KØB / HOLD / SÆLG (helt simpelt)
    """
    if not sig:
        return "INGEN DATA"

    if sig.get("risk") == "Høj":
        return "HOLD (risiko)"
    if sig.get("buy_early"):
        return "KØB (tidlig)"
    if sig.get("trend_up"):
        return "HOLD"
    return "SÆLG / UNDGÅ"
