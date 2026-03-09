# src/quant_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_pct(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a / b - 1.0) * 100.0


def build_quant_snapshot(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Close" not in df.columns:
        return {
            "last": np.nan,
            "ret_1m": np.nan,
            "ret_3m": np.nan,
            "ret_6m": np.nan,
            "vol_3m": np.nan,
            "ema50_gap_pct": np.nan,
            "ema200_gap_pct": np.nan,
            "drawdown_6m_pct": np.nan,
            "momentum_score": np.nan,
            "trend_score": np.nan,
            "low_vol_score": np.nan,
            "quant_score": np.nan,
        }

    d = df.copy()
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna(subset=["Close"]).reset_index(drop=True)

    if len(d) < 80:
        return {
            "last": np.nan,
            "ret_1m": np.nan,
            "ret_3m": np.nan,
            "ret_6m": np.nan,
            "vol_3m": np.nan,
            "ema50_gap_pct": np.nan,
            "ema200_gap_pct": np.nan,
            "drawdown_6m_pct": np.nan,
            "momentum_score": np.nan,
            "trend_score": np.nan,
            "low_vol_score": np.nan,
            "quant_score": np.nan,
        }

    close = d["Close"]
    last = float(close.iloc[-1])

    ret_1m = _safe_pct(float(close.iloc[-1]), float(close.iloc[-21])) if len(close) >= 21 else np.nan
    ret_3m = _safe_pct(float(close.iloc[-1]), float(close.iloc[-63])) if len(close) >= 63 else np.nan
    ret_6m = _safe_pct(float(close.iloc[-1]), float(close.iloc[-126])) if len(close) >= 126 else np.nan

    returns = close.pct_change().dropna()
    vol_3m = float(returns.iloc[-63:].std() * 100.0) if len(returns) >= 63 else np.nan

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    ema50_gap_pct = _safe_pct(last, float(ema50.iloc[-1])) if len(ema50) > 0 else np.nan
    ema200_gap_pct = _safe_pct(last, float(ema200.iloc[-1])) if len(ema200) > 0 else np.nan

    if len(close) >= 126:
        peak_6m = float(close.iloc[-126:].max())
        drawdown_6m_pct = _safe_pct(last, peak_6m)
    else:
        drawdown_6m_pct = np.nan

    momentum_score = 0.0
    if pd.notna(ret_1m):
        momentum_score += max(-10.0, min(10.0, ret_1m / 2.0))
    if pd.notna(ret_3m):
        momentum_score += max(-15.0, min(15.0, ret_3m / 2.5))
    if pd.notna(ret_6m):
        momentum_score += max(-15.0, min(15.0, ret_6m / 4.0))

    trend_score = 0.0
    if pd.notna(ema50_gap_pct) and ema50_gap_pct > 0:
        trend_score += 10.0
    if pd.notna(ema200_gap_pct) and ema200_gap_pct > 0:
        trend_score += 15.0
    if len(ema50) > 0 and len(ema200) > 0 and float(ema50.iloc[-1]) > float(ema200.iloc[-1]):
        trend_score += 15.0

    low_vol_score = 0.0
    if pd.notna(vol_3m):
        if vol_3m < 2.0:
            low_vol_score = 15.0
        elif vol_3m < 3.0:
            low_vol_score = 10.0
        elif vol_3m < 4.0:
            low_vol_score = 5.0
        elif vol_3m > 6.0:
            low_vol_score = -10.0

    if pd.notna(drawdown_6m_pct):
        if drawdown_6m_pct < -25:
            low_vol_score -= 10.0
        elif drawdown_6m_pct < -15:
            low_vol_score -= 5.0

    quant_score = momentum_score + trend_score + low_vol_score
    quant_score = max(0.0, min(100.0, quant_score))

    return {
        "last": round(last, 2),
        "ret_1m": round(ret_1m, 2) if pd.notna(ret_1m) else np.nan,
        "ret_3m": round(ret_3m, 2) if pd.notna(ret_3m) else np.nan,
        "ret_6m": round(ret_6m, 2) if pd.notna(ret_6m) else np.nan,
        "vol_3m": round(vol_3m, 2) if pd.notna(vol_3m) else np.nan,
        "ema50_gap_pct": round(ema50_gap_pct, 2) if pd.notna(ema50_gap_pct) else np.nan,
        "ema200_gap_pct": round(ema200_gap_pct, 2) if pd.notna(ema200_gap_pct) else np.nan,
        "drawdown_6m_pct": round(drawdown_6m_pct, 2) if pd.notna(drawdown_6m_pct) else np.nan,
        "momentum_score": round(momentum_score, 2),
        "trend_score": round(trend_score, 2),
        "low_vol_score": round(low_vol_score, 2),
        "quant_score": round(quant_score, 2),
    }