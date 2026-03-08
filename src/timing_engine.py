import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    d = df.copy()
    d["High"] = pd.to_numeric(d["High"], errors="coerce")
    d["Low"] = pd.to_numeric(d["Low"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")

    prev_close = d["Close"].shift(1)

    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - prev_close).abs()
    tr3 = (d["Low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_timing_snapshot(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Close" not in df.columns:
        return {
            "timing_score": np.nan,
            "trend": "No data",
            "momentum_1m": np.nan,
            "momentum_3m": np.nan,
            "rsi": np.nan,
            "atr_pct": np.nan,
            "action": "NO DATA",
        }

    d = df.copy()
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d["Open"] = pd.to_numeric(d.get("Open"), errors="coerce")
    d["High"] = pd.to_numeric(d.get("High"), errors="coerce")
    d["Low"] = pd.to_numeric(d.get("Low"), errors="coerce")
    d = d.dropna(subset=["Close"]).reset_index(drop=True)

    if len(d) < 30:
        return {
            "timing_score": np.nan,
            "trend": "Too little data",
            "momentum_1m": np.nan,
            "momentum_3m": np.nan,
            "rsi": np.nan,
            "atr_pct": np.nan,
            "action": "WAIT",
        }

    close = d["Close"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    rsi14 = rsi(close, 14)

    atr14 = atr(d, 14)
    last = float(close.iloc[-1])

    mom_1m = (close.iloc[-1] / close.iloc[-21] - 1.0) * 100 if len(close) >= 21 else np.nan
    mom_3m = (close.iloc[-1] / close.iloc[-63] - 1.0) * 100 if len(close) >= 63 else np.nan
    atr_pct = float(atr14.iloc[-1] / last * 100) if pd.notna(atr14.iloc[-1]) and last != 0 else np.nan
    rsi_last = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else np.nan

    trend_score = 0
    if last > ema20.iloc[-1]:
        trend_score += 10
    if ema20.iloc[-1] > ema50.iloc[-1]:
        trend_score += 10
    if ema50.iloc[-1] > ema200.iloc[-1]:
        trend_score += 20

    momentum_score = 0
    if pd.notna(mom_1m):
        momentum_score += max(-10, min(10, mom_1m / 2))
    if pd.notna(mom_3m):
        momentum_score += max(-10, min(10, mom_3m / 4))

    rsi_score = 0
    if pd.notna(rsi_last):
        if 50 <= rsi_last <= 65:
            rsi_score = 15
        elif 40 <= rsi_last < 50 or 65 < rsi_last <= 75:
            rsi_score = 8
        else:
            rsi_score = 0

    risk_penalty = 0
    if pd.notna(atr_pct):
        if atr_pct > 5:
            risk_penalty = -10
        elif atr_pct > 3.5:
            risk_penalty = -5

    total_score = trend_score + momentum_score + rsi_score + risk_penalty
    total_score = round(max(0, min(100, total_score)), 2)

    if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
        trend = "Bullish"
    elif ema50.iloc[-1] > ema200.iloc[-1]:
        trend = "Positive"
    elif ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
        trend = "Bearish"
    else:
        trend = "Mixed"

    if total_score >= 65:
        action = "BUY"
    elif total_score >= 40:
        action = "HOLD"
    else:
        action = "SELL"

    return {
        "timing_score": total_score,
        "trend": trend,
        "momentum_1m": round(mom_1m, 2) if pd.notna(mom_1m) else np.nan,
        "momentum_3m": round(mom_3m, 2) if pd.notna(mom_3m) else np.nan,
        "rsi": round(rsi_last, 2) if pd.notna(rsi_last) else np.nan,
        "atr_pct": round(atr_pct, 2) if pd.notna(atr_pct) else np.nan,
        "action": action,
    }