from __future__ import annotations

import pandas as pd


def _to_float(value):
    try:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _score_bucket(score: float | None) -> str:
    if score is None:
        return "Ukendt"
    if score >= 67:
        return "Høj"
    if score >= 34:
        return "Mellem"
    return "Lav"


def _signal_points(action: str) -> int:
    act = str(action or "").upper()
    if act == "BUY":
        return 2
    if act == "HOLD":
        return 0
    if act == "SELL":
        return -2
    return 0


def _macro_points(regime: str) -> int:
    reg = str(regime or "").strip().lower()
    if reg == "risk-on":
        return 1
    if reg == "risk-off":
        return -1
    return 0


def build_recommendation(analysis: dict, diag: dict) -> dict:
    timing = analysis.get("timing", {})
    macro = analysis.get("macro", {})
    record = analysis.get("record", {})

    timing_score = _to_float(timing.get("timing_score"))
    quant_score = _to_float(analysis.get("quant_score"))
    rsi = _to_float(timing.get("rsi"))
    atr_pct = _to_float(timing.get("atr_pct"))
    momentum_1m = _to_float(timing.get("momentum_1m"))
    momentum_3m = _to_float(timing.get("momentum_3m"))
    action = str(timing.get("action", "")).upper()
    regime = str(macro.get("regime", ""))

    rows = len(diag.get("df", pd.DataFrame()))
    has_theme = bool(str(record.get("themes", "")).strip())

    score = 50

    # Hovedsignal
    score += _signal_points(action) * 8

    # Timing / quant
    if timing_score is not None:
        score += (timing_score - 50) * 0.35
    if quant_score is not None:
        score += (quant_score - 50) * 0.25

    # Momentum
    if momentum_1m is not None:
        score += 6 if momentum_1m > 0 else -6
    if momentum_3m is not None:
        score += 8 if momentum_3m > 0 else -8

    # RSI
    if rsi is not None:
        if 45 <= rsi <= 65:
            score += 4
        elif rsi > 75:
            score -= 5
        elif rsi < 25:
            score -= 3

    # Volatilitet
    if atr_pct is not None:
        if atr_pct > 8:
            score -= 5
        elif atr_pct < 4:
            score += 2

    # Makro
    score += _macro_points(regime) * 4

    # Theme support
    if has_theme:
        score += 3

    # Datakvalitet
    if rows >= 750:
        score += 4
    elif rows >= 250:
        score += 2
    else:
        score -= 4

    score = max(0, min(100, round(score, 1)))

    if score >= 72:
        recommendation = "KØB"
        recommendation_color = "buy"
    elif score >= 45:
        recommendation = "HOLD"
        recommendation_color = "hold"
    else:
        recommendation = "SÆLG"
        recommendation_color = "sell"

    reasons = []

    if action == "BUY":
        reasons.append("det tekniske signal er BUY")
    elif action == "SELL":
        reasons.append("det tekniske signal er SELL")

    if timing_score is not None:
        reasons.append(f"timing score er {_score_bucket(timing_score).lower()} ({timing_score:.0f})")

    if quant_score is not None:
        reasons.append(f"quant score er {_score_bucket(quant_score).lower()} ({quant_score:.0f})")

    if momentum_1m is not None:
        reasons.append("1M momentum er positivt" if momentum_1m > 0 else "1M momentum er negativt")

    if momentum_3m is not None:
        reasons.append("3M momentum er positivt" if momentum_3m > 0 else "3M momentum er negativt")

    if regime:
        reasons.append(f"makroregimet er {regime}")

    if has_theme:
        reasons.append("aktivet understøttes af tema/strategikontekst")

    return {
        "score": score,
        "recommendation": recommendation,
        "recommendation_color": recommendation_color,
        "timing_bucket": _score_bucket(timing_score),
        "quant_bucket": _score_bucket(quant_score),
        "rsi_bucket": (
            "Høj" if (rsi is not None and rsi > 70)
            else "Lav" if (rsi is not None and rsi < 30)
            else "Mellem" if rsi is not None
            else "Ukendt"
        ),
        "vol_bucket": (
            "Høj" if (atr_pct is not None and atr_pct > 8)
            else "Lav" if (atr_pct is not None and atr_pct < 4)
            else "Mellem" if atr_pct is not None
            else "Ukendt"
        ),
        "data_bucket": (
            "Høj" if rows >= 750
            else "Mellem" if rows >= 250
            else "Lav"
        ),
        "reasons": reasons[:6],
    }