from __future__ import annotations

import pandas as pd

from src.technical_view_engine import build_technical_view


def _to_float(value):
    try:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _score_bucket(score: float | None) -> str:
    if score is None:
        return "Ukendt"
    if score >= 67:
        return "Høj"
    if score >= 34:
        return "Mellem"
    return "Lav"


def _signal_from_score(score: float | None) -> str:
    if score is None:
        return "HOLD"
    if score >= 63:
        return "KØB"
    if score <= 37:
        return "SÆLG"
    return "HOLD"


def build_technical_signal_history(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    tech_df = build_technical_view(price_df)
    if tech_df.empty or "Date" not in tech_df.columns or "Close" not in tech_df.columns:
        return pd.DataFrame()

    work = tech_df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    work["Momentum1M"] = pd.to_numeric(work["Close"], errors="coerce").pct_change(21) * 100.0
    work["Momentum3M"] = pd.to_numeric(work["Close"], errors="coerce").pct_change(63) * 100.0

    technical_scores = []
    technical_signals = []
    trend_states = []

    for _, row in work.iterrows():
        close = _to_float(row.get("Close"))
        ema20 = _to_float(row.get("EMA20"))
        ema50 = _to_float(row.get("EMA50"))
        ema200 = _to_float(row.get("EMA200"))
        rsi = _to_float(row.get("RSI14"))
        mom1 = _to_float(row.get("Momentum1M"))
        mom3 = _to_float(row.get("Momentum3M"))

        score = 50.0

        if close is not None and ema20 is not None:
            score += 10 if close > ema20 else -10

        if ema20 is not None and ema50 is not None:
            score += 8 if ema20 > ema50 else -8

        if ema50 is not None and ema200 is not None:
            score += 7 if ema50 > ema200 else -7

        if mom1 is not None:
            score += 8 if mom1 > 0 else -8
            if mom1 > 5:
                score += 4
            elif mom1 < -5:
                score -= 4

        if mom3 is not None:
            score += 10 if mom3 > 0 else -10
            if mom3 > 10:
                score += 4
            elif mom3 < -10:
                score -= 4

        if rsi is not None:
            if 48 <= rsi <= 68:
                score += 5
            elif 40 <= rsi < 48:
                score += 2
            elif 68 < rsi <= 75:
                score -= 2
            elif rsi > 75:
                score -= 6
            elif rsi < 28:
                score -= 3

        early_buy = False
        early_sell = False

        if close is not None and ema20 is not None and ema50 is not None and mom1 is not None:
            if close > ema20 and ema20 >= ema50 and mom1 > 0:
                early_buy = True
            if close < ema20 and ema20 <= ema50 and mom1 < 0:
                early_sell = True

        if early_buy:
            score += 4
        if early_sell:
            score -= 4

        score = round(_clamp(score), 1)
        signal = _signal_from_score(score)

        trend_state = "Neutral"
        if close is not None and ema20 is not None and ema50 is not None:
            if close > ema20 and ema20 > ema50:
                trend_state = "Positiv"
            elif close < ema20 and ema20 < ema50:
                trend_state = "Negativ"

        technical_scores.append(score)
        technical_signals.append(signal)
        trend_states.append(trend_state)

    work["Technical Score"] = technical_scores
    work["Technical Signal"] = technical_signals
    work["Trend State"] = trend_states

    keep_cols = [
        c
        for c in [
            "Date",
            "Close",
            "EMA20",
            "EMA50",
            "EMA200",
            "RSI14",
            "Momentum1M",
            "Momentum3M",
            "Technical Score",
            "Technical Signal",
            "Trend State",
        ]
        if c in work.columns
    ]
    return work[keep_cols].copy()


def build_unified_signal_snapshot(
    analysis: dict,
    diag: dict,
    news_bias: float | None = None,
) -> dict:
    timing = analysis.get("timing", {})
    macro = analysis.get("macro", {})
    record = analysis.get("record", {})
    price_df = analysis.get("df", pd.DataFrame())

    hist = build_technical_signal_history(price_df)
    latest_hist = hist.iloc[-1].to_dict() if not hist.empty else {}

    technical_score = _to_float(latest_hist.get("Technical Score"))
    technical_signal = str(latest_hist.get("Technical Signal", "HOLD"))
    trend_state = str(latest_hist.get("Trend State", timing.get("trend", "Neutral")))

    timing_score = _to_float(timing.get("timing_score"))
    quant_score = _to_float(analysis.get("quant_score"))
    rsi = _to_float(timing.get("rsi"))
    atr_pct = _to_float(timing.get("atr_pct"))
    momentum_1m = _to_float(timing.get("momentum_1m"))
    momentum_3m = _to_float(timing.get("momentum_3m"))
    regime = str(macro.get("regime", "") or "")
    rows = len(diag.get("df", pd.DataFrame()))
    has_theme = bool(str(record.get("themes", "")).strip())
    has_strategy_context = not analysis.get("strategy_context_df", pd.DataFrame()).empty
    has_theme_context = not analysis.get("theme_context_df", pd.DataFrame()).empty

    overall_score = 50.0

    if technical_score is not None:
        overall_score += (technical_score - 50) * 0.55

    if timing_score is not None:
        overall_score += (timing_score - 50) * 0.20

    if quant_score is not None:
        overall_score += (quant_score - 50) * 0.12

    if momentum_1m is not None:
        overall_score += 6 if momentum_1m > 0 else -6
    if momentum_3m is not None:
        overall_score += 8 if momentum_3m > 0 else -8

    if rsi is not None:
        if 45 <= rsi <= 65:
            overall_score += 3
        elif rsi > 75:
            overall_score -= 5
        elif rsi < 25:
            overall_score -= 2

    if atr_pct is not None:
        if atr_pct > 8:
            overall_score -= 4
        elif atr_pct < 4:
            overall_score += 2

    if regime.lower() == "risk-on":
        overall_score += 4
    elif regime.lower() == "risk-off":
        overall_score -= 4

    if has_theme:
        overall_score += 2
    if has_theme_context:
        overall_score += 2
    if has_strategy_context:
        overall_score += 2

    if rows >= 750:
        overall_score += 3
        data_bucket = "Høj"
    elif rows >= 250:
        overall_score += 1
        data_bucket = "Mellem"
    else:
        overall_score -= 4
        data_bucket = "Lav"

    applied_news_bias = 0.0
    if news_bias is not None:
        applied_news_bias = float(news_bias)
        overall_score += applied_news_bias

    overall_score = round(_clamp(overall_score), 1)
    overall_signal = _signal_from_score(overall_score)

    reasons = []
    reasons.append(f"teknisk signal er {technical_signal}")

    if technical_score is not None:
        reasons.append(f"teknisk score er {_score_bucket(technical_score).lower()} ({technical_score:.0f})")

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

    if applied_news_bias >= 4:
        reasons.append("nyhedsflowet er tydeligt positivt")
    elif applied_news_bias <= -4:
        reasons.append("nyhedsflowet er tydeligt negativt")
    elif abs(applied_news_bias) > 0:
        reasons.append("nyhedsflowet giver en mindre bias")

    return {
        "technical_signal": technical_signal,
        "technical_score": technical_score,
        "trend_state": trend_state,
        "overall_signal": overall_signal,
        "overall_score": overall_score,
        "timing_bucket": _score_bucket(timing_score),
        "quant_bucket": _score_bucket(quant_score),
        "technical_bucket": _score_bucket(technical_score),
        "news_bias": applied_news_bias,
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
        "data_bucket": data_bucket,
        "reasons": reasons[:7],
        "history_df": hist,
    }