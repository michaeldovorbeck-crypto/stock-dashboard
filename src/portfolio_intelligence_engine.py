# src/portfolio_intelligence_engine.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.portfolio_snapshot_builder import (
    build_portfolio_snapshot,
    coalesce_columns,
    normalize_to_100,
    signal_to_score,
    trend_to_score,
)


HEALTH_LABELS = [
    (0, 40, "Weak"),
    (40, 60, "Neutral"),
    (60, 80, "Strong"),
    (80, 101, "Very Strong"),
]

DEFAULT_HEALTH_WEIGHTS = {
    "timing_score": 0.24,
    "momentum_score": 0.18,
    "trend_score": 0.18,
    "signal_score": 0.18,
    "volatility_score": 0.12,
    "news_score": 0.10,
}


def _health_label(score: float) -> str:
    for lower, upper, label in HEALTH_LABELS:
        if lower <= score < upper:
            return label
    return "Neutral"


def compute_portfolio_health_score(
    snapshot_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame([{
            "portfolio_health_score": 0.0,
            "health_status": "No Data",
            "positions": 0,
        }])

    w = weights or DEFAULT_HEALTH_WEIGHTS
    df = snapshot_df.copy()

    timing_score = normalize_to_100(coalesce_columns(df, ["timing_score"]))

    m1 = pd.to_numeric(coalesce_columns(df, ["momentum_1m"]), errors="coerce")
    m3 = pd.to_numeric(coalesce_columns(df, ["momentum_3m"]), errors="coerce")
    momentum_raw = np.where(
        ~pd.isna(m3),
        0.4 * pd.Series(m1, index=df.index).fillna(0) + 0.6 * pd.Series(m3, index=df.index).fillna(0),
        pd.Series(m1, index=df.index).fillna(0),
    )
    momentum_score = normalize_to_100(pd.Series(momentum_raw, index=df.index))

    trend_raw = coalesce_columns(df, ["trend_score"])
    trend_raw = np.where(
        pd.to_numeric(trend_raw, errors="coerce").notna(),
        trend_raw,
        trend_to_score(df["trend"]),
    )
    trend_score = normalize_to_100(pd.Series(trend_raw, index=df.index))

    signal_score = signal_to_score(coalesce_columns(df, ["signal"], default="HOLD"))

    vol_proxy = coalesce_columns(df, ["hist_volatility", "volatility", "atr_pct", "atr"])
    vol_raw = pd.to_numeric(vol_proxy, errors="coerce")
    if vol_raw.notna().sum() == 0:
        volatility_score = pd.Series([55.0] * len(df), index=df.index)
    else:
        volatility_score = 100.0 - normalize_to_100(vol_raw)

    news_raw = pd.to_numeric(coalesce_columns(df, ["news_sentiment"]), errors="coerce")
    if news_raw.notna().sum() == 0:
        news_score = pd.Series([55.0] * len(df), index=df.index)
    else:
        if news_raw.dropna().between(-1, 1).mean() > 0.9:
            news_score = ((news_raw.fillna(0) + 1.0) / 2.0) * 100.0
        elif news_raw.dropna().between(0, 1).mean() > 0.9:
            news_score = news_raw.fillna(0.5) * 100.0
        else:
            news_score = normalize_to_100(news_raw)

    df["health_component"] = (
        timing_score * w["timing_score"]
        + momentum_score * w["momentum_score"]
        + trend_score * w["trend_score"]
        + signal_score * w["signal_score"]
        + volatility_score * w["volatility_score"]
        + news_score * w["news_score"]
    )

    weighted_score = float((df["health_component"] * df["weight"]).sum())
    weighted_score = round(weighted_score, 1)

    return pd.DataFrame([{
        "portfolio_health_score": weighted_score,
        "health_status": _health_label(weighted_score),
        "positions": int(len(df)),
        "buy_positions": int((df["signal"].astype(str).str.upper() == "BUY").sum()),
        "sell_positions": int((df["signal"].astype(str).str.upper() == "SELL").sum()),
        "macro_regime": str(df["macro_regime"].iloc[0]) if "macro_regime" in df.columns and len(df) else "NEUTRAL",
    }])


def compute_signal_distribution(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(columns=["signal", "count", "weight"])

    df = snapshot_df.copy()
    df["signal"] = df["signal"].astype(str).str.upper().str.strip()

    out = (
        df.groupby("signal", dropna=False)
        .agg(count=("ticker", "count"), weight=("weight", "sum"))
        .reset_index()
        .sort_values(["count", "weight"], ascending=[False, False])
    )
    out["weight_pct"] = (out["weight"] * 100.0).round(2)
    return out.drop(columns=["weight"])


def compute_signal_drift(
    signal_history_df: Optional[pd.DataFrame],
    current_snapshot_df: Optional[pd.DataFrame] = None,
    max_states: int = 5,
) -> pd.DataFrame:
    if signal_history_df is None or signal_history_df.empty:
        return pd.DataFrame(columns=["ticker", "signal_path", "current_signal", "previous_signal", "drift_flag"])

    hist = signal_history_df.copy()
    if "ticker" not in hist.columns or "signal" not in hist.columns:
        return pd.DataFrame(columns=["ticker", "signal_path", "current_signal", "previous_signal", "drift_flag"])

    date_col = "date" if "date" in hist.columns else ("timestamp" if "timestamp" in hist.columns else None)
    if date_col is None:
        return pd.DataFrame(columns=["ticker", "signal_path", "current_signal", "previous_signal", "drift_flag"])

    hist["ticker"] = hist["ticker"].astype(str).str.upper().str.strip()
    hist["signal"] = hist["signal"].astype(str).str.upper().str.strip()
    hist[date_col] = pd.to_datetime(hist[date_col], errors="coerce")
    hist = hist.sort_values(["ticker", date_col])

    rows = []
    for ticker, grp in hist.groupby("ticker"):
        states = grp["signal"].dropna().tolist()
        if not states:
            continue
        recent_states = states[-max_states:]
        current_signal = recent_states[-1]
        previous_signal = recent_states[-2] if len(recent_states) >= 2 else None
        drift_flag = previous_signal is not None and current_signal != previous_signal

        rows.append({
            "ticker": ticker,
            "signal_path": " → ".join(recent_states),
            "current_signal": current_signal,
            "previous_signal": previous_signal,
            "drift_flag": drift_flag,
        })

    out = pd.DataFrame(rows)

    if current_snapshot_df is not None and not current_snapshot_df.empty and not out.empty:
        cur = current_snapshot_df[["ticker", "signal"]].copy()
        cur["signal"] = cur["signal"].astype(str).str.upper().str.strip()
        out = out.drop(columns=["current_signal"], errors="ignore").merge(cur, on="ticker", how="left")
        out = out.rename(columns={"signal": "current_signal"})

    return out.sort_values(["drift_flag", "ticker"], ascending=[False, True]).reset_index(drop=True)


def generate_portfolio_alerts(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(columns=["ticker", "severity", "alert_type", "message"])

    df = snapshot_df.copy()
    timing = normalize_to_100(coalesce_columns(df, ["timing_score"]))
    trend_score = normalize_to_100(trend_to_score(df["trend"]))
    signal = coalesce_columns(df, ["signal"], default="HOLD").astype(str).str.upper().str.strip()
    news = pd.to_numeric(coalesce_columns(df, ["news_sentiment"]), errors="coerce").fillna(0)

    alerts = []

    for i, row in df.iterrows():
        ticker = row["ticker"]
        weight = float(row.get("weight", 0.0))

        if signal.iloc[i] in {"SELL", "STRONG SELL"}:
            alerts.append({
                "ticker": ticker,
                "severity": "HIGH",
                "alert_type": "SELL_SIGNAL",
                "message": f"{ticker} → {signal.iloc[i]} signal",
            })

        if signal.iloc[i] in {"BUY", "STRONG BUY"} and timing.iloc[i] >= 75 and trend_score.iloc[i] >= 70:
            alerts.append({
                "ticker": ticker,
                "severity": "MEDIUM",
                "alert_type": "STRONG_BUY",
                "message": f"{ticker} → Strong buy setup",
            })

        if weight >= 0.20 and signal.iloc[i] in {"HOLD", "SELL", "STRONG SELL"}:
            alerts.append({
                "ticker": ticker,
                "severity": "MEDIUM",
                "alert_type": "OVERSIZED_NONBUY",
                "message": f"{ticker} → Large position with weak signal ({weight:.1%})",
            })

        if news.iloc[i] < -0.35:
            alerts.append({
                "ticker": ticker,
                "severity": "LOW",
                "alert_type": "NEGATIVE_NEWS",
                "message": f"{ticker} → Negative news sentiment detected",
            })

    if not alerts:
        return pd.DataFrame([{
            "ticker": "-",
            "severity": "INFO",
            "alert_type": "NONE",
            "message": "No critical portfolio alerts detected",
        }])

    out = pd.DataFrame(alerts)
    sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    out["sev_rank"] = out["severity"].map(sev_rank).fillna(9)
    return out.sort_values(["sev_rank", "ticker"]).drop(columns=["sev_rank"]).reset_index(drop=True)


def generate_rebalance_suggestions(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(columns=["ticker", "action", "priority", "rationale"])

    df = snapshot_df.copy()

    signal = coalesce_columns(df, ["signal"], default="HOLD").astype(str).str.upper().str.strip()
    timing = normalize_to_100(coalesce_columns(df, ["timing_score"]))
    trend_score = normalize_to_100(trend_to_score(df["trend"]))
    weight = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    disc = pd.to_numeric(coalesce_columns(df, ["discovery_score"]), errors="coerce")
    if disc.notna().sum() == 0:
        discovery_score = pd.Series([50.0] * len(df), index=df.index)
    else:
        discovery_score = normalize_to_100(disc)

    rows = []

    for i, row in df.iterrows():
        ticker = row["ticker"]
        sig = signal.iloc[i]
        ts = float(timing.iloc[i])
        tr = float(trend_score.iloc[i])
        wt = float(weight.iloc[i])
        ds = float(discovery_score.iloc[i])

        if sig in {"SELL", "STRONG SELL"}:
            action = "REDUCE"
            priority = "HIGH"
            rationale = f"{sig} signal with weak technical profile"
        elif sig == "BUY" and ts >= 70 and tr >= 70 and ds >= 65 and wt < 0.10:
            action = "INCREASE"
            priority = "HIGH"
            rationale = "Strong technicals plus strong theme/discovery support"
        elif sig == "BUY" and ts >= 65 and tr >= 65 and wt < 0.12:
            action = "INCREASE"
            priority = "MEDIUM"
            rationale = "Good technical quality with room to scale"
        elif sig == "HOLD" and wt > 0.18 and (ts < 55 or tr < 55):
            action = "TRIM"
            priority = "MEDIUM"
            rationale = "Large position without strong confirmation"
        else:
            action = "HOLD"
            priority = "LOW"
            rationale = "No rebalance needed"

        rows.append({
            "ticker": ticker,
            "action": action,
            "priority": priority,
            "current_weight_pct": round(wt * 100.0, 2),
            "timing_score": round(ts, 1),
            "trend_score": round(tr, 1),
            "discovery_score": round(ds, 1),
            "signal": sig,
            "rationale": rationale,
        })

    out = pd.DataFrame(rows)
    prio_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    action_rank = {"REDUCE": 0, "TRIM": 1, "INCREASE": 2, "HOLD": 3}
    out["prio_rank"] = out["priority"].map(prio_rank).fillna(9)
    out["action_rank"] = out["action"].map(action_rank).fillna(9)

    return (
        out.sort_values(["prio_rank", "action_rank", "current_weight_pct"], ascending=[True, True, False])
        .drop(columns=["prio_rank", "action_rank"])
        .reset_index(drop=True)
    )


def build_portfolio_intelligence(
    portfolio_df: pd.DataFrame,
    analysis_df: Optional[pd.DataFrame] = None,
    signal_df: Optional[pd.DataFrame] = None,
    signal_history_df: Optional[pd.DataFrame] = None,
    news_df: Optional[pd.DataFrame] = None,
    discovery_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    snapshot = build_portfolio_snapshot(
        portfolio_df=portfolio_df,
        analysis_df=analysis_df,
        signal_df=signal_df,
        news_df=news_df,
        discovery_df=discovery_df,
        macro_df=macro_df,
    )

    return {
        "snapshot": snapshot,
        "health_summary": compute_portfolio_health_score(snapshot),
        "signal_distribution": compute_signal_distribution(snapshot),
        "signal_drift": compute_signal_drift(signal_history_df, current_snapshot_df=snapshot),
        "alerts": generate_portfolio_alerts(snapshot),
        "rebalance": generate_rebalance_suggestions(snapshot),
    }