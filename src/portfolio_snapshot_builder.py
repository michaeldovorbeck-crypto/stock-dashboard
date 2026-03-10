# src/portfolio_snapshot_builder.py

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def safe_copy(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def standardize_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    ticker_col = find_column(out, ["ticker", "symbol", "asset", "instrument"])
    if ticker_col is None:
        raise ValueError("No ticker/symbol column found in dataframe.")

    if ticker_col != "ticker":
        out = out.rename(columns={ticker_col: "ticker"})

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out


def coalesce_columns(df: pd.DataFrame, candidates: List[str], default=np.nan) -> pd.Series:
    for col in candidates:
        real = find_column(df, [col])
        if real is not None:
            return df[real]
    return pd.Series([default] * len(df), index=df.index)


def to_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def normalize_to_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series([50.0] * len(series), index=series.index)

    if s.dropna().between(0, 100).mean() > 0.95:
        return s.fillna(s.median()).clip(0, 100)

    rank = s.rank(pct=True, method="average")
    return (rank * 100.0).fillna(50.0)


def trend_to_score(series: pd.Series) -> pd.Series:
    trend_score_map = {
        "STRONG UPTREND": 100.0,
        "UPTREND": 80.0,
        "NEUTRAL": 55.0,
        "DOWNTREND": 25.0,
        "STRONG DOWNTREND": 0.0,
    }
    s = series.astype(str).str.upper().str.strip()
    return s.map(trend_score_map).fillna(55.0)


def signal_to_score(series: pd.Series) -> pd.Series:
    signal_score_map = {
        "STRONG BUY": 100.0,
        "BUY": 85.0,
        "HOLD": 55.0,
        "SELL": 20.0,
        "STRONG SELL": 0.0,
    }
    s = series.astype(str).str.upper().str.strip()
    return s.map(signal_score_map).fillna(50.0)


def split_themes(value) -> List[str]:
    if pd.isna(value):
        return ["Unclassified"]

    text = str(value).strip()
    if not text:
        return ["Unclassified"]

    for sep in ["|", ";", ","]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()]

    return [text]


def build_portfolio_snapshot(
    portfolio_df: pd.DataFrame,
    analysis_df: Optional[pd.DataFrame] = None,
    signal_df: Optional[pd.DataFrame] = None,
    news_df: Optional[pd.DataFrame] = None,
    discovery_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    pf = standardize_ticker_column(portfolio_df)
    an = safe_copy(analysis_df)
    sg = safe_copy(signal_df)
    nw = safe_copy(news_df)
    dc = safe_copy(discovery_df)
    mc = safe_copy(macro_df)

    if not an.empty:
        an = standardize_ticker_column(an)
    if not sg.empty:
        sg = standardize_ticker_column(sg)
    if not nw.empty:
        nw = standardize_ticker_column(nw)
    if not dc.empty:
        dc = standardize_ticker_column(dc)

    out = pf.copy()

    value = coalesce_columns(out, ["market_value", "position_value", "value"])
    if value.isna().all() or (to_numeric(value, fill_value=0.0) == 0).all():
        qty = to_numeric(coalesce_columns(out, ["quantity", "qty", "shares"]), fill_value=0.0)
        price = to_numeric(coalesce_columns(out, ["current_price", "price", "last_close", "close"]), fill_value=0.0)
        out["market_value"] = qty * price
    else:
        out["market_value"] = to_numeric(value, fill_value=0.0)

    total_value = float(out["market_value"].sum())
    out["weight"] = np.where(total_value > 0, out["market_value"] / total_value, 0.0)

    if not an.empty:
        keep = [
            c for c in an.columns
            if c in {
                "ticker", "timing_score", "momentum_1m", "momentum_3m",
                "trend", "trend_score", "signal", "volatility",
                "hist_volatility", "atr_pct", "atr", "beta", "sector"
            }
        ]
        if keep:
            out = out.merge(an[keep].drop_duplicates(subset=["ticker"]), on="ticker", how="left")

    if not sg.empty:
        keep = [c for c in sg.columns if c in {"ticker", "signal", "signal_score", "signal_strength", "signal_streak"}]
        if keep:
            signal_overlay = sg[keep].drop_duplicates(subset=["ticker"]).copy()
            if "signal" in signal_overlay.columns and "signal" in out.columns:
                signal_overlay = signal_overlay.rename(columns={"signal": "signal_overlay"})
                out = out.merge(signal_overlay, on="ticker", how="left")
                out["signal"] = out["signal_overlay"].combine_first(out["signal"])
                out = out.drop(columns=["signal_overlay"])
            else:
                out = out.merge(signal_overlay, on="ticker", how="left")

    if not nw.empty:
        keep = [c for c in nw.columns if c in {"ticker", "news_sentiment", "sentiment", "sentiment_score"}]
        if keep:
            overlay = nw[keep].drop_duplicates(subset=["ticker"]).copy()
            if "sentiment" in overlay.columns:
                overlay = overlay.rename(columns={"sentiment": "news_sentiment"})
            if "sentiment_score" in overlay.columns:
                overlay = overlay.rename(columns={"sentiment_score": "news_sentiment"})
            out = out.merge(overlay, on="ticker", how="left")

    if not dc.empty:
        keep = [c for c in dc.columns if c in {"ticker", "themes", "theme", "discovery_score"}]
        if keep:
            overlay = dc[keep].drop_duplicates(subset=["ticker"]).copy()
            if "theme" in overlay.columns and "themes" not in overlay.columns:
                overlay["themes"] = overlay["theme"]
            out = out.merge(overlay.drop(columns=["theme"], errors="ignore"), on="ticker", how="left")

    if "signal" not in out.columns:
        out["signal"] = "HOLD"
    if "trend" not in out.columns:
        out["trend"] = "NEUTRAL"
    if "sector" not in out.columns:
        out["sector"] = "Unknown"
    if "themes" not in out.columns:
        out["themes"] = "Unclassified"
    if "discovery_score" not in out.columns:
        out["discovery_score"] = np.nan
    if "news_sentiment" not in out.columns:
        out["news_sentiment"] = np.nan

    # Macro regime bliver broadcastet til alle rækker
    out["macro_regime"] = "NEUTRAL"
    out["macro_risk_modifier"] = 1.00

    if not mc.empty:
        regime_col = find_column(mc, ["macro_regime", "regime"])
        modifier_col = find_column(mc, ["macro_risk_modifier", "risk_modifier"])
        if regime_col is not None:
            out["macro_regime"] = str(mc.iloc[0][regime_col]).upper().strip()
        if modifier_col is not None:
            try:
                out["macro_risk_modifier"] = float(mc.iloc[0][modifier_col])
            except Exception:
                out["macro_risk_modifier"] = 1.00
        else:
            regime = str(out["macro_regime"].iloc[0]).upper()
            regime_modifier_map = {
                "RISK_ON": 0.90,
                "NEUTRAL": 1.00,
                "RISK_OFF": 1.15,
                "HIGH_INFLATION": 1.10,
                "RECESSION": 1.20,
            }
            out["macro_risk_modifier"] = regime_modifier_map.get(regime, 1.00)

    return out