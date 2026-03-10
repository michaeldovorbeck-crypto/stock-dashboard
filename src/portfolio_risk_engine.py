# src/portfolio_risk_engine.py

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.portfolio_snapshot_builder import build_portfolio_snapshot, split_themes


RISK_LABELS = [
    (0, 25, "Low"),
    (25, 50, "Moderate"),
    (50, 75, "Elevated"),
    (75, 101, "High"),
]


def _risk_label(score: float) -> str:
    for lower, upper, label in RISK_LABELS:
        if lower <= score < upper:
            return label
    return "Moderate"


def compute_position_concentration(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame([{
            "top_position_ticker": None,
            "top_position_weight_pct": 0.0,
            "top_3_weight_pct": 0.0,
            "single_position_alert": False,
            "top3_alert": False,
        }])

    df = snapshot_df.sort_values("weight", ascending=False).reset_index(drop=True)
    top_position_ticker = df.loc[0, "ticker"]
    top_position_weight = float(df.loc[0, "weight"]) * 100.0
    top_3_weight = float(df["weight"].head(3).sum()) * 100.0

    return pd.DataFrame([{
        "top_position_ticker": top_position_ticker,
        "top_position_weight_pct": round(top_position_weight, 2),
        "top_3_weight_pct": round(top_3_weight, 2),
        "single_position_alert": top_position_weight > 30.0,
        "top3_alert": top_3_weight > 60.0,
    }])


def compute_sector_exposure(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(columns=["sector", "weight_pct", "positions"])

    df = snapshot_df.copy()
    df["sector"] = df["sector"].fillna("Unknown").astype(str)

    out = (
        df.groupby("sector", dropna=False)
        .agg(weight=("weight", "sum"), positions=("ticker", "count"))
        .reset_index()
        .sort_values("weight", ascending=False)
    )
    out["weight_pct"] = (out["weight"] * 100.0).round(2)
    return out.drop(columns=["weight"])


def compute_theme_exposure(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(columns=["theme", "weight_pct", "positions"])

    rows = []
    for _, row in snapshot_df.iterrows():
        themes = split_themes(row.get("themes", "Unclassified"))
        split_weight = float(row.get("weight", 0.0)) / max(1, len(themes))
        for theme in themes:
            rows.append({
                "theme": theme,
                "ticker": row["ticker"],
                "weight": split_weight,
            })

    exploded = pd.DataFrame(rows)
    out = (
        exploded.groupby("theme", dropna=False)
        .agg(weight=("weight", "sum"), positions=("ticker", pd.Series.nunique))
        .reset_index()
        .sort_values("weight", ascending=False)
    )
    out["weight_pct"] = (out["weight"] * 100.0).round(2)
    return out.drop(columns=["weight"])


def compute_volatility_risk(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame([{
            "portfolio_volatility_proxy": 0.0,
            "volatility_bucket": "Unknown",
        }])

    vol_col = None
    for c in ["hist_volatility", "volatility", "atr_pct", "atr"]:
        if c in snapshot_df.columns:
            vol_col = c
            break

    if vol_col is None:
        return pd.DataFrame([{
            "portfolio_volatility_proxy": 0.0,
            "volatility_bucket": "Unknown",
        }])

    vol = pd.to_numeric(snapshot_df[vol_col], errors="coerce")
    if vol.notna().sum() == 0:
        portfolio_vol = 0.0
        bucket = "Unknown"
    else:
        portfolio_vol = float((vol.fillna(vol.median()) * snapshot_df["weight"]).sum())
        if portfolio_vol < 0.025:
            bucket = "Low"
        elif portfolio_vol < 0.05:
            bucket = "Medium"
        else:
            bucket = "High"

    return pd.DataFrame([{
        "portfolio_volatility_proxy": round(portfolio_vol, 4),
        "volatility_bucket": bucket,
    }])


def compute_diversification_metrics(
    snapshot_df: pd.DataFrame,
    sector_exposure_df: pd.DataFrame,
    theme_exposure_df: pd.DataFrame,
) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame([{
            "position_count": 0,
            "effective_n_positions": 0.0,
            "sector_count": 0,
            "theme_count": 0,
            "hhi_positions": 0.0,
        }])

    w = snapshot_df["weight"].fillna(0.0)
    hhi_positions = float((w ** 2).sum())
    effective_n = float(1.0 / hhi_positions) if hhi_positions > 0 else 0.0

    return pd.DataFrame([{
        "position_count": int(snapshot_df["ticker"].nunique()),
        "effective_n_positions": round(effective_n, 2),
        "sector_count": int(sector_exposure_df["sector"].nunique()) if not sector_exposure_df.empty else 0,
        "theme_count": int(theme_exposure_df["theme"].nunique()) if not theme_exposure_df.empty else 0,
        "hhi_positions": round(hhi_positions, 4),
    }])


def compute_risk_score(
    snapshot_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    sector_exposure_df: pd.DataFrame,
    theme_exposure_df: pd.DataFrame,
    volatility_df: pd.DataFrame,
) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame([{
            "risk_score": 0.0,
            "risk_label": "No Data",
            "macro_regime": "NEUTRAL",
            "macro_risk_modifier": 1.00,
        }])

    top1 = float(concentration_df.loc[0, "top_position_weight_pct"])
    top3 = float(concentration_df.loc[0, "top_3_weight_pct"])
    sector_top = float(sector_exposure_df["weight_pct"].max()) if not sector_exposure_df.empty else 0.0
    theme_top = float(theme_exposure_df["weight_pct"].max()) if not theme_exposure_df.empty else 0.0
    vol_bucket = str(volatility_df.loc[0, "volatility_bucket"])

    hhi = float((snapshot_df["weight"].fillna(0.0) ** 2).sum())
    effective_n = (1.0 / hhi) if hhi > 0 else 0.0

    top1_risk = min(100.0, (top1 / 30.0) * 100.0)
    top3_risk = min(100.0, (top3 / 60.0) * 100.0)
    sector_risk = min(100.0, (sector_top / 45.0) * 100.0)
    theme_risk = min(100.0, (theme_top / 35.0) * 100.0)

    vol_risk_map = {"LOW": 20.0, "MEDIUM": 55.0, "HIGH": 85.0, "UNKNOWN": 50.0}
    vol_risk = vol_risk_map.get(vol_bucket.upper(), 50.0)

    diversification_risk = 100.0 if effective_n <= 3 else max(
        0.0,
        min(100.0, 100.0 - ((effective_n - 3) / 12.0) * 100.0),
    )

    base_risk_score = (
        0.24 * top1_risk
        + 0.20 * top3_risk
        + 0.18 * sector_risk
        + 0.12 * theme_risk
        + 0.16 * vol_risk
        + 0.10 * diversification_risk
    )

    macro_regime = str(snapshot_df["macro_regime"].iloc[0]) if "macro_regime" in snapshot_df.columns else "NEUTRAL"
    macro_risk_modifier = float(snapshot_df["macro_risk_modifier"].iloc[0]) if "macro_risk_modifier" in snapshot_df.columns else 1.00

    adjusted_risk = min(100.0, max(0.0, base_risk_score * macro_risk_modifier))
    adjusted_risk = round(float(adjusted_risk), 1)

    return pd.DataFrame([{
        "risk_score": adjusted_risk,
        "risk_label": _risk_label(adjusted_risk),
        "macro_regime": macro_regime,
        "macro_risk_modifier": round(macro_risk_modifier, 2),
        "base_risk_score": round(float(base_risk_score), 1),
    }])


def generate_risk_alerts(
    concentration_df: pd.DataFrame,
    sector_exposure_df: pd.DataFrame,
    theme_exposure_df: pd.DataFrame,
    volatility_df: pd.DataFrame,
    risk_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    alerts = []

    if not concentration_df.empty:
        row = concentration_df.iloc[0]
        if bool(row["single_position_alert"]):
            alerts.append({
                "severity": "HIGH",
                "risk_type": "POSITION_CONCENTRATION",
                "message": f"Top position exceeds 30% ({row['top_position_ticker']} = {row['top_position_weight_pct']}%)",
            })
        if bool(row["top3_alert"]):
            alerts.append({
                "severity": "HIGH",
                "risk_type": "TOP3_CONCENTRATION",
                "message": f"Top 3 positions exceed 60% ({row['top_3_weight_pct']}%)",
            })

    if not sector_exposure_df.empty and float(sector_exposure_df["weight_pct"].max()) > 45:
        top_sector = sector_exposure_df.iloc[0]
        alerts.append({
            "severity": "MEDIUM",
            "risk_type": "SECTOR_CONCENTRATION",
            "message": f"Sector concentration high in {top_sector['sector']} ({top_sector['weight_pct']}%)",
        })

    if not theme_exposure_df.empty and float(theme_exposure_df["weight_pct"].max()) > 35:
        top_theme = theme_exposure_df.iloc[0]
        alerts.append({
            "severity": "MEDIUM",
            "risk_type": "THEME_CONCENTRATION",
            "message": f"Theme concentration high in {top_theme['theme']} ({top_theme['weight_pct']}%)",
        })

    if not volatility_df.empty and str(volatility_df.loc[0, "volatility_bucket"]).upper() == "HIGH":
        alerts.append({
            "severity": "MEDIUM",
            "risk_type": "VOLATILITY",
            "message": "Portfolio volatility bucket is HIGH",
        })

    if not risk_summary_df.empty:
        regime = str(risk_summary_df.loc[0, "macro_regime"]).upper()
        if regime in {"RISK_OFF", "RECESSION"}:
            alerts.append({
                "severity": "MEDIUM",
                "risk_type": "MACRO_REGIME",
                "message": f"Macro regime is {regime}, which increases portfolio risk",
            })

    if not alerts:
        alerts.append({
            "severity": "INFO",
            "risk_type": "NONE",
            "message": "No major portfolio risk alerts detected",
        })

    return pd.DataFrame(alerts)


def build_portfolio_risk(
    portfolio_df: pd.DataFrame,
    analysis_df: Optional[pd.DataFrame] = None,
    discovery_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    snapshot = build_portfolio_snapshot(
        portfolio_df=portfolio_df,
        analysis_df=analysis_df,
        discovery_df=discovery_df,
        macro_df=macro_df,
    )

    concentration = compute_position_concentration(snapshot)
    sector_exposure = compute_sector_exposure(snapshot)
    theme_exposure = compute_theme_exposure(snapshot)
    volatility = compute_volatility_risk(snapshot)
    diversification = compute_diversification_metrics(snapshot, sector_exposure, theme_exposure)
    risk_summary = compute_risk_score(snapshot, concentration, sector_exposure, theme_exposure, volatility)
    risk_alerts = generate_risk_alerts(concentration, sector_exposure, theme_exposure, volatility, risk_summary)

    return {
        "snapshot": snapshot,
        "concentration": concentration,
        "sector_exposure": sector_exposure,
        "theme_exposure": theme_exposure,
        "volatility": volatility,
        "diversification": diversification,
        "risk_summary": risk_summary,
        "risk_alerts": risk_alerts,
    }