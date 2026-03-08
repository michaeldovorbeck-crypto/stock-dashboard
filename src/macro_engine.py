# src/macro_engine.py
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def get_secret(name: str) -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return os.getenv(name, "").strip()


FRED_API_KEY = get_secret("FRED_API_KEY")


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    if not FRED_API_KEY:
        return pd.DataFrame()

    params = {
        "series_id": series_id,
        "file_type": "json",
        "api_key": FRED_API_KEY,
    }

    try:
        r = requests.get(FRED_BASE, params=params, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()

        js = r.json()
        obs = js.get("observations", [])
        if not obs:
            return pd.DataFrame()

        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        return df[["date", "value"]]

    except Exception:
        return pd.DataFrame()


def latest_value(series_id: str):
    df = fetch_fred_series(series_id)
    if df.empty:
        return None

    clean = df.dropna(subset=["value"])
    if clean.empty:
        return None

    return float(clean["value"].iloc[-1])


def yoy_change(series_id: str):
    df = fetch_fred_series(series_id)
    if df.empty:
        return None

    clean = df.dropna(subset=["value"]).copy()
    if len(clean) < 13:
        return None

    latest = float(clean["value"].iloc[-1])
    prev = float(clean["value"].iloc[-13])

    if prev == 0:
        return None

    return (latest / prev - 1.0) * 100.0


def macro_snapshot() -> dict:
    inflation_yoy = yoy_change("CPIAUCSL")
    industrial_yoy = yoy_change("INDPRO")
    oil_yoy = yoy_change("DCOILWTICO")

    us_10y = latest_value("DGS10")
    us_2y = latest_value("DGS2")
    unemployment = latest_value("UNRATE")

    regime = "Neutral"

    if inflation_yoy is not None and industrial_yoy is not None:
        if industrial_yoy > 0 and inflation_yoy < 4:
            regime = "Risk-on"
        elif industrial_yoy < 0 and inflation_yoy > 4:
            regime = "Risk-off"

    rate_curve = None
    if us_10y is not None and us_2y is not None:
        rate_curve = us_10y - us_2y

    return {
        "fred_key_loaded": bool(FRED_API_KEY),
        "regime": regime,
        "inflation_yoy_pct": round(inflation_yoy, 2) if inflation_yoy is not None else None,
        "industrial_production_yoy_pct": round(industrial_yoy, 2) if industrial_yoy is not None else None,
        "oil_yoy_pct": round(oil_yoy, 2) if oil_yoy is not None else None,
        "us_10y": round(us_10y, 2) if us_10y is not None else None,
        "us_2y": round(us_2y, 2) if us_2y is not None else None,
        "rate_curve_10y_minus_2y": round(rate_curve, 2) if rate_curve is not None else None,
        "unemployment": round(unemployment, 2) if unemployment is not None else None,
    }