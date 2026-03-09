# src/data_sources.py
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
import streamlit as st


UNIVERSE_DIR = Path("data") / "universes"
TD_BASE = "https://api.twelvedata.com"


def get_secret(name: str) -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return os.getenv(name, "").strip()


TWELVE_DATA_API_KEY = get_secret("TWELVE_DATA_API_KEY")


def _safe_read_csv_text(text: str) -> pd.DataFrame:
    text = (text or "").strip()
    if not text:
        return pd.DataFrame()

    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        try:
            return pd.read_csv(StringIO(text), sep=";")
        except Exception:
            return pd.DataFrame()


def normalize_for_stooq(symbol: str) -> str:
    """
    Stooq bruger typisk:
    - aapl.us
    - msft.us
    - novo-b.co
    - sie.de
    """
    sym = (symbol or "").strip()
    if not sym:
        return ""

    sym = sym.replace(" ", "")

    if "." in sym:
        return sym.lower()

    return f"{sym.lower()}.us"


def fetch_daily_ohlcv_twelve(symbol: str, years: int = 5) -> pd.DataFrame:
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()

    outputsize = min(5000, max(300, int(years) * 260))

    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": outputsize,
        "format": "JSON",
        "order": "ASC",
        "apikey": TWELVE_DATA_API_KEY,
    }

    try:
        r = requests.get(f"{TD_BASE}/time_series", params=params, timeout=25)
        if r.status_code != 200:
            return pd.DataFrame()

        js = r.json()
        values = js.get("values", [])
        if not values:
            return pd.DataFrame()

        df = pd.DataFrame(values)
        if df.empty or "datetime" not in df.columns:
            return pd.DataFrame()

        df = df.rename(
            columns={
                "datetime": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep_cols].copy()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

        if years and years > 0 and not df.empty:
            cutoff = df["Date"].max() - pd.Timedelta(days=int(365.25 * years))
            df = df[df["Date"] >= cutoff].reset_index(drop=True)

        return df

    except Exception:
        return pd.DataFrame()


def fetch_daily_ohlcv_stooq(symbol: str, years: int = 5) -> pd.DataFrame:
    sym = normalize_for_stooq(symbol)
    if not sym:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    df = _safe_read_csv_text(r.text)
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if years and years > 0 and not df.empty:
        cutoff = df["Date"].max() - pd.Timedelta(days=int(365.25 * years))
        df = df[df["Date"] >= cutoff].reset_index(drop=True)

    return df


def load_universe_csv(filename: str) -> Tuple[pd.DataFrame, str]:
    file_path = UNIVERSE_DIR / filename

    if not file_path.exists():
        return pd.DataFrame(), f"Fandt ikke filen: {file_path}"

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return pd.DataFrame(), f"Kunne ikke læse CSV: {file_path} ({e})"

    df.columns = [c.strip().lower() for c in df.columns]

    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})

    if "name" not in df.columns and "company" in df.columns:
        df = df.rename(columns={"company": "name"})

    if "ticker" not in df.columns or "name" not in df.columns:
        return pd.DataFrame(), f"CSV mangler kolonner ticker + name ({file_path})"

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    if "sector" not in df.columns:
        df["sector"] = ""

    if "country" not in df.columns:
        df["country"] = ""

    df["sector"] = df["sector"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()

    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    return df, f"Universe loaded: {file_path} ({len(df)} tickers)"


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Standard V3 wrapper med cache:
    1) Twelve Data som primær
    2) Stooq som fallback

    Returnerer standardformat:
    Date, Open, High, Low, Close, Volume
    """
    ticker = (ticker or "").strip()
    if not ticker:
        return pd.DataFrame()

    df = fetch_daily_ohlcv_twelve(ticker, years=years)
    if not df.empty:
        return df

    return fetch_daily_ohlcv_stooq(ticker, years=years)