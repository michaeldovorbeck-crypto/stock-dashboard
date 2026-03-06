# src/data_sources.py
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests


UNIVERSE_DIR = Path("data") / "universes"


def _safe_read_csv_text(text: str) -> pd.DataFrame:
    """Robust CSV read (stooq is simple CSV)."""
    text = (text or "").strip()
    if not text:
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        # fallback: try semicolon separator
        try:
            return pd.read_csv(StringIO(text), sep=";")
        except Exception:
            return pd.DataFrame()


def normalize_for_stooq(symbol: str) -> str:
    """
    Stooq bruger ofte små bogstaver og suffix for marked:
    - US: aapl.us
    - Mange EU: san.fr, sie.de, novo-b.co osv.
    Hvis brugeren allerede har '.', bruger vi den direkte (lowercase).
    Hvis ikke, antager vi US og tilføjer '.us'.
    """
    sym = (symbol or "").strip()
    if not sym:
        return ""
    sym = sym.replace(" ", "")
    if "." in sym:
        return sym.lower()
    # default: US
    return f"{sym.lower()}.us"


def fetch_daily_ohlcv_stooq(symbol: str, years: int = 5) -> pd.DataFrame:
    """
    Henter daglige OHLCV-data fra Stooq (gratis).
    Returnerer df med kolonner: Date, Open, High, Low, Close, Volume (hvis tilgængeligt)
    Returnerer tom df hvis fejl.
    """
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

    # Standardiser kolonner
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if years and years > 0 and not df.empty:
        cutoff = df["Date"].max() - pd.Timedelta(days=int(365.25 * years))
        df = df[df["Date"] >= cutoff].reset_index(drop=True)

    return df


def load_universe_csv(filename: str) -> Tuple[pd.DataFrame, str]:
    """
    Loader en universe-liste fra data/universes/<filename>.
    Forventede kolonner (mindst): ticker, name
    Ekstra (valgfrit): sector, country

    Returnerer: (df, status-tekst)
    """
    file_path = UNIVERSE_DIR / filename
    if not file_path.exists():
        return pd.DataFrame(), f"Fandt ikke filen: {file_path}"

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return pd.DataFrame(), f"Kunne ikke læse CSV: {file_path} ({e})"

    # Normaliser kolonnenavne
    df.columns = [c.strip().lower() for c in df.columns]

    # Tillad alternative kolonnenavne
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    if "name" not in df.columns and "company" in df.columns:
        df = df.rename(columns={"company": "name"})

    if "ticker" not in df.columns or "name" not in df.columns:
        return pd.DataFrame(), f"CSV mangler kolonner: ticker + name ({file_path})"

    # Rens
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip()
    else:
        df["sector"] = ""

    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()
    else:
        df["country"] = ""

    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df, f"Universe loaded: {file_path} ({len(df)} tickers)"