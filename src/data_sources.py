from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import requests
import streamlit as st

from src.resolver_engine import get_alternative_tickers, normalize_symbol
from src.yahoo_source import fetch_yahoo_ohlcv


UNIVERSE_DIR = Path("data") / "universes"
TD_BASE = "https://api.twelvedata.com"

TD_SUFFIX_MAP = {
    ".CO": ":XCSE",
    ".ST": ":XSTO",
    ".OL": ":XOSL",
    ".HE": ":XHEL",
    ".DE": ":XETRA",
    ".PA": ":XPAR",
    ".AS": ":XAMS",
    ".SW": ":XSWX",
    ".MI": ":XMIL",
    ".MC": ":XMAD",
    ".L": ":XLON",
    ".TO": ":XTSE",
    ".V": ":XTSX",
    ".HK": ":XHKG",
    ".NS": ":XNSE",
    ".BO": ":XBOM",
    ".T": ":XTKS",
    ".BR": ":XBRU",
}

STOOQ_SUFFIX_MAP = {
    ".CO": ".co",
    ".ST": ".se",
    ".OL": ".no",
    ".HE": ".fi",
    ".DE": ".de",
    ".PA": ".fr",
    ".AS": ".nl",
    ".SW": ".ch",
    ".MI": ".it",
    ".MC": ".es",
    ".L": ".uk",
    ".TO": ".ca",
    ".V": ".ca",
    ".HK": ".hk",
    ".NS": ".in",
    ".BO": ".in",
    ".T": ".jp",
    ".BR": ".be",
}

STOOQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://stooq.com/",
    "Connection": "keep-alive",
}


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


def twelve_symbol_candidates(symbol: str) -> list[str]:
    sym = normalize_symbol(symbol)
    if not sym:
        return []

    candidates: list[str] = []

    for alt in get_alternative_tickers(sym):
        candidates.append(alt)

        if "." not in alt and ":" not in alt:
            candidates.append(f"{alt}:US")

        for suffix, mapped in TD_SUFFIX_MAP.items():
            if alt.endswith(suffix):
                base = alt[: -len(suffix)]
                candidates.append(f"{base}{mapped}")
                candidates.append(base)
                break

        if "." in alt:
            base = alt.split(".", 1)[0]
            candidates.append(base)
            candidates.append(f"{base}:US")

    seen = set()
    out = []
    for c in candidates:
        c = normalize_symbol(c)
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def yahoo_symbol_candidates(symbol: str) -> list[str]:
    sym = normalize_symbol(symbol)
    if not sym:
        return []

    candidates: list[str] = []
    for alt in get_alternative_tickers(sym):
        if ":" in alt:
            candidates.append(alt.split(":", 1)[0])
        else:
            candidates.append(alt)

    seen = set()
    out = []
    for c in candidates:
        c = normalize_symbol(c)
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def stooq_symbol_candidates(symbol: str) -> list[str]:
    sym = normalize_symbol(symbol)
    if not sym:
        return []

    candidates: list[str] = []

    for alt in get_alternative_tickers(sym):
        for suffix, mapped in STOOQ_SUFFIX_MAP.items():
            if alt.endswith(suffix):
                base = alt[: -len(suffix)]
                candidates.append(f"{base.lower()}{mapped}")
                candidates.append(base.lower())
                break
        else:
            if "." in alt:
                candidates.append(alt.lower())
                candidates.append(alt.split(".", 1)[0].lower())
            elif ":" in alt:
                base = alt.split(":", 1)[0]
                candidates.append(base.lower())
                candidates.append(f"{base.lower()}.us")
            else:
                candidates.append(f"{alt.lower()}.us")
                candidates.append(alt.lower())

    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rename_map = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl == "date":
            rename_map[col] = "Date"
        elif cl == "open":
            rename_map[col] = "Open"
        elif cl == "high":
            rename_map[col] = "High"
        elif cl == "low":
            rename_map[col] = "Low"
        elif cl == "close":
            rename_map[col] = "Close"
        elif cl == "volume":
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols].copy()

    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


def _trim_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty or years <= 0:
        return df.reset_index(drop=True)

    cutoff = df["Date"].max() - pd.Timedelta(days=int(365.25 * years))
    return df[df["Date"] >= cutoff].reset_index(drop=True)


def _fetch_from_twelve(candidate: str, outputsize: int, years: int) -> tuple[pd.DataFrame, str]:
    params = {
        "symbol": candidate,
        "interval": "1day",
        "outputsize": outputsize,
        "format": "JSON",
        "order": "ASC",
        "apikey": TWELVE_DATA_API_KEY,
    }

    r = requests.get(f"{TD_BASE}/time_series", params=params, timeout=25)
    if r.status_code != 200:
        return pd.DataFrame(), f"HTTP {r.status_code}"

    js = r.json()
    values = js.get("values", [])
    if not values:
        return pd.DataFrame(), str(js.get("message") or js.get("status") or "No values")

    df = pd.DataFrame(values).rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df = _standardize_ohlcv(df)
    df = _trim_years(df, years)

    if df.empty:
        return pd.DataFrame(), "Empty dataframe"

    return df, "Success"


def _fetch_from_stooq(candidate: str, years: int) -> tuple[pd.DataFrame, str]:
    url = f"https://stooq.com/q/d/l/?s={candidate}&i=d"

    r = requests.get(
        url,
        headers=STOOQ_HEADERS,
        timeout=20,
    )

    if r.status_code != 200:
        return pd.DataFrame(), f"HTTP {r.status_code}"

    df = _safe_read_csv_text(r.text)

    if df.empty:
        preview = (r.text or "")[:180].replace("\n", " ")
        return pd.DataFrame(), f"No CSV data | preview={preview}"

    if "Date" not in df.columns:
        preview = (r.text or "")[:180].replace("\n", " ")
        return pd.DataFrame(), f"Missing Date column | preview={preview}"

    df = _standardize_ohlcv(df)
    df = _trim_years(df, years)

    if df.empty:
        return pd.DataFrame(), "Empty dataframe"

    return df, "Success"


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_history_with_meta(ticker: str, years: int = 5) -> dict[str, Any]:
    ticker = normalize_symbol(ticker)
    if not ticker:
        return {"df": pd.DataFrame(), "source": "", "used_symbol": "", "attempts": [], "alternatives": []}

    outputsize = min(5000, max(300, int(years) * 260))
    attempts: list[dict[str, str]] = []
    alternatives = get_alternative_tickers(ticker)

    if TWELVE_DATA_API_KEY:
        for candidate in twelve_symbol_candidates(ticker):
            try:
                df, status = _fetch_from_twelve(candidate, outputsize, years)
                attempts.append({"source": "Twelve Data", "symbol": candidate, "status": status})
                if not df.empty:
                    return {
                        "df": df,
                        "source": "Twelve Data",
                        "used_symbol": candidate,
                        "attempts": attempts,
                        "alternatives": alternatives,
                    }
            except Exception as e:
                attempts.append({"source": "Twelve Data", "symbol": candidate, "status": f"Error: {e}"})
    else:
        attempts.append({"source": "Twelve Data", "symbol": "", "status": "Missing API key"})

    for candidate in yahoo_symbol_candidates(ticker):
        try:
            df, status = fetch_yahoo_ohlcv(candidate, years=years)
            attempts.append({"source": "Yahoo", "symbol": candidate, "status": status})
            if not df.empty:
                return {
                    "df": df,
                    "source": "Yahoo",
                    "used_symbol": candidate,
                    "attempts": attempts,
                    "alternatives": alternatives,
                }
        except Exception as e:
            attempts.append({"source": "Yahoo", "symbol": candidate, "status": f"Error: {e}"})

    for candidate in stooq_symbol_candidates(ticker):
        try:
            df, status = _fetch_from_stooq(candidate, years)
            attempts.append({"source": "Stooq", "symbol": candidate, "status": status})
            if not df.empty:
                return {
                    "df": df,
                    "source": "Stooq",
                    "used_symbol": candidate,
                    "attempts": attempts,
                    "alternatives": alternatives,
                }
        except Exception as e:
            attempts.append({"source": "Stooq", "symbol": candidate, "status": f"Error: {e}"})

    return {
        "df": pd.DataFrame(),
        "source": "",
        "used_symbol": "",
        "attempts": attempts,
        "alternatives": alternatives,
    }


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_history(ticker: str, years: int = 5) -> pd.DataFrame:
    return fetch_history_with_meta(ticker, years=years).get("df", pd.DataFrame())


def fetch_daily_ohlcv_twelve(symbol: str, years: int = 5) -> pd.DataFrame:
    meta = fetch_history_with_meta(symbol, years=years)
    return meta.get("df", pd.DataFrame()) if meta.get("source") == "Twelve Data" else pd.DataFrame()


def fetch_daily_ohlcv_yahoo(symbol: str, years: int = 5) -> pd.DataFrame:
    meta = fetch_history_with_meta(symbol, years=years)
    return meta.get("df", pd.DataFrame()) if meta.get("source") == "Yahoo" else pd.DataFrame()


def fetch_daily_ohlcv_stooq(symbol: str, years: int = 5) -> pd.DataFrame:
    meta = fetch_history_with_meta(symbol, years=years)
    return meta.get("df", pd.DataFrame()) if meta.get("source") == "Stooq" else pd.DataFrame()


def get_data_diagnostics(ticker: str, years: int = 5) -> pd.DataFrame:
    meta = fetch_history_with_meta(ticker, years=years)
    attempts = meta.get("attempts", [])
    if not attempts:
        return pd.DataFrame(columns=["source", "symbol", "status"])
    return pd.DataFrame(attempts)


def get_data_source_info(ticker: str, years: int = 5) -> dict[str, Any]:
    meta = fetch_history_with_meta(ticker, years=years)
    return {
        "source": meta.get("source", ""),
        "used_symbol": meta.get("used_symbol", ""),
        "alternatives": meta.get("alternatives", []),
        "rows": len(meta.get("df", pd.DataFrame())),
    }


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