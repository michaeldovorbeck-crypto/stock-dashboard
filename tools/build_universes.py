import os
from typing import Dict, List

import pandas as pd
import requests

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data"
UNIVERSE_DIR = os.path.join(DATA_DIR, "universes")
HTTP_TIMEOUT = 25
TD_BASE = "https://api.twelvedata.com"

os.makedirs(UNIVERSE_DIR, exist_ok=True)

TD_EXCHANGES: Dict[str, List[str]] = {
    "USA": ["NASDAQ", "NYSE", "AMEX"],
    "Germany": ["XETRA", "FWB"],
    "Denmark": ["XCSE"],
    "Sweden": ["XSTO"],
    "United Kingdom": ["XLON"],
    "France": ["XPAR"],
    "Netherlands": ["XAMS"],
    "Switzerland": ["XSWX"],
    "Italy": ["XMIL"],
    "Spain": ["XMAD"],
    "Norway": ["XOSL"],
    "Finland": ["XHEL"],
    "Belgium": ["XBRU"],
    "Canada": ["XTSE", "XTSX"],
    "Japan": ["XTKS"],
    "Hong Kong": ["XHKG"],
    "India": ["XNSE", "XBOM"],
}

# =========================================================
# API KEY
# =========================================================
def get_api_key() -> str:
    # Først miljøvariabel, derefter evt. lokal secrets-fil hvis du vil udbygge senere
    return os.getenv("TWELVE_DATA_API_KEY", "").strip()


TD_API_KEY = get_api_key()


# =========================================================
# HELPERS
# =========================================================
def http_get(url: str, params: dict | None = None) -> requests.Response:
    return requests.get(url, params=params, timeout=HTTP_TIMEOUT)


def td_get(endpoint: str, params: dict | None = None) -> dict:
    if not TD_API_KEY:
        return {"status": "error", "message": "Missing TWELVE_DATA_API_KEY"}

    payload = dict(params or {})
    payload["apikey"] = TD_API_KEY

    try:
        r = http_get(f"{TD_BASE}/{endpoint}", params=payload)
        if r.status_code != 200:
            return {"status": "error", "message": f"HTTP {r.status_code}"}
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"]
        )

    df = normalize_cols(df)

    rename_map = {}
    for c in df.columns:
        if c in ("symbol", "ticker_code"):
            rename_map[c] = "ticker"
        elif c in ("instrument_name", "company", "companyname", "security", "name"):
            rename_map[c] = "name"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "ticker" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "ticker"})

    for col in ("name", "country", "exchange", "type", "source", "yahoo_symbol"):
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).fillna("").str.strip()
    df["country"] = df["country"].astype(str).fillna("").str.strip()
    df["exchange"] = df["exchange"].astype(str).fillna("").str.strip()
    df["type"] = df["type"].astype(str).fillna("").str.strip()
    df["source"] = df["source"].astype(str).fillna("").str.strip()
    df["yahoo_symbol"] = df["yahoo_symbol"].astype(str).fillna("").str.strip()

    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker", "exchange"])
    return df[["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"]].reset_index(drop=True)


def universe_file(key: str) -> str:
    return os.path.join(UNIVERSE_DIR, f"{key}.csv")


def make_yahoo_symbol(ticker: str, exchange: str) -> str:
    t = str(ticker).strip().upper()
    ex = str(exchange).strip().upper()

    suffix = {
        "XCSE": ".CO",
        "XSTO": ".ST",
        "XHEL": ".HE",
        "XOSL": ".OL",
        "XPAR": ".PA",
        "XAMS": ".AS",
        "XBRU": ".BR",
        "XLON": ".L",
        "XMIL": ".MI",
        "XMAD": ".MC",
        "XSWX": ".SW",
        "XTKS": ".T",
        "XHKG": ".HK",
        "XNSE": ".NS",
        "XBOM": ".BO",
        "XTSE": ".TO",
        "XTSX": ".V",
        "XETRA": ".DE",
        "FWB": ".DE",
        "NASDAQ": "",
        "NYSE": "",
        "AMEX": "",
    }

    if ex in suffix:
        return f"{t}{suffix[ex]}"
    return t


# =========================================================
# BUILDERS
# =========================================================
def fetch_stocks(exchange: str = "", country: str = "") -> pd.DataFrame:
    payload = {"format": "JSON"}
    if exchange:
        payload["exchange"] = exchange
    if country:
        payload["country"] = country

    data = td_get("stocks", payload)

    rows = []
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            rows = data["data"]
        elif isinstance(data.get("values"), list):
            rows = data["values"]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_country_universe(country_name: str, exchanges: List[str]) -> pd.DataFrame:
    frames = []

    for ex in exchanges:
        df = fetch_stocks(exchange=ex, country=country_name)
        if df.empty:
            print(f"[WARN] Ingen data for {country_name} / {ex}")
            continue

        df = normalize_cols(df)

        if "symbol" not in df.columns:
            print(f"[WARN] Ingen symbol-kolonne for {country_name} / {ex}")
            continue

        if "name" not in df.columns:
            for candidate in ["instrument_name", "company_name"]:
                if candidate in df.columns:
                    df["name"] = df[candidate]
                    break

        if "name" not in df.columns:
            df["name"] = ""

        if "country" not in df.columns:
            df["country"] = country_name
        if "exchange" not in df.columns:
            df["exchange"] = ex
        if "type" not in df.columns:
            df["type"] = ""

        df["source"] = "Twelve Data"
        df["yahoo_symbol"] = df.apply(
            lambda r: make_yahoo_symbol(r.get("symbol", ""), r.get("exchange", ex)),
            axis=1,
        )

        keep = ["symbol", "name", "country", "exchange", "type", "source", "yahoo_symbol"]
        df = df[keep].rename(columns={"symbol": "ticker"})
        frames.append(df)

        print(f"[OK] {country_name} / {ex}: {len(df)} symboler")

    if not frames:
        return pd.DataFrame(
            columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"]
        )

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "exchange"])
    return ensure_schema(out)


def build_all_universes() -> None:
    if not TD_API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY mangler som miljøvariabel.")

    all_frames = []
    counts = {}

    for country_name, exchanges in TD_EXCHANGES.items():
        df = build_country_universe(country_name, exchanges)

        key = country_name.lower().replace(" ", "_")
        path = universe_file(key)
        df.to_csv(path, index=False, encoding="utf-8")
        counts[key] = len(df)

        print(f"[SAVE] {path} ({len(df)} rækker)")

        if not df.empty:
            all_frames.append(df)

    if all_frames:
        global_df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["ticker", "exchange"])
    else:
        global_df = pd.DataFrame(
            columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"]
        )

    global_df = ensure_schema(global_df)
    global_df.to_csv(universe_file("global_all"), index=False, encoding="utf-8")
    print(f"[SAVE] {universe_file('global_all')} ({len(global_df)} rækker)")

    usa_df = pd.read_csv(universe_file("usa")) if os.path.exists(universe_file("usa")) else pd.DataFrame()
    if not usa_df.empty:
        usa_df = ensure_schema(usa_df)
        usa_df.to_csv(universe_file("us_all"), index=False, encoding="utf-8")
        print(f"[SAVE] {universe_file('us_all')} ({len(usa_df)} rækker)")

    print("\n=== SUMMARY ===")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"global_all: {len(global_df)}")


if __name__ == "__main__":
    build_all_universes()