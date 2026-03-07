import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Stock Dashboard (Global screener + teknisk analyse)",
    layout="wide",
    page_icon="📊",
)

APP_TITLE = "📊 Stock Dashboard (Global screener + teknisk analyse)"
DATA_DIR = "data"
UNIVERSE_DIR = os.path.join(DATA_DIR, "universes")
SIGNAL_LOG = os.path.join(DATA_DIR, "signals_log.csv")
DEFAULT_TOPN = 15
HTTP_TIMEOUT = 25
TD_BASE = "https://api.twelvedata.com"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

os.makedirs(DATA_DIR, exist_ok=True)
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

THEMES: Dict[str, dict] = {
    "AI & Software": {
        "proxy": "QQQ",
        "benchmark": "SPY",
        "members": ["QQQ", "XLK", "MSFT", "NVDA", "AVGO", "GOOGL", "AMZN"],
        "drivers": [
            "Datacenter capex",
            "GPU-efterspørgsel",
            "Enterprise software spending",
            "Cloud-vækst",
            "Produktivitetsinvesteringer",
        ],
        "headwinds": [
            "Høje multipler",
            "Capex-normalisering",
            "Regulatoriske risici",
            "Langsommere AI-monetisering",
        ],
        "macro_series": {
            "10Y rente": "DGS10",
            "Inflation": "CPIAUCSL",
            "Industriproduktion": "INDPRO",
        },
        "description": "Temaet dækker software, hyperscalers, AI-infrastruktur og halvlederefterspørgsel drevet af datacenterudbygning og enterprise adoption.",
    },
    "Semiconductors": {
        "proxy": "SOXX",
        "benchmark": "SPY",
        "members": ["SOXX", "SMH", "NVDA", "AVGO", "AMD", "TSM", "ASML"],
        "drivers": [
            "AI-servere",
            "Foundry-kapacitet",
            "Memory-cyklus",
            "Industrikapacitet",
            "Elektrificering",
        ],
        "headwinds": [
            "Cyklicitet",
            "Geopolitik",
            "Overkapacitet",
            "Lagerkorrektioner",
        ],
        "macro_series": {
            "Industriproduktion": "INDPRO",
            "10Y rente": "DGS10",
            "USD indeks": "DTWEXBGS",
        },
        "description": "Halvledertemaet drives af AI, datacenterkapacitet, foundry-økonomi og global industriefterspørgsel.",
    },
    "Defense/Aerospace": {
        "proxy": "ITA",
        "benchmark": "SPY",
        "members": ["ITA", "XAR", "LMT", "NOC", "RTX", "GD"],
        "drivers": [
            "Forsvarsbudgetter",
            "Lagergenopbygning",
            "Geopolitisk spænding",
            "NATO-spending",
        ],
        "headwinds": [
            "Budgetskifte",
            "Kontraktforsinkelser",
            "Forsyningskædepres",
        ],
        "macro_series": {
            "10Y rente": "DGS10",
            "Industriproduktion": "INDPRO",
            "Olie": "DCOILWTICO",
        },
        "description": "Defense/Aerospace støttes af stigende forsvarsbudgetter, lagre og et langvarigt sikkerhedspolitisk fokus.",
    },
    "Clean Energy": {
        "proxy": "ICLN",
        "benchmark": "SPY",
        "members": ["ICLN", "PBW", "TAN", "ENPH", "FSLR"],
        "drivers": [
            "Subsidier",
            "Netudbygning",
            "Teknologiomkostninger",
            "Elektrificering",
        ],
        "headwinds": [
            "Høje renter",
            "Projektforsinkelser",
            "Politisk usikkerhed",
            "Volatile inputpriser",
        ],
        "macro_series": {
            "10Y rente": "DGS10",
            "Inflation": "CPIAUCSL",
            "Olie": "DCOILWTICO",
        },
        "description": "Clean Energy er rentefølsomt og afhænger af subsidier, netudbygning og konkurrenceevne mod fossile alternativer.",
    },
    "Banks": {
        "proxy": "XLF",
        "benchmark": "SPY",
        "members": ["XLF", "JPM", "BAC", "WFC", "GS", "MS"],
        "drivers": [
            "Rentekurve",
            "Kreditvækst",
            "Net interest margin",
            "Kapitalmarkedsaktivitet",
        ],
        "headwinds": [
            "Kreditforringelser",
            "Flad rentekurve",
            "Regulatorisk pres",
        ],
        "macro_series": {
            "10Y rente": "DGS10",
            "2Y rente": "DGS2",
            "Arbejdsløshed": "UNRATE",
        },
        "description": "Banker påvirkes især af rentekurven, kreditkvalitet og økonomisk aktivitet.",
    },
    "Gold": {
        "proxy": "GLD",
        "benchmark": "SPY",
        "members": ["GLD", "IAU", "NEM", "AEM"],
        "drivers": [
            "Reale renter",
            "USD-retning",
            "Inflationsforventninger",
            "Geopolitisk risiko",
        ],
        "headwinds": [
            "Stigende reale renter",
            "Stærk USD",
            "Lav risikofrygt",
        ],
        "macro_series": {
            "10Y rente": "DGS10",
            "Inflation": "CPIAUCSL",
            "USD indeks": "DTWEXBGS",
        },
        "description": "Guld fungerer ofte som hedge mod realrenter, dollarstyrke og geopolitisk usikkerhed.",
    },
}

YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}


# =========================================================
# SECRETS / KEYS
# =========================================================
def get_secret(name: str) -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:
        return os.getenv(name, "").strip()


TD_API_KEY = get_secret("TWELVE_DATA_API_KEY")
FRED_API_KEY = get_secret("FRED_API_KEY")


# =========================================================
# GENERIC HELPERS
# =========================================================
def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
    return requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def safe_display_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B"
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value:,.0f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


# =========================================================
# SYMBOL HELPERS
# =========================================================
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
    return f"{t}{suffix[ex]}" if ex in suffix else t


def make_td_symbol(ticker: str, exchange: str) -> str:
    t = str(ticker).strip().upper()
    ex = str(exchange).strip().upper()

    if not t:
        return ""
    if ex in ("NASDAQ", "NYSE", "AMEX"):
        return t
    if ex:
        return f"{t}:{ex}"
    return t


def yahoo_symbol_candidates(symbol: str) -> List[str]:
    s = (symbol or "").strip().upper()
    if not s:
        return []

    out = [s]
    replacements = {
        ".XCSE": ".CO", ".XSTO": ".ST", ".XHEL": ".HE", ".XOSL": ".OL",
        ".XPAR": ".PA", ".XAMS": ".AS", ".XBRU": ".BR", ".XLON": ".L",
        ".XMIL": ".MI", ".XMAD": ".MC", ".XSWX": ".SW", ".XTKS": ".T",
        ".XHKG": ".HK", ".XNSE": ".NS", ".XBOM": ".BO", ".XTSE": ".TO",
        ".XTSX": ".V", ".XETRA": ".DE", ".FWB": ".DE",
        ":XCSE": ".CO", ":XSTO": ".ST", ":XHEL": ".HE", ":XOSL": ".OL",
        ":XPAR": ".PA", ":XAMS": ".AS", ":XBRU": ".BR", ":XLON": ".L",
        ":XMIL": ".MI", ":XMAD": ".MC", ":XSWX": ".SW", ":XTKS": ".T",
        ":XHKG": ".HK", ":XNSE": ".NS", ":XBOM": ".BO", ":XTSE": ".TO",
        ":XTSX": ".V", ":XETRA": ".DE", ":FWB": ".DE",
    }
    for old, new in replacements.items():
        if old in s:
            out.append(s.replace(old, new))
    if ":" in s:
        left, right = s.split(":", 1)
        out.extend([left, right])

    seen = set()
    final = []
    for x in out:
        if x and x not in seen:
            final.append(x)
            seen.add(x)
    return final


# =========================================================
# UNIVERSE HELPERS
# =========================================================
def ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol", "td_symbol"]
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

    for col in ("name", "country", "exchange", "type", "source", "yahoo_symbol", "td_symbol"):
        if col not in df.columns:
            df[col] = ""

    for col in ("ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol", "td_symbol"):
        df[col] = df[col].astype(str).fillna("").str.strip()

    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker", "exchange"])
    return df[["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol", "td_symbol"]].reset_index(drop=True)


def universe_file(key: str) -> str:
    return os.path.join(UNIVERSE_DIR, f"{key}.csv")


def list_universes() -> Dict[str, str]:
    out = {}
    if not os.path.exists(UNIVERSE_DIR):
        return out
    for fn in sorted(os.listdir(UNIVERSE_DIR)):
        if fn.endswith(".csv"):
            out[fn[:-4]] = os.path.join(UNIVERSE_DIR, fn)
    return out


def load_universe(key: str) -> Tuple[pd.DataFrame, str]:
    path = list_universes().get(key, "")
    if not path:
        return pd.DataFrame(), "Ukendt univers."
    df = safe_read_csv(path)
    df = ensure_universe_schema(df)
    if df.empty:
        return df, f"Tomt eller ulæseligt univers: {path}"
    return df, ""


# =========================================================
# TWELVE DATA
# =========================================================
def td_get(endpoint: str, params: Optional[dict] = None) -> dict:
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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def td_fetch_history(symbol: str, years: int = 10) -> pd.DataFrame:
    outputsize = min(5000, max(300, int(years) * 260))
    data = td_get(
        "time_series",
        {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": outputsize,
            "format": "JSON",
            "order": "ASC",
        },
    )

    values = data.get("values", [])
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
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def td_fetch_stocks(exchange: str = "", country: str = "") -> pd.DataFrame:
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


# =========================================================
# YAHOO
# =========================================================
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def yahoo_fetch_history(symbol: str, years: int = 10) -> pd.DataFrame:
    range_str = "10y" if years >= 10 else f"{max(1, int(years))}y"

    for sym in yahoo_symbol_candidates(symbol):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
            r = http_get(
                url,
                params={"interval": "1d", "range": range_str, "includeAdjustedClose": "true"},
                headers=YAHOO_HEADERS,
            )
            if r.status_code != 200:
                continue

            js = r.json()
            result = js.get("chart", {}).get("result", [])
            if not result:
                continue

            res0 = result[0]
            timestamps = res0.get("timestamp", [])
            quote = ((res0.get("indicators", {}) or {}).get("quote", [{}]) or [{}])[0]
            if not timestamps:
                continue

            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(timestamps, unit="s", errors="coerce"),
                    "Open": pd.to_numeric(quote.get("open", []), errors="coerce"),
                    "High": pd.to_numeric(quote.get("high", []), errors="coerce"),
                    "Low": pd.to_numeric(quote.get("low", []), errors="coerce"),
                    "Close": pd.to_numeric(quote.get("close", []), errors="coerce"),
                    "Volume": pd.to_numeric(quote.get("volume", []), errors="coerce"),
                }
            )
            df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
            if not df.empty:
                return df
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def yahoo_fetch_overview(symbol: str) -> dict:
    for sym in yahoo_symbol_candidates(symbol):
        try:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
            r = http_get(
                url,
                params={"modules": "price,summaryDetail,defaultKeyStatistics,financialData"},
                headers=YAHOO_HEADERS,
            )
            if r.status_code != 200:
                continue

            js = r.json()
            result = js.get("quoteSummary", {}).get("result", [])
            if not result:
                continue

            data = result[0]

            def raw(mod: str, key: str):
                val = ((data.get(mod, {}) or {}).get(key, {}))
                if isinstance(val, dict):
                    return val.get("raw")
                return val

            return {
                "pe": raw("summaryDetail", "trailingPE") or raw("defaultKeyStatistics", "trailingPE"),
                "forward_pe": raw("summaryDetail", "forwardPE"),
                "market_cap": raw("price", "marketCap"),
                "avg_volume": raw("summaryDetail", "averageVolume"),
                "beta": raw("defaultKeyStatistics", "beta"),
                "fifty_two_week_high": raw("summaryDetail", "fiftyTwoWeekHigh"),
                "fifty_two_week_low": raw("summaryDetail", "fiftyTwoWeekLow"),
                "dividend_yield": raw("summaryDetail", "dividendYield"),
            }
        except Exception:
            continue

    return {}


# =========================================================
# MAIN FETCH LOGIC
# =========================================================
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_history(ticker: str, td_symbol: str = "", yahoo_symbol: str = "", years: int = 10) -> Tuple[pd.DataFrame, str]:
    td_candidates = []
    if td_symbol:
        td_candidates.append(td_symbol)
    if ticker:
        td_candidates.append(ticker)

    seen = set()
    td_candidates = [x for x in td_candidates if not (x in seen or seen.add(x))]

    for sym in td_candidates:
        df = td_fetch_history(sym, years=years)
        if not df.empty:
            return df, "Twelve Data"

    yh_candidates = []
    if yahoo_symbol:
        yh_candidates.append(yahoo_symbol)
    if ticker:
        yh_candidates.append(ticker)

    seen = set()
    yh_candidates = [x for x in yh_candidates if not (x in seen or seen.add(x))]

    for sym in yh_candidates:
        df = yahoo_fetch_history(sym, years=years)
        if not df.empty:
            return df, "Yahoo"

    return pd.DataFrame(), ""


# =========================================================
# FRED
# =========================================================
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_fred_series(series_id: str, limit_years: int = 12) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "json",
    }
    if FRED_API_KEY:
        params["api_key"] = FRED_API_KEY

    try:
        r = http_get(FRED_BASE, params=params)
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        obs = js.get("observations", [])
        if not obs:
            return pd.DataFrame()

        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if limit_years > 0 and not df.empty:
            cutoff = df["date"].max() - pd.Timedelta(days=int(365.25 * limit_years))
            df = df[df["date"] >= cutoff]
        return df[["date", "value"]].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# =========================================================
# UNIVERSE BUILDER
# =========================================================
def build_universe_for_country(country_name: str, exchanges: List[str]) -> pd.DataFrame:
    frames = []

    for ex in exchanges:
        df = td_fetch_stocks(exchange=ex, country=country_name)
        if df.empty:
            continue

        df = normalize_cols(df)
        if "symbol" not in df.columns:
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
        df["yahoo_symbol"] = df.apply(lambda r: make_yahoo_symbol(r.get("symbol", ""), r.get("exchange", ex)), axis=1)
        df["td_symbol"] = df.apply(lambda r: make_td_symbol(r.get("symbol", ""), r.get("exchange", ex)), axis=1)

        keep = ["symbol", "name", "country", "exchange", "type", "source", "yahoo_symbol", "td_symbol"]
        frames.append(df[keep].rename(columns={"symbol": "ticker"}))

    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol", "td_symbol"])

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "exchange"])
    return ensure_universe_schema(out)


def build_all_universes(selected_countries: List[str]) -> Dict[str, int]:
    counts = {}
    all_frames = []

    for country in selected_countries:
        exchanges = TD_EXCHANGES.get(country, [])
        df = build_universe_for_country(country, exchanges)
        key = country.lower().replace(" ", "_")
        df.to_csv(universe_file(key), index=False, encoding="utf-8")
        counts[key] = len(df)
        if not df.empty:
            all_frames.append(df)

    if all_frames:
        global_df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["ticker", "exchange"])
    else:
        global_df = pd.DataFrame(columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol", "td_symbol"])

    global_df = ensure_universe_schema(global_df)
    global_df.to_csv(universe_file("global_all"), index=False, encoding="utf-8")
    counts["global_all"] = len(global_df)

    usa_path = universe_file("usa")
    usa_df = safe_read_csv(usa_path)
    if not usa_df.empty:
        usa_df = ensure_universe_schema(usa_df)
        usa_df.to_csv(universe_file("us_all"), index=False, encoding="utf-8")
        counts["us_all"] = len(usa_df)

    return counts


# =========================================================
# TECHNICALS
# =========================================================
def pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a / b - 1.0) * 100.0


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()
    d["RSI14"] = rsi(d["Close"], 14)

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIGNAL"]

    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["BB_MID"] = sma20
    d["BB_UPPER"] = sma20 + 2 * std20
    d["BB_LOWER"] = sma20 - 2 * std20
    return d


def latest_technical_snapshot(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    d = add_technical_columns(df).dropna(subset=["Close"]).copy()
    if d.empty:
        return {}

    close = d["Close"]
    last = float(close.iloc[-1])

    ema20 = float(d["EMA20"].iloc[-1]) if pd.notna(d["EMA20"].iloc[-1]) else np.nan
    ema50 = float(d["EMA50"].iloc[-1]) if pd.notna(d["EMA50"].iloc[-1]) else np.nan
    ema200 = float(d["EMA200"].iloc[-1]) if pd.notna(d["EMA200"].iloc[-1]) else np.nan
    rsi14 = float(d["RSI14"].iloc[-1]) if pd.notna(d["RSI14"].iloc[-1]) else np.nan
    macd = float(d["MACD"].iloc[-1]) if pd.notna(d["MACD"].iloc[-1]) else np.nan
    macd_signal = float(d["MACD_SIGNAL"].iloc[-1]) if pd.notna(d["MACD_SIGNAL"].iloc[-1]) else np.nan
    bb_upper = float(d["BB_UPPER"].iloc[-1]) if pd.notna(d["BB_UPPER"].iloc[-1]) else np.nan
    bb_lower = float(d["BB_LOWER"].iloc[-1]) if pd.notna(d["BB_LOWER"].iloc[-1]) else np.nan

    ret = close.pct_change().dropna()
    vol20 = float(ret.rolling(20).std().iloc[-1] * 100.0) if len(ret) >= 25 else np.nan

    dd = np.nan
    if len(close) >= 63:
        peak = float(close.iloc[-63:].max())
        if peak != 0:
            dd = (last / peak - 1.0) * 100.0

    trend_up = False
    if not np.isnan(ema50) and not np.isnan(ema200):
        trend_up = ema50 > ema200
    elif not np.isnan(ema20) and not np.isnan(ema50):
        trend_up = ema20 > ema50

    if np.isnan(rsi14):
        rsi_state = "—"
    elif rsi14 >= 70:
        rsi_state = "Overkøbt"
    elif rsi14 <= 30:
        rsi_state = "Oversolgt"
    else:
        rsi_state = "Neutral"

    macd_state = "Bullish" if (not np.isnan(macd) and not np.isnan(macd_signal) and macd > macd_signal) else "Bearish"

    if np.isnan(bb_upper) or np.isnan(bb_lower):
        bb_state = "—"
    elif last > bb_upper:
        bb_state = "Over upper band"
    elif last < bb_lower:
        bb_state = "Under lower band"
    else:
        bb_state = "Inside bands"

    score = 0.0
    score += 2.0 if trend_up else 0.0
    if not np.isnan(rsi14):
        score += max(0.0, 2.0 - abs(rsi14 - 50) / 25)
    if len(close) >= 21:
        mom20 = pct_change(last, float(close.iloc[-21]))
        if not np.isnan(mom20):
            score += max(0.0, min(3.0, mom20 / 5.0))
    else:
        mom20 = np.nan
    if not np.isnan(dd):
        score += max(0.0, min(2.0, (10.0 + dd) / 10.0))

    risk = "OK"
    if (not np.isnan(vol20) and vol20 > 4.5) or (not np.isnan(dd) and dd < -15):
        risk = "Høj"
    if not np.isnan(dd) and dd < -25:
        risk = "Meget høj"

    buy_zone = trend_up and (not np.isnan(rsi14)) and (35 <= rsi14 <= 60) and (not np.isnan(mom20)) and (mom20 >= 0)
    sell_zone = ((not trend_up) and (not np.isnan(rsi14)) and (rsi14 < 40)) or (risk == "Meget høj")

    if sell_zone:
        action = "SÆLG / UNDGÅ"
    elif buy_zone:
        action = "KØB / KIG NÆRMERE"
    else:
        action = "HOLD / AFVENT"

    return {
        "last": last,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi14,
        "rsi_state": rsi_state,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_state": macd_state,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_state": bb_state,
        "vol20": vol20,
        "dd3m": dd,
        "mom20": mom20,
        "trend_up": trend_up,
        "risk": risk,
        "score": round(score, 2),
        "action": action,
    }


def period_returns(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return {}

    d = df[["Date", "Close"]].dropna().copy()
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna(subset=["Close"]).sort_values("Date")
    if d.empty:
        return {}

    last_date = d["Date"].iloc[-1]
    last_close = float(d["Close"].iloc[-1])

    def close_on_or_before(target_date: pd.Timestamp) -> float:
        sub = d[d["Date"] <= target_date]
        if sub.empty:
            return np.nan
        return float(sub["Close"].iloc[-1])

    out = {}
    out["1D"] = pct_change(last_close, float(d["Close"].iloc[-2])) if len(d) >= 2 else np.nan
    out["1W"] = pct_change(last_close, close_on_or_before(last_date - pd.Timedelta(days=7)))
    out["1M"] = pct_change(last_close, close_on_or_before(last_date - pd.Timedelta(days=30)))
    out["3M"] = pct_change(last_close, close_on_or_before(last_date - pd.Timedelta(days=90)))
    out["6M"] = pct_change(last_close, close_on_or_before(last_date - pd.Timedelta(days=182)))
    out["YTD"] = pct_change(last_close, close_on_or_before(pd.Timestamp(year=last_date.year, month=1, day=1)))
    out["1Y"] = pct_change(last_close, close_on_or_before(last_date - pd.Timedelta(days=365)))
    out["3Y"] = pct_change(last_close, close_on_or_before(last_date - pd.Timedelta(days=365 * 3)))
    out["5Y"] = pct_change(last_close, close_on_or_before(last_date - pd.Timedelta(days=365 * 5)))
    out["MAX"] = pct_change(last_close, float(d["Close"].iloc[0]))
    return out


# =========================================================
# LOG
# =========================================================
def append_signal_log(ticker: str, action: str, score: float, last: float) -> None:
    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "ticker": ticker,
        "action": action,
        "score": score,
        "last": last,
    }
    new_df = pd.DataFrame([row])

    if os.path.exists(SIGNAL_LOG):
        try:
            old = pd.read_csv(SIGNAL_LOG)
            out = pd.concat([old, new_df], ignore_index=True)
        except Exception:
            out = new_df
    else:
        out = new_df

    out.to_csv(SIGNAL_LOG, index=False, encoding="utf-8")


def read_signal_log(ticker: str) -> pd.DataFrame:
    if not os.path.exists(SIGNAL_LOG):
        return pd.DataFrame()
    try:
        df = pd.read_csv(SIGNAL_LOG)
        if "ticker" not in df.columns:
            return pd.DataFrame()
        return df[df["ticker"].astype(str) == str(ticker)].tail(50).copy()
    except Exception:
        return pd.DataFrame()


# =========================================================
# THEME ENGINE
# =========================================================
def year_by_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Year", "Return %"])

    d = df[["Date", "Close"]].dropna().copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna().sort_values("Date")
    if d.empty:
        return pd.DataFrame(columns=["Year", "Return %"])

    d["Year"] = d["Date"].dt.year
    rows = []
    for year, grp in d.groupby("Year"):
        grp = grp.sort_values("Date")
        start = float(grp["Close"].iloc[0])
        end = float(grp["Close"].iloc[-1])
        ret = (end / start - 1.0) * 100 if start else np.nan
        rows.append({"Year": int(year), "Return %": ret})

    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)


def cumulative_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["Date", "Close"]].dropna().copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna().sort_values("Date")
    if d.empty:
        return pd.DataFrame(columns=["Date", "Index"])
    base = float(d["Close"].iloc[0])
    d["Index"] = d["Close"] / base * 100.0 if base else np.nan
    return d[["Date", "Index"]].reset_index(drop=True)


def calc_drawdown(close: pd.Series) -> float:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if s.empty:
        return np.nan
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min() * 100.0)


def calc_cagr(df: pd.DataFrame) -> float:
    d = df[["Date", "Close"]].dropna().copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna().sort_values("Date")
    if len(d) < 2:
        return np.nan
    start = float(d["Close"].iloc[0])
    end = float(d["Close"].iloc[-1])
    years = (d["Date"].iloc[-1] - d["Date"].iloc[0]).days / 365.25
    if start <= 0 or years <= 0:
        return np.nan
    return float((end / start) ** (1 / years) - 1.0) * 100.0


def calc_relative_strength(proxy_df: pd.DataFrame, bench_df: pd.DataFrame, window_days: int) -> float:
    if proxy_df.empty or bench_df.empty:
        return np.nan

    a = proxy_df[["Date", "Close"]].dropna().copy()
    b = bench_df[["Date", "Close"]].dropna().copy()
    a["Date"] = pd.to_datetime(a["Date"], errors="coerce")
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce")
    a = a.sort_values("Date")
    b = b.sort_values("Date")

    end = min(a["Date"].iloc[-1], b["Date"].iloc[-1])
    start = end - pd.Timedelta(days=window_days)

    def close_on_or_before(df_: pd.DataFrame, dt: pd.Timestamp) -> float:
        sub = df_[df_["Date"] <= dt]
        if sub.empty:
            return np.nan
        return float(sub["Close"].iloc[-1])

    a0 = close_on_or_before(a, start)
    a1 = close_on_or_before(a, end)
    b0 = close_on_or_before(b, start)
    b1 = close_on_or_before(b, end)

    vals = [a0, a1, b0, b1]
    if any(pd.isna(x) for x in vals) or a0 == 0 or b0 == 0:
        return np.nan

    return float((a1 / a0 - 1.0) - (b1 / b0 - 1.0))


def calc_speed_label(rs_1m: float, rs_3m: float, ema20_slope: float, macd_hist_delta: float) -> str:
    accel = 0.0
    for x in [rs_1m, rs_3m, ema20_slope, macd_hist_delta]:
        if pd.notna(x):
            accel += float(x)

    if accel > 8:
        return "Acceleration op"
    if accel > 2:
        return "Stabil optrend"
    if accel > -2:
        return "Sideways / neutral"
    if accel > -8:
        return "Aftagende styrke"
    return "Acceleration ned"


def calc_regime_label(close: pd.Series) -> str:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 220:
        return "For lidt data"

    ema20 = s.ewm(span=20, adjust=False).mean()
    ema50 = s.ewm(span=50, adjust=False).mean()
    ema200 = s.ewm(span=200, adjust=False).mean()

    c = float(s.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    e200 = float(ema200.iloc[-1])

    if c > e20 > e50 > e200:
        return "Stærk optrend"
    if c > e50 > e200:
        return "Optrend"
    if c < e20 < e50 < e200:
        return "Stærk nedtrend"
    if c < e50 < e200:
        return "Nedtrend"
    return "Blandet / overgang"


def build_theme_yearly_comparison(proxy_df: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
    py = year_by_year_returns(proxy_df)
    by = year_by_year_returns(bench_df)
    if py.empty:
        return pd.DataFrame(columns=["Year", "Theme %", "Benchmark %", "Alpha %"])

    out = py.merge(by, on="Year", how="left", suffixes=(" Theme", " Benchmark"))
    out = out.rename(columns={"Return % Theme": "Theme %", "Return % Benchmark": "Benchmark %"})
    if "Benchmark %" not in out.columns:
        out["Benchmark %"] = np.nan
    out["Alpha %"] = out["Theme %"] - out["Benchmark %"]
    return out.sort_values("Year").reset_index(drop=True)


def build_theme_rankings() -> pd.DataFrame:
    rows = []

    for theme_name, cfg in THEMES.items():
        proxy = cfg["proxy"]
        bench = cfg.get("benchmark", "SPY")

        proxy_df, _ = fetch_history(proxy, td_symbol=proxy, yahoo_symbol=proxy, years=10)
        bench_df, _ = fetch_history(bench, td_symbol=bench, yahoo_symbol=bench, years=10)

        if proxy_df.empty or bench_df.empty:
            continue

        close = pd.to_numeric(proxy_df["Close"], errors="coerce").dropna()
        if len(close) < 220:
            continue

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        rsi14 = rsi(close, 14)

        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        rs_1m = calc_relative_strength(proxy_df, bench_df, 30)
        rs_3m = calc_relative_strength(proxy_df, bench_df, 90)
        rs_6m = calc_relative_strength(proxy_df, bench_df, 180)

        trend_up = bool(ema50.iloc[-1] > ema200.iloc[-1])

        score_trend = 0.0
        if trend_up:
            score_trend += 25.0
        if float(close.iloc[-1]) > float(ema200.iloc[-1]):
            score_trend += 10.0
        if pd.notna(rsi14.iloc[-1]):
            rr = float(rsi14.iloc[-1])
            score_trend += max(0.0, 10.0 - abs(rr - 55) / 2.5)

        score_relative = 0.0
        for rs_, w in [(rs_1m, 20.0), (rs_3m, 30.0), (rs_6m, 20.0)]:
            if pd.notna(rs_):
                score_relative += max(-w, min(w, rs_ * 100.0))

        ema20_slope = np.nan
        if len(ema20) >= 21:
            ema20_slope = float((ema20.iloc[-1] / ema20.iloc[-21] - 1.0) * 100.0)

        macd_hist_delta = np.nan
        if len(macd_hist) >= 20:
            macd_hist_delta = float(macd_hist.iloc[-1] - macd_hist.iloc[-20])

        score_acceleration = 0.0
        for x, scale in [(ema20_slope, 2.0), (macd_hist_delta, 20.0)]:
            if pd.notna(x):
                score_acceleration += max(-15.0, min(15.0, x * scale))

        dd = calc_drawdown(close)
        score_stability = 0.0 if pd.isna(dd) else max(-20.0, min(10.0, 10.0 + dd / 2.0))

        # breadth
        members = cfg.get("members", [])[:8]
        breadth_vals = []
        for m in members:
            mdf, _ = fetch_history(m, td_symbol=m, yahoo_symbol=m, years=5)
            if mdf.empty or "Close" not in mdf.columns:
                continue
            mc = pd.to_numeric(mdf["Close"], errors="coerce").dropna()
            if len(mc) < 220:
                continue
            me50 = mc.ewm(span=50, adjust=False).mean()
            me200 = mc.ewm(span=200, adjust=False).mean()
            mrs3 = calc_relative_strength(mdf, bench_df, 90)
            local = 0.0
            if bool(me50.iloc[-1] > me200.iloc[-1]):
                local += 1.0
            if pd.notna(mrs3) and mrs3 > 0:
                local += 1.0
            breadth_vals.append(local)

        score_breadth = float(np.mean(breadth_vals) * 10.0) if breadth_vals else 0.0
        total = score_trend + score_relative + score_acceleration + score_stability + score_breadth

        speed_label = calc_speed_label(
            rs_1m=(0.0 if pd.isna(rs_1m) else rs_1m * 100.0),
            rs_3m=(0.0 if pd.isna(rs_3m) else rs_3m * 100.0),
            ema20_slope=(0.0 if pd.isna(ema20_slope) else ema20_slope),
            macd_hist_delta=(0.0 if pd.isna(macd_hist_delta) else macd_hist_delta * 100.0),
        )
        regime_label = calc_regime_label(close)

        rows.append(
            {
                "Tema": theme_name,
                "Proxy": proxy,
                "Samlet score": round(total, 2),
                "Trend": round(score_trend, 2),
                "Relativ styrke": round(score_relative, 2),
                "Acceleration": round(score_acceleration, 2),
                "Stabilitet": round(score_stability, 2),
                "Breadth": round(score_breadth, 2),
                "Regime": regime_label,
                "Hastighed": speed_label,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Samlet score", ascending=False).reset_index(drop=True)


# =========================================================
# GOOGLE NEWS
# =========================================================
def google_news_link(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/search?q={q}&hl=da&gl=DK&ceid=DK%3Ada"


# =========================================================
# SESSION STATE
# =========================================================
if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = []


# =========================================================
# UI
# =========================================================
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Indstillinger")
    top_n = st.slider("Top N (screening)", 5, 50, DEFAULT_TOPN, 1)
    years = st.slider("Historik (år)", 1, 10, 5, 1)
    max_screen = st.slider("Max tickers pr. screening", 50, 2000, 500, 50)
    min_price = st.number_input("Min pris", min_value=0.0, value=1.0, step=1.0)
    only_common_stocks = st.checkbox("Kun almindelige aktier", value=True)

    st.divider()
    st.subheader("🔑 Datakilde")
    if TD_API_KEY:
        st.success("Twelve Data key indlæst i dette miljø")
    else:
        st.warning("Ingen Twelve Data key fundet i dette miljø. Yahoo fallback bruges.")

    st.divider()
    st.subheader("🌍 Universe builder")
    selected_countries = st.multiselect(
        "Vælg lande/markeder",
        list(TD_EXCHANGES.keys()),
        default=["USA", "Germany", "Denmark", "Sweden", "United Kingdom"],
    )

    if st.button("Byg / opdater universer", type="primary"):
        if not TD_API_KEY:
            st.error("Sæt TWELVE_DATA_API_KEY først.")
        elif not selected_countries:
            st.error("Vælg mindst ét land.")
        else:
            with st.spinner("Henter symboler fra Twelve Data ..."):
                counts = build_all_universes(selected_countries)
            st.success("Universer er opdateret.")
            st.json(counts)
            st.rerun()

    st.divider()
    st.subheader("📌 Hjælp")
    st.markdown(
        """
- **Søg & analyse**: kurs, P/E, RSI, trend, teknisk analyse
- **Screening**: global prefilter + Top N scoring
- **Portefølje**: beholdninger med signaler og nøgletal
- **Tema**: dyb temaanalyse med historik og drivere
        """
    )

tab_search, tab_screener, tab_portfolio, tab_themes, tab_data = st.tabs(
    ["🔎 Søg & analyse", "🏁 Screening", "💼 Portefølje", "🧭 Tema", "🛠 Data"]
)


# =========================================================
# TAB 1 SEARCH
# =========================================================
with tab_search:
    st.subheader("🔎 Søg & analyse")

    files = list_universes()
    if not files:
        st.info("Ingen univers-filer endnu. Brug 'Byg / opdater universer' i sidebaren.")
    else:
        left, right = st.columns([1, 2])

        with left:
            keys = sorted(files.keys())
            default_idx = keys.index("global_all") if "global_all" in keys else 0
            universe_key = st.selectbox("Vælg univers", keys, index=default_idx)

            uni, err = load_universe(universe_key)
            if err:
                st.error(err)

            ticker = ""
            td_symbol = ""
            yahoo_symbol = ""
            name = ""
            country = ""
            exchange = ""

            if not uni.empty:
                uni = uni.copy()
                uni["display"] = uni.apply(
                    lambda r: f"{r['ticker']} — {r['name']}" if str(r.get("name", "")).strip() else f"{r['ticker']}",
                    axis=1,
                )

                q = st.text_input("Søg ticker eller navn", "", key="search_q")
                view = uni
                if q.strip():
                    qq = q.strip().lower()
                    view = view[
                        view["ticker"].astype(str).str.lower().str.contains(qq, na=False)
                        | view["name"].astype(str).str.lower().str.contains(qq, na=False)
                    ]

                if view.empty:
                    st.info("Ingen match.")
                else:
                    selection = st.selectbox("Vælg papir", view["display"].tolist(), index=0)
                    row = view[view["display"] == selection].iloc[0]
                    ticker = str(row["ticker"]).strip()
                    td_symbol = str(row.get("td_symbol", "")).strip()
                    yahoo_symbol = str(row.get("yahoo_symbol", "")).strip()
                    name = str(row.get("name", "")).strip()
                    country = str(row.get("country", "")).strip()
                    exchange = str(row.get("exchange", "")).strip()

                    st.caption(f"Ticker: **{ticker}**")
                    if td_symbol:
                        st.caption(f"Twelve Data symbol: {td_symbol}")
                    if yahoo_symbol:
                        st.caption(f"Yahoo fallback: {yahoo_symbol}")
                    if name:
                        st.caption(f"Navn: {name}")
                    if country:
                        st.caption(f"Land: {country}")
                    if exchange:
                        st.caption(f"Exchange: {exchange}")

        with right:
            if ticker:
                df, source_used = fetch_history(ticker=ticker, td_symbol=td_symbol, yahoo_symbol=yahoo_symbol, years=years)
                overview = yahoo_fetch_overview(yahoo_symbol or ticker)
                tech = latest_technical_snapshot(df)
                rets = period_returns(df)

                if df.empty:
                    st.error("Kunne ikke hente kursdata fra Twelve Data eller Yahoo fallback.")
                else:
                    last = float(df["Close"].iloc[-1])
                    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
                    chg = (last / prev - 1.0) * 100 if prev else 0.0

                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Seneste close", f"{last:,.2f}")
                    m2.metric("Dag %", f"{chg:+.2f}%")
                    m3.metric("RSI14", "—" if pd.isna(tech.get("rsi", np.nan)) else f"{tech['rsi']:.1f}")
                    m4.metric("Signal", tech.get("action", "—"))
                    m5.metric("Trend", "Op" if tech.get("trend_up") else "Ned")
                    m6.metric("Kilde", source_used or "—")

                    if tech:
                        append_signal_log(
                            ticker=ticker,
                            action=str(tech.get("action", "")),
                            score=float(tech.get("score", np.nan)),
                            last=float(tech.get("last", np.nan)),
                        )

                    st.markdown("#### Overblik / nøgletal")
                    overview_rows = {
                        "P/E": overview.get("pe"),
                        "Forward P/E": overview.get("forward_pe"),
                        "Market Cap": overview.get("market_cap"),
                        "Avg Volume": overview.get("avg_volume"),
                        "Beta": overview.get("beta"),
                        "Dividend Yield": overview.get("dividend_yield"),
                        "52W High": overview.get("fifty_two_week_high"),
                        "52W Low": overview.get("fifty_two_week_low"),
                        "RSI State": tech.get("rsi_state"),
                        "MACD State": tech.get("macd_state"),
                        "BBands": tech.get("bb_state"),
                        "Vol20 %": tech.get("vol20"),
                        "Drawdown 3m %": tech.get("dd3m"),
                        "Momentum 20d %": tech.get("mom20"),
                    }
                    show_overview = pd.DataFrame(
                        [{"Felt": str(k), "Værdi": safe_display_value(v)} for k, v in overview_rows.items()]
                    )
                    st.dataframe(show_overview, use_container_width=True, hide_index=True)

                    st.markdown("#### Periode-afkast")
                    labels = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
                    for i in range(0, len(labels), 5):
                        chunk = labels[i:i + 5]
                        cols = st.columns(len(chunk))
                        for j, k in enumerate(chunk):
                            v = rets.get(k, np.nan)
                            txt = "—" if pd.isna(v) else f"{v:+.2f}%"
                            cols[j].metric(k, txt)

                    st.markdown("#### Kurs")
                    st.line_chart(df.set_index("Date")["Close"])

                    st.markdown("#### Teknisk analyse")
                    tech_df = add_technical_columns(df)
                    st.markdown("##### Pris vs EMA")
                    st.line_chart(tech_df.set_index("Date")[["Close", "EMA20", "EMA50", "EMA200"]].dropna(how="all"))

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("##### RSI14")
                        st.line_chart(tech_df.set_index("Date")[["RSI14"]].dropna())
                    with c2:
                        st.markdown("##### MACD")
                        st.line_chart(tech_df.set_index("Date")[["MACD", "MACD_SIGNAL"]].dropna())

                    st.markdown("##### Bollinger Bands")
                    st.line_chart(
                        tech_df.set_index("Date")[["Close", "BB_UPPER", "BB_MID", "BB_LOWER"]].dropna(how="all")
                    )

                    st.markdown("#### Seneste tekniske snapshot")
                    tech_snapshot = pd.DataFrame(
                        [
                            {"Felt": "EMA20", "Værdi": safe_display_value(tech.get("ema20"))},
                            {"Felt": "EMA50", "Værdi": safe_display_value(tech.get("ema50"))},
                            {"Felt": "EMA200", "Værdi": safe_display_value(tech.get("ema200"))},
                            {"Felt": "RSI14", "Værdi": safe_display_value(tech.get("rsi"))},
                            {"Felt": "MACD", "Værdi": safe_display_value(tech.get("macd"))},
                            {"Felt": "MACD Signal", "Værdi": safe_display_value(tech.get("macd_signal"))},
                            {"Felt": "BB Upper", "Værdi": safe_display_value(tech.get("bb_upper"))},
                            {"Felt": "BB Lower", "Værdi": safe_display_value(tech.get("bb_lower"))},
                            {"Felt": "Risiko", "Værdi": safe_display_value(tech.get("risk"))},
                            {"Felt": "Score", "Værdi": safe_display_value(tech.get("score"))},
                            {"Felt": "Handling", "Værdi": safe_display_value(tech.get("action"))},
                        ]
                    )
                    st.dataframe(tech_snapshot, use_container_width=True, hide_index=True)

                    st.markdown("#### Nyheder")
                    st.markdown(f"[Google News]({google_news_link(f'{ticker} {name}'.strip())})")

                    with st.expander("Signal-log"):
                        hist = read_signal_log(ticker)
                        if hist.empty:
                            st.info("Ingen log endnu.")
                        else:
                            st.dataframe(hist, use_container_width=True, hide_index=True)


# =========================================================
# TAB 2 SCREENER
# =========================================================
with tab_screener:
    st.subheader("🏁 Screening")

    files = list_universes()
    if not files:
        st.info("Ingen univers-filer endnu.")
    else:
        keys = sorted(files.keys())
        default_idx = keys.index("global_all") if "global_all" in keys else 0
        universe_key = st.selectbox("Vælg univers til screening", keys, index=default_idx, key="screen_uni")

        uni, err = load_universe(universe_key)
        if err:
            st.error(err)

        if uni.empty:
            st.warning("Tomt univers.")
        else:
            country_filter = st.multiselect(
                "Land filter (valgfrit)",
                sorted([x for x in uni["country"].dropna().astype(str).unique().tolist() if x]),
                default=[],
            )

            if st.button("Kør screening", type="primary"):
                work = uni.copy()

                if country_filter:
                    work = work[work["country"].isin(country_filter)]

                if only_common_stocks and "type" in work.columns:
                    mask = ~work["type"].astype(str).str.lower().str.contains("etf|fund|warrant|bond|preferred", na=False)
                    work = work[mask]

                if len(work) > max_screen:
                    work = work.sample(n=max_screen, random_state=42).reset_index(drop=True)
                else:
                    work = work.reset_index(drop=True)

                rows = []
                prog = st.progress(0)
                status = st.empty()

                for i, (_, r) in enumerate(work.iterrows(), start=1):
                    t = str(r["ticker"]).strip()
                    td_sym = str(r.get("td_symbol", "")).strip()
                    y = str(r.get("yahoo_symbol", "")).strip()
                    status.write(f"Henter {i}/{len(work)}: {t}")

                    df, source_used = fetch_history(ticker=t, td_symbol=td_sym, yahoo_symbol=y, years=max(3, years))
                    if df.empty:
                        prog.progress(i / len(work))
                        continue

                    tech = latest_technical_snapshot(df)
                    if not tech:
                        prog.progress(i / len(work))
                        continue

                    if pd.notna(tech.get("last", np.nan)) and tech["last"] < min_price:
                        prog.progress(i / len(work))
                        continue

                    overview = yahoo_fetch_overview(y or t)

                    rows.append(
                        {
                            "Ticker": t,
                            "Navn": str(r.get("name", "")).strip(),
                            "Land": str(r.get("country", "")).strip(),
                            "Exchange": str(r.get("exchange", "")).strip(),
                            "Kilde": source_used,
                            "Pris": tech["last"],
                            "P/E": overview.get("pe"),
                            "MCap": overview.get("market_cap"),
                            "RSI": tech["rsi"],
                            "Trend": "✅" if tech["trend_up"] else "—",
                            "Signal": tech["action"],
                            "Score": tech["score"],
                            "Mom20%": tech["mom20"],
                            "Vol20%": tech["vol20"],
                            "DD3m%": tech["dd3m"],
                            "MACD": tech["macd_state"],
                            "RSI State": tech["rsi_state"],
                            "BB": tech["bb_state"],
                            "Risiko": tech["risk"],
                        }
                    )

                    prog.progress(i / len(work))

                status.empty()
                prog.empty()

                if not rows:
                    st.warning("Ingen tickers gav brugbar data.")
                else:
                    out = pd.DataFrame(rows).sort_values(["Score", "Mom20%"], ascending=[False, False]).reset_index(drop=True)
                    top = out.head(top_n)

                    display_df = top.copy()
                    for col in ["Pris", "P/E", "MCap", "RSI", "Score", "Mom20%", "Vol20%", "DD3m%"]:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(safe_display_value)

                    st.markdown(f"### Top {top_n}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    with st.expander("Vis hele screener-resultatet"):
                        full_df = out.copy()
                        for col in ["Pris", "P/E", "MCap", "RSI", "Score", "Mom20%", "Vol20%", "DD3m%"]:
                            if col in full_df.columns:
                                full_df[col] = full_df[col].apply(safe_display_value)
                        st.dataframe(full_df, use_container_width=True, hide_index=True)


# =========================================================
# TAB 3 PORTFOLIO
# =========================================================
with tab_portfolio:
    st.subheader("💼 Portefølje")

    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        p_ticker = st.text_input("Ticker", value="AAPL", key="pf_ticker")
    with c2:
        p_td = st.text_input("TD symbol (valgfri)", value="", key="pf_td")
    with c3:
        p_yahoo = st.text_input("Yahoo symbol (valgfri)", value="", key="pf_yahoo")
    with c4:
        p_shares = st.number_input("Antal", min_value=0.0, value=1.0, step=1.0, key="pf_shares")
    with c5:
        p_name = st.text_input("Navn", value="", key="pf_name")

    if st.button("➕ Tilføj til portefølje"):
        if p_ticker.strip():
            st.session_state["portfolio"].append(
                {
                    "ticker": p_ticker.strip(),
                    "td_symbol": p_td.strip(),
                    "yahoo_symbol": p_yahoo.strip(),
                    "shares": float(p_shares),
                    "name": p_name.strip(),
                }
            )
            st.rerun()

    if not st.session_state["portfolio"]:
        st.info("Porteføljen er tom.")
    else:
        pdf = pd.DataFrame(st.session_state["portfolio"])
        rows = []
        for _, r in pdf.iterrows():
            t = str(r.get("ticker", "")).strip()
            td_sym = str(r.get("td_symbol", "")).strip()
            y = str(r.get("yahoo_symbol", "")).strip()
            shares = float(r.get("shares", 0.0))
            name = str(r.get("name", "")).strip()

            dfx, src = fetch_history(ticker=t, td_symbol=td_sym, yahoo_symbol=y, years=max(3, years))
            tech = latest_technical_snapshot(dfx)
            overview = yahoo_fetch_overview(y or t)

            last = float(dfx["Close"].iloc[-1]) if not dfx.empty else np.nan
            value = shares * last if np.isfinite(last) else np.nan

            rows.append(
                {
                    "Ticker": t,
                    "Navn": name,
                    "Antal": shares,
                    "Seneste": safe_display_value(last),
                    "Værdi": safe_display_value(value),
                    "P/E": safe_display_value(overview.get("pe")),
                    "MCap": safe_display_value(overview.get("market_cap")),
                    "Kilde": src,
                    "Signal": tech.get("action", "—"),
                    "Score": safe_display_value(tech.get("score", np.nan)),
                    "RSI": safe_display_value(tech.get("rsi", np.nan)),
                    "Trend": "✅" if tech.get("trend_up") else "—",
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =========================================================
# TAB 4 THEMES
# =========================================================
with tab_themes:
    st.subheader("🧭 Temaanalyse (stærk version)")

    ranking_df = build_theme_rankings()
    st.markdown("### Tema-rangering")
    if ranking_df.empty:
        st.warning("Kunne ikke beregne tema-rangering endnu.")
    else:
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    selected_theme = st.selectbox("Vælg tema", list(THEMES.keys()), index=0, key="theme_pick")
    cfg = THEMES[selected_theme]
    proxy = cfg["proxy"]
    benchmark = cfg.get("benchmark", "SPY")

    proxy_df, _ = fetch_history(ticker=proxy, td_symbol=proxy, yahoo_symbol=proxy, years=10)
    bench_df, _ = fetch_history(ticker=benchmark, td_symbol=benchmark, yahoo_symbol=benchmark, years=10)

    if proxy_df.empty:
        st.error(f"Kunne ikke hente data for proxy {proxy}")
    else:
        yearly = build_theme_yearly_comparison(proxy_df, bench_df if not bench_df.empty else pd.DataFrame())
        proxy_index = cumulative_index(proxy_df)
        bench_index = cumulative_index(bench_df) if not bench_df.empty else pd.DataFrame()

        cagr = calc_cagr(proxy_df)
        max_dd = calc_drawdown(pd.to_numeric(proxy_df["Close"], errors="coerce"))
        beat_rate = np.nan
        if not yearly.empty and "Alpha %" in yearly.columns:
            beat_rate = float((yearly["Alpha %"] > 0).mean() * 100.0)

        # scores from ranking
        row = ranking_df[ranking_df["Tema"] == selected_theme]
        regime = row["Regime"].iloc[0] if not row.empty else "—"
        speed = row["Hastighed"].iloc[0] if not row.empty else "—"
        total_score = row["Samlet score"].iloc[0] if not row.empty else np.nan

        st.markdown(f"## {selected_theme}")
        st.write(cfg.get("description", ""))

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Proxy", proxy)
        m2.metric("CAGR 10 år", "—" if pd.isna(cagr) else f"{cagr:.2f}%")
        m3.metric("Max drawdown", "—" if pd.isna(max_dd) else f"{max_dd:.1f}%")
        m4.metric("Slår benchmark", "—" if pd.isna(beat_rate) else f"{beat_rate:.0f}% år")
        m5.metric("Score", "—" if pd.isna(total_score) else f"{total_score:.1f}")

        rs_1m = calc_relative_strength(proxy_df, bench_df, 30) if not bench_df.empty else np.nan
        rs_3m = calc_relative_strength(proxy_df, bench_df, 90) if not bench_df.empty else np.nan
        rs_6m = calc_relative_strength(proxy_df, bench_df, 180) if not bench_df.empty else np.nan

        driver_text = ", ".join(cfg.get("drivers", [])[:3])
        headwind_text = ", ".join(cfg.get("headwinds", [])[:3])

        st.markdown("### Hvad sker lige nu")
        st.info(
            f"{selected_theme} er aktuelt i regime '{regime}' med hastighed '{speed}'. "
            f"Relativ styrke mod benchmark er 1M={('—' if pd.isna(rs_1m) else f'{rs_1m*100:.1f}%')}, "
            f"3M={('—' if pd.isna(rs_3m) else f'{rs_3m*100:.1f}%')}, "
            f"6M={('—' if pd.isna(rs_6m) else f'{rs_6m*100:.1f}%')}. "
            f"De vigtigste aktuelle drivere er {driver_text}. "
            f"De vigtigste modvinde er {headwind_text}."
        )

        st.markdown("### De kommende år")
        st.info(
            f"De kommende år vil {selected_theme} især afhænge af: {driver_text}. "
            f"Væsentlige risici er: {headwind_text}. "
            f"Hvis makro og kapitalflow fortsætter i samme retning, er sandsynligheden højere for fortsat styrke; "
            f"ved regimeskifte kan acceleration og relative score falde hurtigt."
        )

        st.markdown("### Historik: år-for-år (10 år)")
        if yearly.empty:
            st.warning("Ingen år-for-år-data endnu.")
        else:
            show_yearly = yearly.copy()
            for col in ["Theme %", "Benchmark %", "Alpha %"]:
                if col in show_yearly.columns:
                    show_yearly[col] = show_yearly[col].round(2)
            st.dataframe(show_yearly, use_container_width=True, hide_index=True)

        st.markdown("### Kumulativ udvikling")
        if not proxy_index.empty:
            chart_df = proxy_index.rename(columns={"Index": selected_theme}).copy()
            if not bench_index.empty:
                chart_df = chart_df.merge(
                    bench_index.rename(columns={"Index": benchmark}),
                    on="Date",
                    how="left",
                )
            st.line_chart(chart_df.set_index("Date"))

        st.markdown("### Drivere og modvinde")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Positive drivere**")
            for x in cfg.get("drivers", []):
                st.write(f"- {x}")
        with d2:
            st.markdown("**Modvinde / risici**")
            for x in cfg.get("headwinds", []):
                st.write(f"- {x}")

        st.markdown("### Global markedskontekst")
        macro_rows = []
        for label, series_id in cfg.get("macro_series", {}).items():
            dfm = fetch_fred_series(series_id, limit_years=12)
            if dfm.empty:
                macro_rows.append({"Indikator": label, "Serie": series_id, "Seneste": np.nan, "1 år ændring %": np.nan})
                continue

            latest = float(dfm["value"].dropna().iloc[-1]) if not dfm["value"].dropna().empty else np.nan
            prev = np.nan
            if len(dfm.dropna(subset=["value"])) >= 13:
                prev = float(dfm.dropna(subset=["value"])["value"].iloc[-13])

            yoy = np.nan
            if pd.notna(latest) and pd.notna(prev) and prev != 0:
                yoy = (latest / prev - 1.0) * 100.0

            macro_rows.append(
                {
                    "Indikator": label,
                    "Serie": series_id,
                    "Seneste": latest,
                    "1 år ændring %": yoy,
                }
            )

        macro_df = pd.DataFrame(macro_rows)
        if not macro_df.empty:
            for col in ["Seneste", "1 år ændring %"]:
                if col in macro_df.columns:
                    macro_df[col] = macro_df[col].round(2)
            st.dataframe(macro_df, use_container_width=True, hide_index=True)

        st.markdown("### Hastighed i udviklingen")
        st.write(f"**Aktuel hastighed:** {speed}")
        st.write(f"**Regime:** {regime}")
        st.write(
            "Hastigheden er udledt af relativ styrke, acceleration i trend, EMA-hældning og ændring i MACD-dynamik."
        )


# =========================================================
# TAB 5 DATA
# =========================================================
with tab_data:
    st.subheader("🛠 Data")

    files = list_universes()
    if not files:
        st.info("Ingen univers-filer endnu.")
    else:
        rows = []
        for key, path in files.items():
            df = safe_read_csv(path)
            rows.append({"Univers": key, "Fil": path, "Rækker": len(df)})
        st.dataframe(pd.DataFrame(rows).sort_values("Univers"), use_container_width=True, hide_index=True)

    if st.button("Slet signals_log.csv"):
        if os.path.exists(SIGNAL_LOG):
            os.remove(SIGNAL_LOG)
        st.success("Signal-log slettet.")
        st.rerun()


st.caption("Primær datakilde: Twelve Data. Fallback og nøgletal: Yahoo Finance. FRED bruges til makro, hvis tilgængelig. Ikke finansiel rådgivning.")