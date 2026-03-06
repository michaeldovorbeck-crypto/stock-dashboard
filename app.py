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

THEMES: Dict[str, List[str]] = {
    "AI & Software": ["QQQ", "XLK", "MSFT", "NVDA"],
    "Semiconductors": ["SOXX", "SMH", "NVDA", "AVGO"],
    "Cybersecurity": ["HACK", "CIBR", "PANW", "CRWD"],
    "Defense/Aerospace": ["ITA", "XAR", "LMT", "NOC"],
    "Cloud/Datacenter": ["SKYY", "AMZN", "GOOGL"],
    "Solar": ["TAN", "ENPH", "FSLR"],
    "Clean Energy": ["ICLN", "PBW"],
    "Uranium": ["URA", "CCJ"],
    "EV & Batteries": ["LIT", "TSLA", "ALB"],
    "Healthcare": ["XLV", "UNH", "JNJ"],
    "Biotech": ["IBB", "XBI"],
    "Banks": ["XLF", "JPM", "BAC"],
    "Japan": ["EWJ"],
    "Emerging Markets": ["EEM", "VWO"],
    "Gold": ["GLD", "IAU"],
    "Utilities": ["XLU"],
    "Momentum": ["MTUM"],
    "Small Cap": ["IWM"],
    "Growth": ["VUG"],
    "Value": ["VTV"],
}
THEME_BENCHMARK = "SPY"

YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}


# =========================================================
# SECRETS / API KEY
# =========================================================
def get_twelve_data_api_key() -> str:
    try:
        return str(st.secrets["TWELVE_DATA_API_KEY"]).strip()
    except Exception:
        return os.getenv("TWELVE_DATA_API_KEY", "").strip()


TD_API_KEY = get_twelve_data_api_key()


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


def ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
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


def list_universes() -> Dict[str, str]:
    out: Dict[str, str] = {}
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


def google_news_link(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/search?q={q}&hl=da&gl=DK&ceid=DK%3Ada"


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
def td_fetch_history(symbol: str, years: int = 5) -> pd.DataFrame:
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

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(ttl=60 * 60, show_spinner=False)
def td_fetch_quote(symbol: str) -> dict:
    return td_get("quote", {"symbol": symbol})


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def td_fetch_stocks(exchange: str = "", country: str = "") -> pd.DataFrame:
    payload: Dict[str, str] = {"format": "JSON"}
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
# YAHOO FALLBACK
# =========================================================
def yahoo_symbol_candidates(symbol: str) -> List[str]:
    s = (symbol or "").strip().upper()
    if not s:
        return []

    out = [s]

    replacements = {
        ".XCSE": ".CO",
        ".XSTO": ".ST",
        ".XHEL": ".HE",
        ".XOSL": ".OL",
        ".XPAR": ".PA",
        ".XAMS": ".AS",
        ".XBRU": ".BR",
        ".XLON": ".L",
        ".XMIL": ".MI",
        ".XMAD": ".MC",
        ".XSWX": ".SW",
        ".XTKS": ".T",
        ".XHKG": ".HK",
        ".XNSE": ".NS",
        ".XBOM": ".BO",
        ".XTSE": ".TO",
        ".XTSX": ".V",
        ".XETRA": ".DE",
        ".FWB": ".DE",
        ":XCSE": ".CO",
        ":XSTO": ".ST",
        ":XHEL": ".HE",
        ":XOSL": ".OL",
        ":XPAR": ".PA",
        ":XAMS": ".AS",
        ":XBRU": ".BR",
        ":XLON": ".L",
        ":XMIL": ".MI",
        ":XMAD": ".MC",
        ":XSWX": ".SW",
        ":XTKS": ".T",
        ":XHKG": ".HK",
        ":XNSE": ".NS",
        ":XBOM": ".BO",
        ":XTSE": ".TO",
        ":XTSX": ".V",
        ":XETRA": ".DE",
        ":FWB": ".DE",
    }

    for old, new in replacements.items():
        if old in s:
            out.append(s.replace(old, new))

    if ":" in s:
        left, right = s.split(":", 1)
        out.append(left)
        out.append(right)

    seen = set()
    dedup = []
    for item in out:
        if item and item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def yahoo_fetch_history(symbol: str, years: int = 5) -> pd.DataFrame:
    range_str = "10y" if years >= 10 else f"{max(1, int(years))}y"

    for sym in yahoo_symbol_candidates(symbol):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
            r = http_get(
                url,
                params={
                    "interval": "1d",
                    "range": range_str,
                    "includeAdjustedClose": "true",
                },
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
                "currency": raw("price", "currency"),
            }
        except Exception:
            continue

    return {}


# =========================================================
# PRIMARY FETCHER
# =========================================================
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_history(symbol: str, yahoo_symbol: str = "", years: int = 5) -> Tuple[pd.DataFrame, str]:
    df = td_fetch_history(symbol, years=years)
    if not df.empty:
        return df, "Twelve Data"

    if yahoo_symbol.strip():
        df2 = yahoo_fetch_history(yahoo_symbol, years=years)
        if not df2.empty:
            return df2, "Yahoo"

    df3 = yahoo_fetch_history(symbol, years=years)
    if not df3.empty:
        return df3, "Yahoo"

    return pd.DataFrame(), ""


# =========================================================
# UNIVERSE BUILDER
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

    if ex in suffix:
        return f"{t}{suffix[ex]}"
    return t


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
        df["yahoo_symbol"] = df.apply(
            lambda r: make_yahoo_symbol(r.get("symbol", ""), r.get("exchange", ex)),
            axis=1,
        )

        keep = ["symbol", "name", "country", "exchange", "type", "source", "yahoo_symbol"]
        frames.append(df[keep].rename(columns={"symbol": "ticker"}))

    if not frames:
        return pd.DataFrame(
            columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"]
        )

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "exchange"])
    return ensure_universe_schema(out)


def build_all_universes(selected_countries: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
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
        global_df = pd.DataFrame(
            columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"]
        )

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
# TECHNICAL ANALYSIS
# =========================================================
def pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
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

    d["VOL20_AVG"] = pd.to_numeric(d.get("Volume", np.nan), errors="coerce").rolling(20).mean()
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
            return float("nan")
        return float(sub["Close"].iloc[-1])

    out: Dict[str, float] = {}
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
# SIGNAL LOG
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
            if not old.empty and {"ticker", "action", "last"}.issubset(old.columns):
                prev = old.iloc[-1].to_dict()
                prev_last = pd.to_numeric(prev.get("last", np.nan), errors="coerce")
                if (
                    str(prev.get("ticker", "")) == ticker
                    and str(prev.get("action", "")) == action
                    and float(prev_last) == float(last)
                ):
                    return
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
# THEME MOMENTUM
# =========================================================
def relative_strength(symbol: str, benchmark: str, days: int) -> float:
    a, _ = fetch_history(symbol, years=5)
    b, _ = fetch_history(benchmark, years=5)

    if a.empty or b.empty:
        return np.nan

    a = a[["Date", "Close"]].dropna().copy().sort_values("Date")
    b = b[["Date", "Close"]].dropna().copy().sort_values("Date")

    end = min(a["Date"].iloc[-1], b["Date"].iloc[-1])
    start = end - pd.Timedelta(days=days)

    def close_on(df_local: pd.DataFrame, d: pd.Timestamp) -> float:
        sub = df_local[df_local["Date"] <= d]
        if sub.empty:
            return np.nan
        return float(sub["Close"].iloc[-1])

    a0 = close_on(a, start)
    a1 = close_on(a, end)
    b0 = close_on(b, start)
    b1 = close_on(b, end)

    vals = [a0, a1, b0, b1]
    if any(pd.isna(x) for x in vals) or a0 == 0 or b0 == 0:
        return np.nan

    return (a1 / a0 - 1.0) - (b1 / b0 - 1.0)


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
        st.success("Twelve Data key fundet")
    else:
        st.warning("Ingen Twelve Data key fundet. Yahoo fallback virker stadig ved gyldige symboler.")

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
- **Tema**: momentum-proxy
        """
    )

tab_search, tab_screener, tab_portfolio, tab_themes, tab_data = st.tabs(
    ["🔎 Søg & analyse", "🏁 Screening", "💼 Portefølje", "🧭 Tema", "🛠 Data"]
)


# =========================================================
# TAB 1: SEARCH
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
            yahoo_symbol = ""
            name = ""
            country = ""
            exchange = ""
            type_ = ""

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
                    yahoo_symbol = str(row.get("yahoo_symbol", "")).strip()
                    name = str(row.get("name", "")).strip()
                    country = str(row.get("country", "")).strip()
                    exchange = str(row.get("exchange", "")).strip()
                    type_ = str(row.get("type", "")).strip()

                    st.caption(f"Ticker: **{ticker}**")
                    if yahoo_symbol:
                        st.caption(f"Yahoo fallback: {yahoo_symbol}")
                    if name:
                        st.caption(f"Navn: {name}")
                    if country:
                        st.caption(f"Land: {country}")
                    if exchange:
                        st.caption(f"Exchange: {exchange}")
                    if type_:
                        st.caption(f"Type: {type_}")

        with right:
            if ticker:
                df, source_used = fetch_history(ticker, yahoo_symbol=yahoo_symbol, years=years)
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
                    tech_df = add_technical_columns(df).copy()

                    st.markdown("##### Pris vs EMA")
                    tech_chart = tech_df.set_index("Date")[["Close", "EMA20", "EMA50", "EMA200"]].dropna(how="all")
                    st.line_chart(tech_chart)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("##### RSI14")
                        rsi_chart = tech_df.set_index("Date")[["RSI14"]].dropna()
                        st.line_chart(rsi_chart)
                    with c2:
                        st.markdown("##### MACD")
                        macd_chart = tech_df.set_index("Date")[["MACD", "MACD_SIGNAL"]].dropna()
                        st.line_chart(macd_chart)

                    st.markdown("##### Bollinger Bands")
                    bb_chart = tech_df.set_index("Date")[["Close", "BB_UPPER", "BB_MID", "BB_LOWER"]].dropna(how="all")
                    st.line_chart(bb_chart)

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

                    qtxt = f"{ticker} {name}".strip()
                    st.markdown("#### Nyheder")
                    st.markdown(f"[Google News]({google_news_link(qtxt)})")

                    with st.expander("Signal-log"):
                        hist = read_signal_log(ticker)
                        if hist.empty:
                            st.info("Ingen log endnu.")
                        else:
                            st.dataframe(hist, use_container_width=True, hide_index=True)


# =========================================================
# TAB 2: SCREENER
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
                    y = str(r.get("yahoo_symbol", "")).strip()
                    status.write(f"Henter {i}/{len(work)}: {t}")

                    df, source_used = fetch_history(t, yahoo_symbol=y, years=max(3, years))
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

                    choices = top.apply(
                        lambda rr: f"{rr['Ticker']} — {rr['Navn']}" if str(rr["Navn"]).strip() else rr["Ticker"],
                        axis=1,
                    ).tolist()

                    if choices:
                        pick = st.selectbox("Vælg kandidat", choices, key="screen_pick")
                        pick_ticker = pick.split(" — ")[0].strip()

                        meta = uni[uni["ticker"] == pick_ticker]
                        ysym = str(meta["yahoo_symbol"].iloc[0]) if not meta.empty else ""

                        dfx, src = fetch_history(pick_ticker, yahoo_symbol=ysym, years=years)
                        if not dfx.empty:
                            tech_df = add_technical_columns(dfx)
                            st.line_chart(tech_df.set_index("Date")[["Close", "EMA20", "EMA50", "EMA200"]].dropna(how="all"))
                            st.caption(f"Kilde: {src}")
                            st.markdown(f"[Nyheder]({google_news_link(pick)})")


# =========================================================
# TAB 3: PORTFOLIO
# =========================================================
def portfolio_to_df() -> pd.DataFrame:
    if not st.session_state["portfolio"]:
        return pd.DataFrame(columns=["ticker", "shares", "name", "yahoo_symbol"])
    df = pd.DataFrame(st.session_state["portfolio"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df["name"] = df.get("name", "").astype(str)
    df["yahoo_symbol"] = df.get("yahoo_symbol", "").astype(str)
    df = df[df["ticker"].str.len() > 0]
    df = df[df["shares"] > 0]
    return df.reset_index(drop=True)


with tab_portfolio:
    st.subheader("💼 Portefølje")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        p_ticker = st.text_input("Ticker", value="AAPL", key="pf_ticker")
    with c2:
        p_yahoo = st.text_input("Yahoo symbol (valgfri)", value="", key="pf_yahoo")
    with c3:
        p_shares = st.number_input("Antal", min_value=0.0, value=1.0, step=1.0, key="pf_shares")
    with c4:
        p_name = st.text_input("Navn", value="", key="pf_name")

    if st.button("➕ Tilføj til portefølje"):
        if p_ticker.strip():
            st.session_state["portfolio"].append(
                {
                    "ticker": p_ticker.strip(),
                    "yahoo_symbol": p_yahoo.strip(),
                    "shares": float(p_shares),
                    "name": p_name.strip(),
                }
            )
            st.rerun()

    dfp = portfolio_to_df()

    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Beholdninger")
        if dfp.empty:
            st.info("Porteføljen er tom.")
        else:
            st.dataframe(dfp, use_container_width=True, hide_index=True)

    with right:
        export_json = json.dumps(st.session_state["portfolio"], ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ Download portfolio.json",
            data=export_json,
            file_name="portfolio.json",
            mime="application/json",
        )

        up = st.file_uploader("Upload portfolio.json", type=["json"], key="pf_upload")
        if up is not None:
            try:
                loaded = json.loads(up.read().decode("utf-8"))
                if isinstance(loaded, list):
                    st.session_state["portfolio"] = loaded
                    st.success("Importeret.")
                    st.rerun()
                else:
                    st.error("JSON skal være en liste.")
            except Exception as e:
                st.error(f"Kunne ikke læse JSON: {e}")

    st.markdown("### Analyse pr. holding")
    if dfp.empty:
        st.info("Tilføj mindst én holding.")
    else:
        rows = []
        with st.spinner("Henter data og beregner signaler ..."):
            for _, r in dfp.iterrows():
                t = str(r["ticker"]).strip()
                y = str(r.get("yahoo_symbol", "")).strip()
                shares = float(r["shares"])
                name = str(r.get("name", "")).strip()

                dfx, src = fetch_history(t, yahoo_symbol=y, years=max(3, years))
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
                        "Mom20%": safe_display_value(tech.get("mom20", np.nan)),
                        "Trend": "✅" if tech.get("trend_up") else "—",
                        "MACD": tech.get("macd_state", "—"),
                        "Risiko": tech.get("risk", "—"),
                        "Forklaring": tech.get("action", ""),
                        "Nyheder": google_news_link(f"{t} {name}".strip()),
                    }
                )

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True, hide_index=True)


# =========================================================
# TAB 4: THEMES
# =========================================================
with tab_themes:
    st.subheader("🧭 Tema-momentum")

    rows = []
    with st.spinner("Beregner temaer ..."):
        for theme, tickers in THEMES.items():
            proxy = tickers[0]
            rs_1m = relative_strength(proxy, THEME_BENCHMARK, 30)
            rs_3m = relative_strength(proxy, THEME_BENCHMARK, 90)

            score = 0.0
            if not np.isnan(rs_1m):
                score += rs_1m * 100
            if not np.isnan(rs_3m):
                score += rs_3m * 50

            rows.append(
                {
                    "Tema": theme,
                    "Proxy": proxy,
                    "MomentumScore": safe_display_value(round(score, 4)),
                    "RS_1M_vs_SPY": safe_display_value(rs_1m),
                    "RS_3M_vs_SPY": safe_display_value(rs_3m),
                }
            )

    dfm = pd.DataFrame(rows)
    st.dataframe(dfm, use_container_width=True, hide_index=True)

    st.markdown("### Top temaer")
    for _, r in dfm.head(10).iterrows():
        st.markdown(f"- **{r['Tema']}** ({r['Proxy']}) — RS 1M: {r['RS_1M_vs_SPY']}, RS 3M: {r['RS_3M_vs_SPY']}")


# =========================================================
# TAB 5: DATA
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

    st.markdown("### Ryd signal-log")
    if st.button("Slet signals_log.csv"):
        if os.path.exists(SIGNAL_LOG):
            os.remove(SIGNAL_LOG)
        st.success("Signal-log slettet.")
        st.rerun()


st.caption("Primær datakilde: Twelve Data. Fallback og nøgletal: Yahoo Finance. Ikke finansiel rådgivning.")