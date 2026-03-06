import os
import json
from io import StringIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Stock Dashboard (Twelve Data + Yahoo fallback)",
    layout="wide",
    page_icon="📊",
)

APP_TITLE = "📊 Stock Dashboard (Twelve Data + Yahoo fallback)"
DATA_DIR = "data"
UNIVERSE_DIR = os.path.join(DATA_DIR, "universes")
SIGNAL_LOG = os.path.join(DATA_DIR, "signals_log.csv")
DEFAULT_TOPN = 10
HTTP_TIMEOUT = 25

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UNIVERSE_DIR, exist_ok=True)

TD_API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", os.getenv("TWELVE_DATA_API_KEY", "")).strip()
TD_BASE = "https://api.twelvedata.com"

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}

TD_EXCHANGES = {
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
    "Portugal": ["XLIS"],
    "Ireland": ["XDUB"],
    "Canada": ["XTSE", "XTSX"],
    "Japan": ["XTKS"],
    "Hong Kong": ["XHKG"],
    "India": ["XBOM", "XNSE"],
    "Brazil": ["B3"],
}

THEMES = {
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


# =========================================================
# HELPERS
# =========================================================
def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
    return requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers=headers)


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

    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"]].reset_index(drop=True)


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


def get_twelve_api_key() -> str:
    return TD_API_KEY


def google_news_link(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/search?q={q}&hl=da&gl=DK&ceid=DK%3Ada"


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

    dedup = []
    seen = set()
    for x in out:
        if x not in seen and x.strip():
            dedup.append(x)
            seen.add(x)
    return dedup


# =========================================================
# TWELVE DATA
# =========================================================
def td_get(endpoint: str, params: Optional[dict] = None) -> dict:
    api_key = get_twelve_api_key()
    if not api_key:
        return {"status": "error", "message": "Missing TWELVE_DATA_API_KEY"}

    p = dict(params or {})
    p["apikey"] = api_key

    try:
        r = http_get(f"{TD_BASE}/{endpoint}", params=p)
        if r.status_code != 200:
            return {"status": "error", "message": f"HTTP {r.status_code}"}
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def td_fetch_history(symbol: str, years: int = 5) -> pd.DataFrame:
    years = max(1, int(years))
    outputsize = min(5000, max(400, years * 260))

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

    rename_map = {
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return df


def td_fetch_quote(symbol: str) -> dict:
    return td_get("quote", {"symbol": symbol})


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def td_fetch_stocks(exchange: str = "", country: str = "", limit: int = 5000) -> pd.DataFrame:
    params = {"format": "JSON"}
    if exchange:
        params["exchange"] = exchange
    if country:
        params["country"] = country

    data = td_get("stocks", params)
    rows = []

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            rows = data["data"]
        elif "values" in data and isinstance(data["values"], list):
            rows = data["values"]
        elif isinstance(data.get("meta"), list):
            rows = data["meta"]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if limit and len(df) > limit:
        df = df.head(limit)
    return df


# =========================================================
# YAHOO FALLBACK
# =========================================================
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def yahoo_fetch_history(symbol: str, years: int = 5) -> pd.DataFrame:
    range_str = "10y" if years >= 10 else f"{max(1, int(years))}y"
    candidates = yahoo_symbol_candidates(symbol)

    for sym in candidates:
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
            if not timestamps or not quote:
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


# =========================================================
# PRICE FETCHER: TWELVE DATA FIRST, YAHOO SECOND
# =========================================================
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_history(symbol: str, yahoo_symbol: str = "", years: int = 5) -> Tuple[pd.DataFrame, str]:
    df = td_fetch_history(symbol, years=years)
    if not df.empty:
        return df, "Twelve Data"

    candidate = yahoo_symbol.strip() or symbol
    df2 = yahoo_fetch_history(candidate, years=years)
    if not df2.empty:
        return df2, "Yahoo"

    if yahoo_symbol.strip() and yahoo_symbol.strip() != symbol:
        df3 = yahoo_fetch_history(symbol, years=years)
        if not df3.empty:
            return df3, "Yahoo"

    return pd.DataFrame(), ""


# =========================================================
# UNIVERSE BUILDER
# =========================================================
def make_yahoo_symbol_from_row(row: pd.Series) -> str:
    ticker = str(row.get("symbol", "") or row.get("ticker", "")).strip().upper()
    exchange = str(row.get("exchange", "")).strip().upper()

    if not ticker:
        return ""

    mapping = {
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
    }

    if exchange in mapping:
        return f"{ticker}{mapping[exchange]}"
    return ticker


def build_universe_for_country(country_name: str, exchanges: List[str], per_exchange_limit: int = 3000) -> pd.DataFrame:
    frames = []

    for ex in exchanges:
        df = td_fetch_stocks(exchange=ex, country=country_name, limit=per_exchange_limit)
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

        if "exchange" not in df.columns:
            df["exchange"] = ex
        if "country" not in df.columns:
            df["country"] = country_name
        if "type" not in df.columns:
            df["type"] = ""

        df["source"] = "Twelve Data"
        df["yahoo_symbol"] = df.apply(make_yahoo_symbol_from_row, axis=1)

        keep = ["symbol", "name", "country", "exchange", "type", "source", "yahoo_symbol"]
        frames.append(df[keep].rename(columns={"symbol": "ticker"}))

    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"])

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
        global_df = pd.DataFrame(columns=["ticker", "name", "country", "exchange", "type", "source", "yahoo_symbol"])

    global_df = ensure_universe_schema(global_df)
    global_df.to_csv(universe_file("global_all"), index=False, encoding="utf-8")
    counts["global_all"] = len(global_df)

    us_df = safe_read_csv(universe_file("usa"))
    if not us_df.empty:
        us_df = ensure_universe_schema(us_df)
        us_df.to_csv(universe_file("us_all"), index=False, encoding="utf-8")
        counts["us_all"] = len(us_df)

    return counts


def load_universe(key: str) -> Tuple[pd.DataFrame, str]:
    path = list_universes().get(key, "")
    if not path:
        return pd.DataFrame(), "Ukendt univers."

    df = safe_read_csv(path)
    df = ensure_universe_schema(df)
    if df.empty:
        return df, f"Tomt univers: {path}"
    return df, ""


# =========================================================
# INDICATORS
# =========================================================
def pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return (a / b - 1.0) * 100.0


def rsi(close: pd.Series, period: int = 14) -> float:
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < period + 5:
        return float("nan")

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    loss = loss.replace(0, np.nan)

    rs = gain / loss
    out = 100 - (100 / (1 + rs))
    if out.empty or pd.isna(out.iloc[-1]):
        return float("nan")
    return float(out.iloc[-1])


def compute_signals(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 80:
        return {}

    last = float(close.iloc[-1])
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    if len(close) >= 200 and not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        trend_up = bool(ma50.iloc[-1] > ma200.iloc[-1])
    elif len(close) >= 50 and not pd.isna(ma50.iloc[-1]):
        trend_up = bool(last > ma50.iloc[-1])
    else:
        trend_up = False

    rsi14 = rsi(close, 14)
    mom20 = pct_change(last, float(close.iloc[-21])) if len(close) >= 21 else float("nan")
    ret = close.pct_change().dropna()
    vol20 = float(ret.rolling(20).std().iloc[-1] * 100.0) if len(ret) >= 25 else float("nan")

    dd = float("nan")
    if len(close) >= 63:
        peak = float(close.iloc[-63:].max())
        if peak != 0:
            dd = (last / peak - 1.0) * 100.0

    score = 0.0
    score += 2.0 if trend_up else 0.0
    if not np.isnan(rsi14):
        score += max(0.0, 2.0 - abs(rsi14 - 50) / 25)
    if not np.isnan(mom20):
        score += max(0.0, min(3.0, mom20 / 5.0))
    if not np.isnan(dd):
        score += max(0.0, min(2.0, (10.0 + dd) / 10.0))

    risk = "OK"
    if (not np.isnan(vol20) and vol20 > 4.5) or (not np.isnan(dd) and dd < -15):
        risk = "Høj"
    if not np.isnan(dd) and dd < -25:
        risk = "Meget høj"

    why = []
    if trend_up:
        why.append("Trend op")
    if not np.isnan(rsi14):
        why.append(f"RSI {rsi14:.0f}")
    if not np.isnan(mom20):
        why.append(f"Momentum 20d {mom20:.1f}%")
    if not np.isnan(vol20):
        why.append(f"Vol20 {vol20:.1f}%")

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
        "rsi": rsi14,
        "mom20": mom20,
        "vol20": vol20,
        "dd3m": dd,
        "trend_up": trend_up,
        "risk": risk,
        "score": round(score, 2),
        "action": action,
        "why": " • ".join(why),
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

    out = {}
    out["1D"] = pct_change(last_close, float(d["Close"].iloc[-2])) if len(d) >= 2 else float("nan")
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
        return float("nan")

    a = a[["Date", "Close"]].dropna().copy().sort_values("Date")
    b = b[["Date", "Close"]].dropna().copy().sort_values("Date")

    end = min(a["Date"].iloc[-1], b["Date"].iloc[-1])
    start = end - pd.Timedelta(days=days)

    def close_on(df_local: pd.DataFrame, d: pd.Timestamp) -> float:
        sub = df_local[df_local["Date"] <= d]
        if sub.empty:
            return float("nan")
        return float(sub["Close"].iloc[-1])

    a0 = close_on(a, start)
    a1 = close_on(a, end)
    b0 = close_on(b, start)
    b1 = close_on(b, end)

    vals = [a0, a1, b0, b1]
    if any(pd.isna(x) for x in vals) or a0 == 0 or b0 == 0:
        return float("nan")

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
    max_screen = st.slider("Max tickers pr. screening", 20, 1000, 200, 10)

    st.divider()
    st.subheader("🔑 Datakilde")
    if TD_API_KEY:
        st.success("Twelve Data key fundet")
    else:
        st.warning("Ingen Twelve Data key fundet. Yahoo fallback kan stadig bruges for manuelle tickers.")

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
- **Søg & analyse**: vælg papir og se kurs, signal og nyheder
- **Screening**: kør Top N på valgt univers
- **Portefølje**: tilføj beholdninger og få signaler
- **Tema**: momentum-proxy
        """
    )

tab_search, tab_screener, tab_portfolio, tab_themes, tab_data = st.tabs(
    ["🔎 Søg & analyse", "🏁 Screening", "💼 Portefølje", "🧭 Tema", "🛠 Data"]
)


# =========================================================
# TAB 1
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

                    st.caption(f"Ticker: **{ticker}**")
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
                df, source_used = fetch_history(ticker, yahoo_symbol=yahoo_symbol, years=years)

                if df.empty:
                    st.error("Kunne ikke hente kursdata fra Twelve Data eller Yahoo fallback.")
                else:
                    sig = compute_signals(df)
                    rets = period_returns(df)

                    last = float(df["Close"].iloc[-1])
                    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
                    chg = (last / prev - 1.0) * 100 if prev else 0.0

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Seneste close", f"{last:,.2f}")
                    m2.metric("Dag %", f"{chg:+.2f}%")
                    m3.metric("Seneste dato", df["Date"].iloc[-1].date().isoformat())
                    m4.metric("Signal", sig.get("action", "—"))
                    m5.metric("Kilde", source_used or "—")

                    if sig:
                        append_signal_log(
                            ticker=ticker,
                            action=str(sig.get("action", "")),
                            score=float(sig.get("score", np.nan)),
                            last=float(sig.get("last", np.nan)),
                        )

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

                    st.markdown("#### OHLC (seneste 10)")
                    st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

                    st.markdown("#### Signal")
                    st.write(sig.get("why", "—"))
                    st.caption(f"Risiko: {sig.get('risk', '—')} | Score: {sig.get('score', '—')}")

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
# TAB 2
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
            if st.button("Kør screening", type="primary"):
                tickers = uni.head(max_screen).copy()

                rows = []
                prog = st.progress(0)
                status = st.empty()

                for i, (_, r) in enumerate(tickers.iterrows(), start=1):
                    t = str(r["ticker"]).strip()
                    y = str(r.get("yahoo_symbol", "")).strip()
                    status.write(f"Henter {i}/{len(tickers)}: {t}")

                    df, source_used = fetch_history(t, yahoo_symbol=y, years=max(3, years))
                    sig = compute_signals(df)

                    if sig:
                        rows.append(
                            {
                                "Ticker": t,
                                "Navn": str(r.get("name", "")).strip(),
                                "Land": str(r.get("country", "")).strip(),
                                "Exchange": str(r.get("exchange", "")).strip(),
                                "Kilde": source_used,
                                "Score": sig["score"],
                                "Signal": sig["action"],
                                "Trend": "✅" if sig["trend_up"] else "—",
                                "RSI": round(sig["rsi"], 1) if not np.isnan(sig["rsi"]) else np.nan,
                                "Mom20%": round(sig["mom20"], 1) if not np.isnan(sig["mom20"]) else np.nan,
                                "Vol20%": round(sig["vol20"], 2) if not np.isnan(sig["vol20"]) else np.nan,
                                "DD3m%": round(sig["dd3m"], 1) if not np.isnan(sig["dd3m"]) else np.nan,
                                "Seneste": round(sig["last"], 2),
                                "Risiko": sig["risk"],
                                "Hvorfor": sig["why"],
                            }
                        )

                    prog.progress(i / len(tickers))

                status.empty()
                prog.empty()

                if not rows:
                    st.warning("Ingen tickers gav brugbar data.")
                else:
                    out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
                    top = out.head(top_n)

                    st.markdown(f"### Top {top_n}")
                    st.dataframe(top, use_container_width=True, hide_index=True)

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
                            st.line_chart(dfx.set_index("Date")["Close"])
                            st.caption(f"Kilde: {src}")
                            st.markdown(f"[Nyheder]({google_news_link(pick)})")


# =========================================================
# TAB 3
# =========================================================
def portfolio_to_df() -> pd.DataFrame:
    if not st.session_state["portfolio"]:
        return pd.DataFrame(columns=["ticker", "shares", "name", "yahoo_symbol"])
    df = pd.DataFrame(st.session_state["portfolio"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"] = df.get("name", "").astype(str)
    df["yahoo_symbol"] = df.get("yahoo_symbol", "").astype(str)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
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
                sig = compute_signals(dfx)

                last = float(dfx["Close"].iloc[-1]) if not dfx.empty else np.nan
                value = shares * last if np.isfinite(last) else np.nan

                rows.append(
                    {
                        "Ticker": t,
                        "Navn": name,
                        "Antal": shares,
                        "Seneste": round(last, 4) if np.isfinite(last) else np.nan,
                        "Værdi": round(value, 2) if np.isfinite(value) else np.nan,
                        "Kilde": src,
                        "Signal": sig.get("action", "—"),
                        "Score": sig.get("score", np.nan),
                        "RSI": round(sig.get("rsi", np.nan), 1) if sig else np.nan,
                        "Mom20%": round(sig.get("mom20", np.nan), 1) if sig else np.nan,
                        "Risiko": sig.get("risk", "—"),
                        "Forklaring": sig.get("why", ""),
                        "Nyheder": google_news_link(f"{t} {name}".strip()),
                    }
                )

        out = pd.DataFrame(rows)
        total = float(out["Værdi"].sum()) if "Værdi" in out.columns else 0.0
        out["Vægt %"] = (out["Værdi"] / total * 100.0).round(2) if total > 0 else np.nan
        st.dataframe(out.sort_values("Vægt %", ascending=False), use_container_width=True, hide_index=True)


# =========================================================
# TAB 4
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
                    "MomentumScore": round(score, 4),
                    "RS_1M_vs_SPY": round(rs_1m, 4) if not np.isnan(rs_1m) else np.nan,
                    "RS_3M_vs_SPY": round(rs_3m, 4) if not np.isnan(rs_3m) else np.nan,
                }
            )

    dfm = pd.DataFrame(rows).sort_values("MomentumScore", ascending=False).reset_index(drop=True)
    st.dataframe(dfm, use_container_width=True, hide_index=True)

    st.markdown("### Top temaer")
    for _, r in dfm.head(10).iterrows():
        rs1 = r["RS_1M_vs_SPY"]
        rs3 = r["RS_3M_vs_SPY"]
        rs1_txt = "—" if pd.isna(rs1) else f"{rs1:+.2%}"
        rs3_txt = "—" if pd.isna(rs3) else f"{rs3:+.2%}"
        st.markdown(f"- **{r['Tema']}** ({r['Proxy']}) — RS 1M: {rs1_txt}, RS 3M: {rs3_txt}")


# =========================================================
# TAB 5
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


st.caption("Primær datakilde: Twelve Data. Fallback: Yahoo Finance. Ikke finansiel rådgivning.")
