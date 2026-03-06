import os
import re
import json
from io import StringIO
from urllib.parse import urljoin, quote
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Stock Dashboard (Global, gratis)",
    layout="wide",
    page_icon="📊",
)

APP_TITLE = "📊 Stock Dashboard (Global + Screener + Portefølje) — gratis"
DATA_DIR = "data"
UNIVERSE_DIR = os.path.join(DATA_DIR, "universes")
SIGNAL_LOG = os.path.join(DATA_DIR, "signals_log.csv")
DEFAULT_TOPN = 10
HTTP_TIMEOUT = 25

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UNIVERSE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; stock-dashboard/1.0; +https://stooq.com/)"
}

# Stooq group pages discovered from their market pages / linked stock groups.
# This is the practical "worldwide as possible" starter set for automatic universes.
STOOQ_GROUPS = {
    "us_all": {
        "label": "US_ALL",
        "url": "https://stooq.com/t/?i=518",
        "country": "US",
    },
    "uk_all": {
        "label": "UK_ALL",
        "url": "https://stooq.com/t/?i=610",
        "country": "UK",
    },
    "japan_all": {
        "label": "JAPAN_ALL",
        "url": "https://stooq.com/t/?i=519",
        "country": "JP",
    },
    "germany_all": {
        "label": "GERMANY_ALL",
        "url": "https://stooq.com/t/?i=521",
        "country": "DE",
    },
    "hongkong_all": {
        "label": "HONGKONG_ALL",
        "url": "https://stooq.com/t/?i=614",
        "country": "HK",
    },
    "poland_all": {
        "label": "POLAND_ALL",
        "url": "https://stooq.com/t/?i=523",
        "country": "PL",
    },
    "hungary_all": {
        "label": "HUNGARY_ALL",
        "url": "https://stooq.com/t/?i=522",
        "country": "HU",
    },
    "us_etfs": {
        "label": "US_ETFS",
        "url": "https://stooq.com/t/?i=609",
        "country": "US",
    },
}

THEMES = {
    "AI & Software": ["QQQ.US", "XLK.US", "MSFT.US", "NVDA.US"],
    "Semiconductors": ["SOXX.US", "SMH.US", "NVDA.US", "AVGO.US"],
    "Cybersecurity": ["HACK.US", "CIBR.US", "PANW.US", "CRWD.US"],
    "Defense/Aerospace": ["ITA.US", "XAR.US", "LMT.US", "NOC.US"],
    "Cloud/Datacenter": ["SKYY.US", "AMZN.US", "GOOGL.US"],
    "Solar": ["TAN.US", "ENPH.US", "FSLR.US"],
    "Clean Energy": ["ICLN.US", "PBW.US"],
    "Uranium": ["URA.US", "CCJ.US"],
    "EV & Batteries": ["LIT.US", "TSLA.US", "ALB.US"],
    "Healthcare": ["XLV.US", "UNH.US", "JNJ.US"],
    "Biotech": ["IBB.US", "XBI.US"],
    "Banks": ["XLF.US", "JPM.US", "BAC.US"],
    "Japan": ["EWJ.US"],
    "Emerging Markets": ["EEM.US", "VWO.US"],
    "Gold": ["GLD.US", "IAU.US"],
    "Utilities": ["XLU.US"],
    "Momentum": ["MTUM.US"],
    "Small Cap": ["IWM.US"],
    "Growth": ["VUG.US"],
    "Value": ["VTV.US"],
}
THEME_BENCHMARK = "SPY.US"


# =========================================================
# LOW-LEVEL HELPERS
# =========================================================
def http_get(url: str) -> requests.Response:
    return requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)


def safe_read_csv_file(path: str) -> pd.DataFrame:
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


def ensure_universe_schema(df: pd.DataFrame, country_default: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country", "source"])

    df = normalize_cols(df)

    rename_map = {}
    for c in df.columns:
        if c in ("symbol", "sym", "code", "act symbol"):
            rename_map[c] = "ticker"
        elif c in ("security name", "company", "companyname", "instrument", "security"):
            rename_map[c] = "name"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "ticker" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "ticker"})

    for col in ("name", "sector", "country", "source"):
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).fillna("")
    df["country"] = df["country"].astype(str).fillna("")
    df["source"] = df["source"].astype(str).fillna("")

    if country_default:
        df.loc[df["country"].str.strip() == "", "country"] = country_default

    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name", "sector", "country", "source"]].reset_index(drop=True)


def google_news_link(query: str) -> str:
    return f"https://news.google.com/search?q={quote(query)}&hl=da&gl=DK&ceid=DK%3Ada"


def universe_path(name: str) -> str:
    return os.path.join(UNIVERSE_DIR, f"{name}.csv")


# =========================================================
# STOOQ PRICE DATA
# =========================================================
def stooq_symbol(symbol: str) -> str:
    return (symbol or "").strip()


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_daily_stooq(symbol: str, years: int = 10) -> pd.DataFrame:
    sym = stooq_symbol(symbol).lower()
    if not sym:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = http_get(url)
        if r.status_code != 200:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame()

    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

    if years and years > 0 and not df.empty:
        cutoff = df["Date"].max() - pd.Timedelta(days=int(365.25 * years))
        df = df[df["Date"] >= cutoff]

    return df.reset_index(drop=True)


# =========================================================
# STOOQ UNIVERSE BUILDER
# =========================================================
def find_next_link(html: str, current_url: str) -> str:
    patterns = [
        r'href="([^"]+)">Next 100<',
        r'href="([^"]+)">Next<',
    ]
    for pattern in patterns:
        m = re.search(pattern, html, flags=re.IGNORECASE)
        if m:
            return urljoin(current_url, m.group(1))
    return ""


def extract_table_from_html(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return pd.DataFrame()

    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "symbol" in cols and "name" in cols:
            t = t.copy()
            t.columns = cols
            return t
    return pd.DataFrame()


def scrape_stooq_group(group_url: str, country: str, source_label: str, max_pages: int = 250) -> pd.DataFrame:
    seen_urls = set()
    current = group_url
    frames = []

    for _ in range(max_pages):
        if not current or current in seen_urls:
            break
        seen_urls.add(current)

        try:
            r = http_get(current)
            if r.status_code != 200:
                break
            html = r.text
        except Exception:
            break

        tbl = extract_table_from_html(html)
        if not tbl.empty:
            tbl = tbl.rename(columns={"symbol": "ticker", "name": "name"})
            keep_cols = [c for c in ["ticker", "name"] if c in tbl.columns]
            tbl = tbl[keep_cols].copy()
            tbl["ticker"] = tbl["ticker"].astype(str).str.strip()
            tbl["name"] = tbl["name"].astype(str).str.strip()
            tbl["sector"] = ""
            tbl["country"] = country
            tbl["source"] = source_label
            tbl = tbl[tbl["ticker"].str.len() > 0]
            frames.append(tbl)

        nxt = find_next_link(html, current)
        if not nxt or nxt == current:
            break
        current = nxt

    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country", "source"])

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker"])
    return ensure_universe_schema(out, country_default=country)


def build_world_universes() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    per_group_frames = []

    for key, meta in STOOQ_GROUPS.items():
        df = scrape_stooq_group(
            group_url=meta["url"],
            country=meta["country"],
            source_label=meta["label"],
        )
        path = universe_path(key)
        df.to_csv(path, index=False, encoding="utf-8")
        counts[key] = len(df)

        if key != "us_etfs":
            per_group_frames.append(df)

    # Build combined US universe
    us_parts = []
    for key in ["us_all"]:
        p = safe_read_csv_file(universe_path(key))
        if not p.empty:
            us_parts.append(ensure_universe_schema(p, country_default="US"))

    us_combined = (
        pd.concat(us_parts, ignore_index=True).drop_duplicates(subset=["ticker"])
        if us_parts else pd.DataFrame(columns=["ticker", "name", "sector", "country", "source"])
    )
    us_combined.to_csv(universe_path("us_combined"), index=False, encoding="utf-8")
    counts["us_combined"] = len(us_combined)

    # Build global combined universe
    existing_local = []
    for fname in os.listdir(UNIVERSE_DIR):
        if not fname.endswith(".csv"):
            continue
        # Skip combined files to avoid recursion
        if fname in {"global_all.csv", "us_combined.csv"}:
            continue
        p = safe_read_csv_file(os.path.join(UNIVERSE_DIR, fname))
        if not p.empty:
            existing_local.append(ensure_universe_schema(p))

    global_df = (
        pd.concat(existing_local, ignore_index=True).drop_duplicates(subset=["ticker"])
        if existing_local else pd.DataFrame(columns=["ticker", "name", "sector", "country", "source"])
    )
    global_df.to_csv(universe_path("global_all"), index=False, encoding="utf-8")
    counts["global_all"] = len(global_df)

    return counts


def available_universes() -> Dict[str, str]:
    files = {}
    if not os.path.exists(UNIVERSE_DIR):
        return files

    for fname in sorted(os.listdir(UNIVERSE_DIR)):
        if not fname.endswith(".csv"):
            continue
        key = fname[:-4]
        path = os.path.join(UNIVERSE_DIR, fname)
        files[key] = path
    return files


def load_universe_by_key(key: str) -> Tuple[pd.DataFrame, str]:
    path = available_universes().get(key, "")
    if not path:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country", "source"]), "Ukendt univers."

    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "name", "sector", "country", "source"]), f"Mangler fil: {path}"

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        if "<<<<<<<" in head or ">>>>>>>" in head or "=======" in head:
            return pd.DataFrame(columns=["ticker", "name", "sector", "country", "source"]), f"Merge-conflict i {path}"
    except Exception:
        pass

    df = safe_read_csv_file(path)
    df = ensure_universe_schema(df)
    if df.empty:
        return df, f"Universet er tomt: {path}"
    return df, ""


def nice_universe_label(key: str) -> str:
    mapping = {
        "global_all": "GLOBAL_ALL",
        "us_combined": "US_COMBINED",
        "us_all": "US_ALL",
        "uk_all": "UK_ALL",
        "japan_all": "JAPAN_ALL",
        "germany_all": "GERMANY_ALL",
        "hongkong_all": "HONGKONG_ALL",
        "poland_all": "POLAND_ALL",
        "hungary_all": "HUNGARY_ALL",
        "us_etfs": "US_ETFS",
    }
    return mapping.get(key, key.upper())


# =========================================================
# INDICATORS / SCORING
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
    first = float(d["Close"].iloc[0])
    out["MAX"] = pct_change(last_close, first)
    return out


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
                if (
                    str(prev.get("ticker", "")) == ticker
                    and str(prev.get("action", "")) == action
                    and pd.to_numeric(prev.get("last", np.nan), errors="coerce") == last
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
def relative_strength(ticker: str, benchmark: str, days: int) -> float:
    a = fetch_daily_stooq(ticker, years=10)
    b = fetch_daily_stooq(benchmark, years=10)

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
# AUTO-BUILD IF NEEDED
# =========================================================
existing = available_universes()
if "global_all" not in existing:
    with st.spinner("Bygger globale univers-filer første gang ..."):
        try:
            build_world_universes()
        except Exception as e:
            st.warning(f"Auto-build kunne ikke gennemføres endnu: {e}")


# =========================================================
# UI
# =========================================================
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Indstillinger")
    top_n = st.slider("Top N (screening)", 5, 50, DEFAULT_TOPN, 1)
    years = st.slider("Historik (år)", 1, 10, 5, 1)
    max_screen = st.slider("Max tickers pr. screening", 20, 1500, 300, 10)

    st.divider()
    st.subheader("🌍 Universe automation")
    if st.button("Byg / opdater alle universer", type="primary"):
        with st.spinner("Henter stock-grupper fra Stooq og bygger CSV-filer ..."):
            counts = build_world_universes()
        st.success("Universer opdateret.")
        st.json(counts)
        st.rerun()

    files = available_universes()
    if files:
        st.caption(f"Tilgængelige universer: {len(files)}")
        for k in sorted(files.keys())[:12]:
            st.caption(f"• {k}")
    else:
        st.caption("Ingen univers-filer endnu.")

    st.divider()
    st.subheader("📌 Hjælp")
    st.markdown(
        """
- **Søg & analyse**: vælg papir, se kurs, afkast, signal og nyheder
- **Screening**: Top N på valgt univers
- **Portefølje**: tilføj beholdninger og få signaler
- **Tema**: momentum-proxy på temaer
        """
    )

tabs = st.tabs(["🔎 Søg & analyse", "🏁 Screening", "💼 Portefølje", "🧭 Tema", "🛠 Data"])

tab_search, tab_screener, tab_portfolio, tab_themes, tab_data = tabs


# =========================================================
# TAB 1: SEARCH
# =========================================================
with tab_search:
    st.subheader("🔎 Søg & analyse")

    universe_keys = sorted(available_universes().keys())
    if not universe_keys:
        st.error("Ingen univers-filer tilgængelige endnu. Brug 'Byg / opdater alle universer'.")
    else:
        left, right = st.columns([1, 2])

        with left:
            default_idx = universe_keys.index("global_all") if "global_all" in universe_keys else 0
            universe_key = st.selectbox(
                "Vælg univers",
                universe_keys,
                index=default_idx,
                format_func=nice_universe_label,
                key="search_universe",
            )

            uni, uni_err = load_universe_by_key(universe_key)
            if uni_err:
                st.error(uni_err)

            ticker = ""
            name = ""
            sector = ""
            country = ""

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
                    name = str(row.get("name", "")).strip()
                    sector = str(row.get("sector", "")).strip()
                    country = str(row.get("country", "")).strip()

                    st.caption(f"Ticker: **{ticker}**")
                    if name:
                        st.caption(f"Navn: {name}")
                    if country:
                        st.caption(f"Land: {country}")
                    if sector:
                        st.caption(f"Sektor: {sector}")

        with right:
            if ticker:
                df = fetch_daily_stooq(ticker, years=years)
                if df.empty:
                    st.error("Kunne ikke hente dagsdata fra Stooq for denne ticker.")
                else:
                    sig = compute_signals(df)
                    rets = period_returns(df)

                    last = float(df["Close"].iloc[-1])
                    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
                    chg = (last / prev - 1.0) * 100 if prev else 0.0

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Seneste close", f"{last:,.2f}")
                    m2.metric("Dag %", f"{chg:+.2f}%")
                    m3.metric("Seneste dato", df["Date"].iloc[-1].date().isoformat())
                    m4.metric("Signal", sig.get("action", "—"))

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

                    st.markdown("#### Seneste 10 rækker")
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
# TAB 2: SCREENER
# =========================================================
with tab_screener:
    st.subheader("🏁 Screening")

    universe_keys = sorted(available_universes().keys())
    if not universe_keys:
        st.error("Ingen univers-filer tilgængelige endnu.")
    else:
        default_idx = universe_keys.index("global_all") if "global_all" in universe_keys else 0
        universe_key2 = st.selectbox(
            "Vælg univers til screening",
            universe_keys,
            index=default_idx,
            format_func=nice_universe_label,
            key="screen_universe",
        )
        uni2, err2 = load_universe_by_key(universe_key2)
        if err2:
            st.error(err2)

        if uni2.empty:
            st.warning("Universet er tomt.")
        else:
            st.caption("Tryk knappen for at køre screening.")
            if st.button("Kør screening", type="primary"):
                tickers = uni2["ticker"].astype(str).str.strip().tolist()
                tickers = [t for t in tickers if t][:max_screen]

                rows = []
                prog = st.progress(0)
                status = st.empty()

                for i, t in enumerate(tickers, start=1):
                    status.write(f"Henter {i}/{len(tickers)}: {t}")
                    df = fetch_daily_stooq(t, years=max(3, years))
                    sig = compute_signals(df)
                    if sig:
                        meta = uni2[uni2["ticker"] == t]
                        nm = str(meta["name"].iloc[0]) if not meta.empty else ""
                        country = str(meta["country"].iloc[0]) if not meta.empty else ""
                        rows.append(
                            {
                                "Ticker": t,
                                "Navn": nm,
                                "Land": country,
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
                    st.dataframe(top, use_container_width=True, hide_index=True)

                    choices = top.apply(
                        lambda r: f"{r['Ticker']} — {r['Navn']}" if str(r['Navn']).strip() else r["Ticker"],
                        axis=1,
                    ).tolist()
                    if choices:
                        pick = st.selectbox("Vælg kandidat", choices, key="screen_pick")
                        pick_ticker = pick.split(" — ")[0].strip()
                        dfp = fetch_daily_stooq(pick_ticker, years=years)
                        if not dfp.empty:
                            st.line_chart(dfp.set_index("Date")["Close"])
                            st.markdown(f"[Nyheder]({google_news_link(pick)})")


# =========================================================
# TAB 3: PORTFOLIO
# =========================================================
def portfolio_to_df() -> pd.DataFrame:
    if not st.session_state["portfolio"]:
        return pd.DataFrame(columns=["ticker", "shares", "name"])
    df = pd.DataFrame(st.session_state["portfolio"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df["name"] = df.get("name", "").astype(str)
    df = df[df["ticker"].str.len() > 0]
    df = df[df["shares"] > 0]
    return df.reset_index(drop=True)


with tab_portfolio:
    st.subheader("💼 Portefølje")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        t = st.text_input("Ticker", value="AAPL.US", key="pf_ticker")
    with c2:
        sh = st.number_input("Antal", min_value=0.0, value=1.0, step=1.0, key="pf_shares")
    with c3:
        nm = st.text_input("Navn", value="", key="pf_name")

    if st.button("➕ Tilføj til portefølje"):
        if t.strip():
            st.session_state["portfolio"].append({"ticker": t.strip(), "shares": float(sh), "name": nm.strip()})
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
        up = st.file_uploader("Upload portfolio.json", type=["json"])
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
                tic = str(r["ticker"]).strip()
                shares = float(r["shares"])
                name = str(r.get("name", "")).strip()

                dfx = fetch_daily_stooq(tic, years=max(3, years))
                sig = compute_signals(dfx)

                last = float(dfx["Close"].iloc[-1]) if not dfx.empty else np.nan
                value = shares * last if np.isfinite(last) else np.nan

                rows.append(
                    {
                        "Ticker": tic,
                        "Navn": name,
                        "Antal": shares,
                        "Seneste": round(last, 4) if np.isfinite(last) else np.nan,
                        "Værdi": round(value, 2) if np.isfinite(value) else np.nan,
                        "Signal": sig.get("action", "—"),
                        "Score": sig.get("score", np.nan),
                        "RSI": round(sig.get("rsi", np.nan), 1) if sig else np.nan,
                        "Mom20%": round(sig.get("mom20", np.nan), 1) if sig else np.nan,
                        "Risiko": sig.get("risk", "—"),
                        "Forklaring": sig.get("why", ""),
                        "Nyheder": google_news_link(f"{tic} {name}".strip()),
                    }
                )

        out = pd.DataFrame(rows)
        total = float(out["Værdi"].sum()) if "Værdi" in out.columns else 0.0
        out["Vægt %"] = (out["Værdi"] / total * 100.0).round(2) if total > 0 else np.nan
        st.dataframe(out.sort_values("Vægt %", ascending=False), use_container_width=True, hide_index=True)


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
# TAB 5: DATA
# =========================================================
with tab_data:
    st.subheader("🛠 Data / universer")

    files = available_universes()
    if not files:
        st.warning("Ingen univers-filer fundet.")
    else:
        rows = []
        for key, path in sorted(files.items()):
            df = safe_read_csv_file(path)
            rows.append(
                {
                    "Univers": nice_universe_label(key),
                    "Fil": path,
                    "Rækker": len(df),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Nulstil signal-log")
    if st.button("Slet signals_log.csv"):
        if os.path.exists(SIGNAL_LOG):
            os.remove(SIGNAL_LOG)
        st.success("Signal-log slettet.")
        st.rerun()


st.caption("Data: Stooq dagsdata + automatisk universe-builder. Nyheder: Google News links. Ikke finansiel rådgivning.")
