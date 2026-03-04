import math
from datetime import datetime, timedelta
from io import StringIO
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =============================
# App config
# =============================
st.set_page_config(
    page_title="Stock Dashboard (gratis)",
    layout="wide",
    page_icon="📊",
)

APP_TITLE = "📊 Stock Dashboard (EU + US + Tema-radar) — gratis"
DATA_DIR = "data/universes"
DEFAULT_TOPN = 10


# =============================
# Utilities
# =============================
def _safe_read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        return pd.DataFrame()


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universe CSV kan indeholde:
      - ticker (påkrævet)
      - name, sector, country (valgfri)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    df = _normalize_cols(df)

    # aliaser
    colmap = {}
    for c in df.columns:
        if c in ("symbol", "sym", "code"):
            colmap[c] = "ticker"
        if c in ("company", "companyname", "instrument", "security"):
            colmap[c] = "name"
    if colmap:
        df = df.rename(columns=colmap)

    if "ticker" not in df.columns:
        # gæt første kolonne
        df = df.rename(columns={df.columns[0]: "ticker"})

    for col in ("name", "sector", "country"):
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name", "sector", "country"]].reset_index(drop=True)


def _stooq_symbol(symbol: str, market_hint: str = "") -> str:
    """
    Stooq bruger suffix fx:
      - AAPL.US
      - NOVO-B.CO
      - SAP.DE
    Hvis ingen '.' og market_hint='US' => tilføj .US
    """
    s = (symbol or "").strip()
    if not s:
        return s
    if "." in s:
        return s
    if market_hint.upper() == "US":
        return f"{s}.US"
    return s


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_daily_ohlc_stooq(symbol: str, years: int = 10, market_hint: str = "") -> pd.DataFrame:
    """
    Gratis daglige OHLCV fra Stooq:
      https://stooq.com/q/d/l/?s={sym}&i=d
    Returnerer df med kolonner: Date, Open, High, Low, Close, Volume
    """
    sym = _stooq_symbol(symbol, market_hint=market_hint).lower()
    if not sym:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = _safe_read_csv_text(r.text)
    except Exception:
        return pd.DataFrame()

    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    if years and years > 0 and not df.empty:
        cutoff = df["Date"].max() - pd.Timedelta(days=int(365.25 * years))
        df = df[df["Date"] >= cutoff]

    return df.reset_index(drop=True)


def google_news_link(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/search?q={q}&hl=da&gl=DK&ceid=DK%3Ada"


# =============================
# Auto-Universe Builder (gratis)
# =============================
def _is_file_probably_empty(df: pd.DataFrame) -> bool:
    return df is None or df.empty or df.shape[0] < 1 or df.shape[1] < 1


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _build_sp500_from_wikipedia() -> pd.DataFrame:
    """
    Bygger S&P500 fra Wikipedia (gratis).
    Output: ticker (Stooq-format), name, sector, country
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    # typisk er første tabel constituents
    t = tables[0].copy()
    t.columns = [str(c).strip() for c in t.columns]

    # kolonnenavne kan variere lidt – vi tager det robuste
    sym_col = None
    name_col = None
    sector_col = None
    for c in t.columns:
        lc = c.lower()
        if "symbol" in lc:
            sym_col = c
        if "security" in lc:
            name_col = c
        if "gics sector" in lc:
            sector_col = c

    if sym_col is None:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    out = pd.DataFrame()
    out["ticker"] = t[sym_col].astype(str).str.strip()
    out["ticker"] = out["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    out["ticker"] = out["ticker"] + ".US"  # Stooq format
    out["name"] = t[name_col].astype(str).str.strip() if name_col else ""
    out["sector"] = t[sector_col].astype(str).str.strip() if sector_col else ""
    out["country"] = "US"
    out = _ensure_universe_schema(out)
    return out


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _build_germany_dax_from_wikipedia() -> pd.DataFrame:
    """
    Gratis – bygger typisk DAX-liste fra Wikipedia.
    (Giver dig et fungerende 'Germany' univers uden tom fil)
    """
    url = "https://en.wikipedia.org/wiki/DAX"
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    # Find en tabel der har 'Ticker symbol' eller lignende
    best = None
    for tb in tables:
        cols = [str(c).lower() for c in tb.columns]
        if any("ticker" in c and "symbol" in c for c in cols) or any("ticker" in c for c in cols):
            best = tb
            break

    if best is None:
        # fallback: prøv første tabel
        best = tables[0]

    t = best.copy()
    t.columns = [str(c).strip() for c in t.columns]
    # gæt kolonner
    sym_col = None
    name_col = None
    sector_col = None
    for c in t.columns:
        lc = c.lower()
        if "ticker" in lc and sym_col is None:
            sym_col = c
        if "company" in lc or "name" in lc:
            name_col = c
        if "industry" in lc or "sector" in lc:
            sector_col = c

    if sym_col is None:
        # prøv første kolonne
        sym_col = t.columns[0]

    out = pd.DataFrame()
    out["ticker"] = t[sym_col].astype(str).str.strip()
    # DAX tickers på Wikipedia kan være "SAP" -> for Stooq skal vi have SAP.DE
    out["ticker"] = out["ticker"].apply(lambda x: x if "." in x else f"{x}.DE")
    out["name"] = t[name_col].astype(str).str.strip() if name_col else ""
    out["sector"] = t[sector_col].astype(str).str.strip() if sector_col else ""
    out["country"] = "DE"
    out = _ensure_universe_schema(out)
    return out


def ensure_universe_files() -> dict:
    """
    Sørger for at univers-filer ikke er tomme.
    - S&P500 og Germany kan auto-bygges
    - Andre kan du stadig vedligeholde manuelt (eller udvide builder senere)
    Returnerer status-messages.
    """
    import os

    os.makedirs(DATA_DIR, exist_ok=True)

    status = {}

    targets = {
        "sp500.csv": ("S&P 500 (US)", _build_sp500_from_wikipedia),
        "germany_de.csv": ("Germany (DE)", _build_germany_dax_from_wikipedia),
    }

    for fname, (label, builder) in targets.items():
        path = f"{DATA_DIR}/{fname}"
        needs = True
        try:
            if os.path.exists(path) and os.path.getsize(path) > 10:
                df = pd.read_csv(path)
                needs = _is_file_probably_empty(df)
        except Exception:
            needs = True

        if needs:
            dfb = builder()
            if dfb is None or dfb.empty:
                status[label] = "Kunne ikke auto-bygge (netværk/format)."
            else:
                dfb.to_csv(path, index=False)
                status[label] = f"Auto-bygget: {len(dfb)} tickers."
        else:
            status[label] = "OK (fil findes)."

    return status


def load_universe_csv(path: str) -> tuple[pd.DataFrame, str]:
    try:
        df = pd.read_csv(path)
        df = _ensure_universe_schema(df)
        if df.empty:
            return df, f"Universe-filen er tom: {path}"
        return df, ""
    except Exception as e:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Kunne ikke læse {path}: {e}"


# =============================
# Indicators / signals
# =============================
def _rsi(close: pd.Series, period: int = 14) -> float:
    close = close.dropna()
    if len(close) < period + 5:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _pct_change(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return (a / b - 1.0) * 100.0


def compute_signals(df: pd.DataFrame) -> dict:
    """
    Robust signal dict (trend, RSI, momentum, vol, drawdown)
    """
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    close = df["Close"].astype(float).dropna()
    if len(close) < 80:
        return {}

    last = float(close.iloc[-1])

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    trend_up = bool(ma50.iloc[-1] > ma200.iloc[-1]) if len(close) >= 200 else False

    rsi14 = _rsi(close, 14)
    mom20 = _pct_change(last, float(close.iloc[-21])) if len(close) >= 21 else float("nan")

    ret = close.pct_change().dropna()
    vol20 = float(ret.rolling(20).std().iloc[-1] * 100.0) if len(ret) >= 25 else float("nan")

    window = 63
    if len(close) >= window:
        peak = float(close.iloc[-window:].max())
        dd = (last / peak - 1.0) * 100.0
    else:
        dd = float("nan")

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
    if (not np.isnan(dd) and dd < -25):
        risk = "Meget høj"

    why = []
    if trend_up:
        why.append("Trend op (MA50>MA200)")
    if not np.isnan(rsi14):
        why.append(f"RSI {rsi14:.0f}")
    if not np.isnan(mom20):
        why.append(f"Momentum 20d {mom20:.1f}%")
    if not np.isnan(dd):
        why.append(f"DD 3m {dd:.1f}%")
    if not np.isnan(vol20):
        why.append(f"Vol20 {vol20:.2f}%")

    buy_zone = trend_up and (not np.isnan(rsi14)) and (35 <= rsi14 <= 60) and (not np.isnan(mom20)) and (mom20 >= 0)
    sell_zone = (not trend_up and (not np.isnan(rsi14)) and (rsi14 < 40)) or (risk in ("Meget høj",))

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
        "score": float(round(score, 2)),
        "action": action,
        "why": " • ".join(why) if why else "",
    }


# =============================
# Learning log (simpelt, gratis, uden DB)
# =============================
def _init_learning():
    if "learning_log" not in st.session_state:
        st.session_state["learning_log"] = []  # list of dicts


def log_signal_event(ticker: str, name: str, signal: dict):
    _init_learning()
    if not signal:
        return
    st.session_state["learning_log"].append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "ticker": ticker,
            "name": name or "",
            "action": signal.get("action", ""),
            "score": signal.get("score", np.nan),
            "last": signal.get("last", np.nan),
            "why": signal.get("why", ""),
        }
    )


def evaluate_learning(df_prices: pd.DataFrame, log_row: dict) -> dict:
    """
    Evaluerer "ramte vi" ved at se efterfølgende afkast siden log-tidspunkt (hvis muligt).
    Dette er en simpel feedback-løkke – ikke ML.
    """
    if df_prices is None or df_prices.empty:
        return {}

    try:
        t0 = pd.to_datetime(log_row["timestamp"], errors="coerce")
    except Exception:
        return {}

    if pd.isna(t0):
        return {}

    p = df_prices.copy()
    p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
    p = p.dropna(subset=["Date"]).sort_values("Date")
    p = p[p["Date"] >= t0]
    if p.empty:
        return {}

    last0 = float(log_row.get("last", np.nan))
    last_now = float(p["Close"].iloc[-1])
    if np.isnan(last0) or last0 == 0:
        return {}

    ret = (last_now / last0 - 1.0) * 100.0
    return {"return_since_%": round(ret, 2), "last_now": round(last_now, 2)}


# =============================
# Tema/forecast helpers (Stooq + Google News RSS)
# =============================
def _to_series_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df.copy()
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    s = s.dropna(subset=["Date"]).sort_values("Date")
    s = s.set_index("Date")["Close"].astype(float)
    return s.dropna()


def _last_n_months_return(close: pd.Series, months: int) -> float:
    if close is None or close.empty:
        return np.nan
    n = int(months * 21)
    if len(close) < n + 5:
        return np.nan
    last = float(close.iloc[-1])
    past = float(close.iloc[-(n + 1)])
    if past == 0:
        return np.nan
    return (last / past - 1.0) * 100.0


def _rel_strength(asset_close: pd.Series, bench_close: pd.Series, months: int) -> float:
    ra = _last_n_months_return(asset_close, months)
    rb = _last_n_months_return(bench_close, months)
    if np.isnan(ra) or np.isnan(rb):
        return np.nan
    return ra - rb


def _trend_regime(close: pd.Series) -> str:
    if close is None or close.empty or len(close) < 220:
        return "—"
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    if np.isnan(ma50) or np.isnan(ma200):
        return "—"
    if ma50 > ma200:
        return "Up"
    if ma50 < ma200:
        return "Down"
    return "Sideways"


def _vol20(close: pd.Series) -> float:
    if close is None or close.empty or len(close) < 40:
        return np.nan
    ret = close.pct_change().dropna()
    if len(ret) < 25:
        return np.nan
    return float(ret.rolling(20).std().iloc[-1] * 100.0)


def _dd3m(close: pd.Series) -> float:
    if close is None or close.empty or len(close) < 80:
        return np.nan
    w = 63
    last = float(close.iloc[-1])
    peak = float(close.iloc[-w:].max())
    if peak == 0:
        return np.nan
    return (last / peak - 1.0) * 100.0


def _momentum_score(rs1, rs3, rs6, rs12, trend: str, vol: float, dd: float) -> float:
    score = 0.0
    for v, w in [(rs1, 1.0), (rs3, 1.5), (rs6, 2.0), (rs12, 2.5)]:
        if not np.isnan(v):
            score += w * (v / 5.0)

    if trend == "Up":
        score += 1.5
    elif trend == "Down":
        score -= 1.0

    if not np.isnan(vol):
        score -= max(0.0, (vol - 2.5) / 2.0)

    if not np.isnan(dd):
        score += min(1.0, max(-2.0, (dd + 10.0) / 10.0))

    return float(round(score, 4))


def _google_news_rss_url(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=da&gl=DK&ceid=DK:da"


def _parse_google_news_rss(query: str, days: int = 7, max_items: int = 30) -> tuple[int, float, list[str]]:
    """
    Returnerer:
      - count (sidste X dage)
      - tone (-1..+1) simpel
      - titles (op til 5)
    """
    url = _google_news_rss_url(query)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200 or not r.text.strip():
            return 0, 0.0, []
    except Exception:
        return 0, 0.0, []

    try:
        root = ET.fromstring(r.text)
    except Exception:
        return 0, 0.0, []

    pos_words = {
        "beats", "surges", "rises", "up", "record", "strong", "growth", "profit", "upgrade",
        "stiger", "opjust", "opjusterer", "rekord", "stærk", "vækst", "overskud", "opgradering"
    }
    neg_words = {
        "misses", "falls", "down", "warning", "weak", "decline", "lawsuit", "downgrade",
        "falder", "nedjust", "nedjusterer", "advarsel", "svag", "fald", "retssag", "nedgradering"
    }

    cutoff = datetime.utcnow() - timedelta(days=days)
    count = 0
    tone = 0.0
    titles = []

    for item in root.findall(".//item")[:max_items]:
        title_el = item.find("title")
        pub_el = item.find("pubDate")
        if title_el is None or pub_el is None:
            continue
        title = (title_el.text or "").strip()
        pub = (pub_el.text or "").strip()

        try:
            dt = parsedate_to_datetime(pub)
            if dt.tzinfo is not None:
                dt = dt.astimezone(tz=None).replace(tzinfo=None)
        except Exception:
            continue

        if dt < cutoff:
            continue

        count += 1
        t = title.lower()
        p = sum(1 for w in pos_words if w in t)
        n = sum(1 for w in neg_words if w in t)
        if p + n > 0:
            tone += (p - n) / (p + n)

        if len(titles) < 5:
            titles.append(title)

    if count == 0:
        return 0, 0.0, []
    tone_score = float(max(-1.0, min(1.0, tone / max(1, count))))
    return count, tone_score, titles


# =============================
# Timeframes for chart
# =============================
def slice_timeframe(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")
    last = d["Date"].max()

    if tf == "Dag":
        return d.tail(60)
    if tf == "Uge":
        w = d.set_index("Date")[["Close"]].resample("W").last().dropna().reset_index()
        return w.tail(104)
    if tf == "Måned":
        m = d.set_index("Date")[["Close"]].resample("M").last().dropna().reset_index()
        return m.tail(120)
    if tf == "3M":
        cutoff = last - pd.Timedelta(days=92)
        return d[d["Date"] >= cutoff]
    if tf == "6M":
        cutoff = last - pd.Timedelta(days=183)
        return d[d["Date"] >= cutoff]
    if tf == "I år":
        cutoff = pd.Timestamp(year=last.year, month=1, day=1)
        return d[d["Date"] >= cutoff]
    if tf == "3 år":
        cutoff = last - pd.Timedelta(days=int(365.25 * 3))
        return d[d["Date"] >= cutoff]
    if tf == "5 år":
        cutoff = last - pd.Timedelta(days=int(365.25 * 5))
        return d[d["Date"] >= cutoff]
    if tf == "10 år":
        cutoff = last - pd.Timedelta(days=int(365.25 * 10))
        return d[d["Date"] >= cutoff]
    if tf == "MAX":
        return d
    return d


# =============================
# Sidebar + universes
# =============================
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Indstillinger")
    top_n = st.slider("Top N (screening)", 5, 50, DEFAULT_TOPN, 1)
    years = st.slider("Historik (år)", 1, 15, 10, 1)
    max_screen = st.slider("Max tickers pr. screening", 20, 400, 150, 10)

    st.divider()
    if st.button("🔁 Auto-byg universer (S&P500 + Germany)", type="primary"):
        with st.spinner("Bygger/retter universer..."):
            status = ensure_universe_files()
        st.success("Færdig.")
        for k, v in status.items():
            st.write(f"- {k}: {v}")

    st.divider()
    st.subheader("📌 Hjælp (dansk)")
    st.markdown(
        """
**Faner**
- **Søg & analyse**: vælg aktie/ETF + timeframe + signal + nyhedsflow.
- **Screening**: top kandidater ud fra trend/RSI/momentum.
- **Portefølje**: dynamisk portefølje i app + signal-vurdering løbende.
- **Tema/forecast**: momentum-proxy via ETF’er + nyhedsheat.

**Signal**
- **KØB / KIG NÆRMERE**: trend op + RSI ok + momentum ikke negativt.
- **HOLD / AFVENT**: blandet billede.
- **SÆLG / UNDGÅ**: svag trend / høj risiko.

Ikke finansiel rådgivning.
        """
    )


UNIVERSE_FILES = {
    "S&P 500 (US)": f"{DATA_DIR}/sp500.csv",
    "Germany (DE)": f"{DATA_DIR}/germany_de.csv",
    "STOXX Europe 600": f"{DATA_DIR}/stoxx600.csv",
    "Nordics (DK)": f"{DATA_DIR}/nordics_dk.csv",
    "Nordics (SE)": f"{DATA_DIR}/nordics_se.csv",
}


def get_universe(name: str) -> pd.DataFrame:
    path = UNIVERSE_FILES.get(name, "")
    df, err = load_universe_csv(path)
    if err:
        # Hvis det er et af de auto-byggelige universer, prøv auto-build én gang
        if "S&P" in name or "Germany" in name:
            ensure_universe_files()
            df, err = load_universe_csv(path)

    if err:
        st.error(f"Kunne ikke indlæse univers: {err}")
    return df


# =============================
# Tabs
# =============================
tab_search, tab_screener, tab_portfolio, tab_themes = st.tabs(
    ["🔎 Søg & analyse", "🏁 Screening (Top N)", "💼 Portefølje", "🧭 Tema/forecast"]
)


# =============================
# TAB: Search & analyse
# =============================
with tab_search:
    st.subheader("🔎 Søg & analyse")
    c1, c2 = st.columns([1, 2])

    with c1:
        universe_name = st.selectbox("Vælg univers", list(UNIVERSE_FILES.keys()), index=0)
        uni = get_universe(universe_name)

        mode = st.radio("Vælg metode", ["Fra univers-liste", "Manuel ticker"], horizontal=True)

        timeframe = st.selectbox(
            "Visning (dagskurser/uge/måned/...)",
            ["Dag", "Uge", "Måned", "3M", "6M", "I år", "3 år", "5 år", "10 år", "MAX"],
            index=3,
        )

        market_hint = "US" if "S&P" in universe_name else ""

        if mode == "Manuel ticker":
            ticker = st.text_input("Indtast ticker (Stooq-format anbefales)", "AAPL.US")
            name = ""
        else:
            if uni.empty:
                st.warning("Universet er tomt. Tryk i sidebar: 'Auto-byg universer' eller brug 'Manuel ticker'.")
                st.stop()

            u = uni.copy()
            u["display"] = u.apply(
                lambda r: f"{r['ticker']} — {r['name']}" if str(r.get("name", "")).strip() else f"{r['ticker']}",
                axis=1,
            )

            query = st.text_input("Søg navn eller ticker", "")
            view = u
            if query.strip():
                q = query.strip().lower()
                view = view[
                    view["ticker"].str.lower().str.contains(q, na=False)
                    | view["name"].astype(str).str.lower().str.contains(q, na=False)
                ]

            if view.empty:
                st.info("Ingen match. Prøv en anden søgning.")
                st.stop()

            selection = st.selectbox("Vælg papir", view["display"].tolist(), index=0)

            rows = view[view["display"] == selection]
            if rows.empty:
                st.warning("Ingen valg i listen endnu. Prøv igen eller skift univers.")
                st.stop()

            sel = rows.iloc[0]
            ticker = str(sel["ticker"]).strip()
            name = str(sel.get("name", "")).strip()

        st.caption(f"Valgt: **{ticker}** {('— ' + name) if name else ''}")

    with c2:
        df = fetch_daily_ohlc_stooq(ticker, years=years, market_hint=market_hint)
        if df.empty:
            st.error(
                "Kunne ikke hente dagskurser fra Stooq.\n\n"
                "Tip: Brug tickers som Stooq forstår (fx AAPL.US, NOVO-B.CO, SAP.DE)."
            )
            st.stop()

        sig = compute_signals(df)
        log_signal_event(ticker, name, sig)

        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
        chg = (last / prev - 1) * 100 if prev else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Seneste close", f"{last:,.2f}")
        m2.metric("Dag %", f"{chg:.2f}%")
        m3.metric("Seneste dato", df["Date"].iloc[-1].date().isoformat())
        m4.metric("Signal", sig.get("action", "—"))

        st.caption(sig.get("why", ""))

        show = slice_timeframe(df, timeframe)
        if "Close" in show.columns:
            st.markdown(f"#### Kurs (Close) — {timeframe}")
            st.line_chart(show.set_index("Date")["Close"])

        st.markdown("#### Seneste OHLC (10 rækker)")
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

        st.markdown("#### Nyheder (følger valgt papir)")
        qtxt = f"{ticker} {name}".strip()
        st.markdown(f"- Google News: {google_news_link(qtxt)}")

        st.markdown("#### Læring (ramte vi sidst?)")
        _init_learning()
        ll = st.session_state["learning_log"][-20:]
        if ll:
            # vis tabel + evaluering for seneste 5 events for dette ticker
            rows = []
            for r in reversed(ll):
                if r["ticker"] != ticker:
                    continue
                ev = evaluate_learning(df, r)
                rows.append({**r, **ev})
                if len(rows) >= 5:
                    break
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Ingen historik for dette ticker endnu (du bygger den ved at bruge app’en).")
        else:
            st.info("Ingen historik endnu (du bygger den ved at bruge app’en).")


# =============================
# TAB: Screener
# =============================
with tab_screener:
    st.subheader("🏁 Screening (Top N) — RSI / momentum / trend")
    universe_name2 = st.selectbox("Vælg univers til screening", list(UNIVERSE_FILES.keys()), index=0, key="uni2")
    uni2 = get_universe(universe_name2)
    market_hint2 = "US" if "S&P" in universe_name2 else ""

    if uni2.empty:
        st.warning("Universet er tomt. Tryk 'Auto-byg universer' (sidebar) eller fyld CSV manuelt.")
        st.stop()

    st.caption("Tryk **Kør screening**. Vi begrænser antal tickers for hastighed.")

    if st.button("Kør screening", type="primary"):
        tickers = uni2["ticker"].astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t][:max_screen]

        rows = []
        prog = st.progress(0)
        status = st.empty()

        for i, t in enumerate(tickers, start=1):
            status.write(f"Henter {i}/{len(tickers)}: {t}")
            df = fetch_daily_ohlc_stooq(t, years=max(3, years), market_hint=market_hint2)
            sig = compute_signals(df)
            if sig:
                r = uni2[uni2["ticker"] == t]
                nm = str(r["name"].iloc[0]).strip() if not r.empty else ""
                sec = str(r["sector"].iloc[0]).strip() if not r.empty else ""
                rows.append(
                    {
                        "Ticker": t,
                        "Navn": nm,
                        "Sektor": sec,
                        "Score": sig["score"],
                        "Signal": sig["action"],
                        "Trend": "✅" if sig["trend_up"] else "—",
                        "RSI": round(sig["rsi"], 1) if not np.isnan(sig["rsi"]) else np.nan,
                        "Momentum 20d %": round(sig["mom20"], 1) if not np.isnan(sig["mom20"]) else np.nan,
                        "Vol20 %": round(sig["vol20"], 2) if not np.isnan(sig["vol20"]) else np.nan,
                        "DD 3m %": round(sig["dd3m"], 1) if not np.isnan(sig["dd3m"]) else np.nan,
                        "Seneste": round(sig["last"], 2),
                        "Hvorfor": sig["why"],
                        "Risiko": sig["risk"],
                    }
                )

            prog.progress(i / len(tickers))

        status.empty()
        prog.empty()

        if not rows:
            st.warning("Ingen tickers gav brugbar data. Tjek tickers i univers (Stooq-format).")
            st.stop()

        out = pd.DataFrame(rows).sort_values(["Score"], ascending=False).reset_index(drop=True)
        top = out.head(top_n)

        st.markdown(f"### Top {top_n} kandidater")
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("### Vælg kandidat og se chart")
        choices = top.apply(lambda r: f"{r['Ticker']} — {r['Navn']}" if r["Navn"] else r["Ticker"], axis=1).tolist()
        pick = st.selectbox("Kandidat", choices)
        pick_ticker = pick.split(" — ")[0].strip()

        dfp = fetch_daily_ohlc_stooq(pick_ticker, years=years, market_hint=market_hint2)
        if dfp.empty:
            st.error("Kunne ikke hente data for valgt kandidat.")
        else:
            st.line_chart(dfp.set_index("Date")["Close"])
            st.caption(f"Nyheder: {google_news_link(pick)}")


# =============================
# TAB: Portfolio (dynamisk)
# =============================
with tab_portfolio:
    st.subheader("💼 Portefølje (dynamisk + vurdering)")

    if "portfolio_items" not in st.session_state:
        st.session_state["portfolio_items"] = []  # list of dict: ticker, shares, name, sector

    st.markdown("### Tilføj position (uden CSV)")
    pc1, pc2, pc3 = st.columns([2, 1, 1])
    with pc1:
        pticker = st.text_input("Ticker", "AAPL.US", key="pticker")
    with pc2:
        pshares = st.number_input("Antal (shares)", min_value=0.0, value=1.0, step=1.0, key="pshares")
    with pc3:
        add = st.button("➕ Tilføj")

    if add and pticker.strip():
        st.session_state["portfolio_items"].append(
            {"ticker": pticker.strip(), "shares": float(pshares), "name": "", "sector": "", "country": ""}
        )
        st.success("Tilføjet til portefølje (session).")

    st.divider()
    st.markdown("### Eller upload CSV")
    up = st.file_uploader("Upload portefølje CSV (ticker, shares[, name, sector])", type=["csv"])

    combined = []
    combined.extend(st.session_state["portfolio_items"])

    if up is not None:
        try:
            raw = pd.read_csv(up)
            raw = _normalize_cols(raw)
            if "ticker" not in raw.columns:
                raw = raw.rename(columns={raw.columns[0]: "ticker"})
            if "shares" not in raw.columns:
                if "antal" in raw.columns:
                    raw = raw.rename(columns={"antal": "shares"})
                if "qty" in raw.columns:
                    raw = raw.rename(columns={"qty": "shares"})
            if "shares" not in raw.columns:
                st.error("CSV mangler 'shares' (eller 'antal/qty').")
                st.stop()

            for col in ("name", "sector", "country"):
                if col not in raw.columns:
                    raw[col] = ""

            for _, r in raw.iterrows():
                combined.append(
                    {
                        "ticker": str(r["ticker"]).strip(),
                        "shares": float(pd.to_numeric(r["shares"], errors="coerce") or 0.0),
                        "name": str(r.get("name", "")).strip(),
                        "sector": str(r.get("sector", "")).strip(),
                        "country": str(r.get("country", "")).strip(),
                    }
                )
        except Exception as e:
            st.error(f"Kunne ikke læse CSV: {e}")
            st.stop()

    if not combined:
        st.info("Tilføj en position eller upload CSV for at se portefølje.")
        st.stop()

    pdf = pd.DataFrame(combined)
    pdf["ticker"] = pdf["ticker"].astype(str).str.strip()
    pdf["shares"] = pd.to_numeric(pdf["shares"], errors="coerce").fillna(0.0)
    pdf = pdf[(pdf["ticker"].str.len() > 0) & (pdf["shares"] > 0)].copy()

    if pdf.empty:
        st.warning("Portefølje er tom efter filtrering (shares > 0).")
        st.stop()

    # priser + signal
    price_map = {}
    sig_map = {}

    with st.spinner("Henter priser og beregner signaler ..."):
        for t in pdf["ticker"].tolist():
            mh = "US" if t.upper().endswith(".US") else ""
            df = fetch_daily_ohlc_stooq(t, years=max(3, years), market_hint=mh)
            if df.empty:
                price_map[t] = np.nan
                sig_map[t] = {}
            else:
                price_map[t] = float(df["Close"].iloc[-1])
                sig_map[t] = compute_signals(df)

    pdf["last_price"] = pdf["ticker"].map(price_map)
    pdf["value"] = pdf["shares"] * pdf["last_price"]
    total = float(pdf["value"].sum()) if np.isfinite(pdf["value"].sum()) else 0.0
    pdf["weight_pct"] = (pdf["value"] / total * 100.0) if total > 0 else 0.0

    pdf["signal"] = pdf["ticker"].apply(lambda t: sig_map.get(t, {}).get("action", "—"))
    pdf["signal_why"] = pdf["ticker"].apply(lambda t: sig_map.get(t, {}).get("why", ""))

    st.markdown("### Beholdning (med vurdering)")
    show_cols = ["ticker", "shares", "last_price", "value", "weight_pct", "signal", "signal_why"]
    st.dataframe(
        pdf[show_cols].sort_values("weight_pct", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Sektorfordeling (hvis sector findes)")
    if "sector" in pdf.columns and not pdf["sector"].astype(str).str.strip().eq("").all():
        by_sector = (
            pdf.assign(sector=pdf["sector"].astype(str).replace("", "Ukendt"))
            .groupby("sector", dropna=False)["weight_pct"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(by_sector)
    else:
        st.info("Ingen 'sector' data. Tilføj sector i CSV eller i session-items.")

    st.markdown("### Nyheder (top 10 vægt)")
    for _, r in pdf.sort_values("weight_pct", ascending=False).head(10).iterrows():
        label = f"{r['ticker']} {r.get('name','')}".strip()
        st.markdown(f"- {label}: {google_news_link(label)}")


# =============================
# TAB: Tema/forecast (Stooq som datakilde)
# =============================
with tab_themes:
    st.subheader("🧭 Tema/forecast (momentum-proxy via ETF’er)")
    st.caption(
        "Tema-radar: Relativ styrke vs SPY + trend/risk + nyhedsheat (Google News RSS). "
        "Forecast er scenario-label (ikke kursmål). Ikke finansiel rådgivning."
    )

    days_news = st.slider("Nyhedsvindue (dage)", 3, 30, 7, 1, key="days_news")

    # Benchmark
    bench_ticker = "SPY.US"
    bench_df = fetch_daily_ohlc_stooq(bench_ticker, years=max(5, years), market_hint="US")
    bench_close = _to_series_close(bench_df)
    if bench_close.empty:
        st.error("Kunne ikke hente benchmark (SPY.US) fra Stooq.")
        st.stop()

    # Du kan udvide denne liste når du vil
    themes = {
        "AI & Software": ["QQQ.US"],
        "Semiconductors": ["SOXX.US"],
        "Elektrificering & batterier": ["LIT.US"],
        "Solenergi": ["TAN.US"],
        "Grøn energi": ["ICLN.US"],
        "Robotics/Automation": ["BOTZ.US"],
        "Defense/Aerospace": ["ITA.US"],
        "Rum / Space": ["ARKX.US"],
        "Cybersecurity": ["HACK.US"],
    }

    rows = []
    with st.spinner("Beregner tema-momentum + nyhedsheat ..."):
        for theme_name, tickers in themes.items():
            for t in tickers:
                df = fetch_daily_ohlc_stooq(t, years=max(5, years), market_hint="US")
                close = _to_series_close(df)
                if close.empty or len(close) < 260:
                    continue

                rs1 = _rel_strength(close, bench_close, 1)
                rs3 = _rel_strength(close, bench_close, 3)
                rs6 = _rel_strength(close, bench_close, 6)
                rs12 = _rel_strength(close, bench_close, 12)

                trend = _trend_regime(close)
                vol = _vol20(close)
                dd = _dd3m(close)
                score = _momentum_score(rs1, rs3, rs6, rs12, trend, vol, dd)

                q = f"{t} {theme_name}"
                news_count, tone_score, titles = _parse_google_news_rss(q, days=days_news)

                tend = []
                if trend == "Up":
                    tend.append("Trend op (MA50>MA200)")
                elif trend == "Down":
                    tend.append("Trend ned (MA50<MA200)")
                if not np.isnan(rs3):
                    tend.append(f"RS 3M vs SPY: {rs3:+.1f}pp")
                if news_count > 0:
                    tend.append(f"Nyheder: {news_count}/{days_news}d")
                if not np.isnan(vol):
                    tend.append(f"Vol20: {vol:.2f}%")
                if not np.isnan(dd):
                    tend.append(f"DD3M: {dd:.1f}%")

                # Scenario-labels (enkle og robuste)
                if trend == "Up" and (not np.isnan(rs6) and rs6 > 0) and tone_score >= -0.1:
                    scen_3m = "Bull/positiv"
                elif trend == "Down" and (not np.isnan(rs3) and rs3 < 0) and tone_score <= 0.1:
                    scen_3m = "Risk-off"
                else:
                    scen_3m = "Neutral"

                if trend == "Up" and (not np.isnan(rs12) and rs12 > 0):
                    scen_12m = "Strukturel styrke"
                elif trend == "Down" and (not np.isnan(rs12) and rs12 < 0):
                    scen_12m = "Strukturel svaghed"
                else:
                    scen_12m = "Uklart"

                rows.append(
                    {
                        "Tema": theme_name,
                        "Ticker": t,
                        "MomentumScore": score,
                        "RS_1M_vs_SPY": rs1,
                        "RS_3M_vs_SPY": rs3,
                        "RS_6M_vs_SPY": rs6,
                        "RS_12M_vs_SPY": rs12,
                        "Trend": trend,
                        "Vol20_%": vol,
                        "DD_3M_%": dd,
                        f"NewsHeat_{days_news}d": news_count,
                        "NewsTone": tone_score,
                        "Forecast_3M": scen_3m,
                        "Forecast_12M": scen_12m,
                        "Tendens (kort)": " • ".join(tend),
                        "NewsLink": google_news_link(q),
                    }
                )

    if not rows:
        st.warning("Ingen tema-ETF’er gav brugbar data (tjek tickers / Stooq).")
        st.stop()

    out = pd.DataFrame(rows).sort_values("MomentumScore", ascending=False).reset_index(drop=True)

    st.dataframe(out, use_container_width=True, hide_index=True)

    st.markdown("### 🔥 Temaer at kigge nærmere på (stærk relativ styrke)")
    topk = out.head(6)
    for _, r in topk.iterrows():
        st.markdown(
            f"- **{r['Tema']}** ({r['Ticker']}) — Score **{r['MomentumScore']:.2f}**, "
            f"RS 1M {r['RS_1M_vs_SPY']:+.1f}pp, RS 3M {r['RS_3M_vs_SPY']:+.1f}pp, "
            f"Trend **{r['Trend']}**, {days_news}d news **{int(r[f'NewsHeat_{days_news}d'])}** "
            f"→ {r['Forecast_3M']} / {r['Forecast_12M']}"
        )

    st.info(
        "Hvis du vil næste: vi kan gøre temaer dynamiske (bygget fra dine univers-CSV’er), "
        "og udvide learning-log til hit-rate pr signaltype."
    )


st.caption("Data: Stooq (gratis dagsdata). Nyheder: Google News links/RSS. Ikke finansiel rådgivning.")
