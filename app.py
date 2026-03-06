# app.py
import os
import json
from datetime import datetime
from io import StringIO
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.indicators import trade_label

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Stock Dashboard (gratis)", layout="wide", page_icon="📊")

APP_TITLE = "📊 Stock Dashboard (EU + US + Tema-radar) — gratis"
DATA_DIR = "data/universes"
LOG_DIR = "data"
SIGNAL_LOG = os.path.join(LOG_DIR, "signals_log.csv")

DEFAULT_TOPN = 10

UNIVERSE_FILES = {
    "US_ALL 5000+ (US)": f"{DATA_DIR}/us_all.csv",
    "S&P 500 (US)": f"{DATA_DIR}/sp500.csv",
    "STOXX Europe 600": f"{DATA_DIR}/stoxx600.csv",
    "Nordics (DK)": f"{DATA_DIR}/nordics_dk.csv",
    "Nordics (SE)": f"{DATA_DIR}/nordics_se.csv",
    "Germany (DE)": f"{DATA_DIR}/germany_de.csv",
}

# -----------------------------
# Robust CSV helpers
# -----------------------------
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
    Universe CSV: ticker (required), name/sector/country optional
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    df = _normalize_cols(df)

    # alias mapping
    colmap = {}
    for c in df.columns:
        if c in ("symbol", "sym", "code", "act symbol"):
            colmap[c] = "ticker"
        if c in ("security name", "company", "companyname", "instrument", "security"):
            colmap[c] = "name"
    if colmap:
        df = df.rename(columns=colmap)

    if "ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ticker"})

    for col in ("name", "sector", "country"):
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name", "sector", "country"]].reset_index(drop=True)

def load_universe_csv(path: str) -> tuple[pd.DataFrame, str]:
    try:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Mangler fil: {path}"
        # Typisk fejl: merge-conflict markers gør CSV ulæselig
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(2048)
        if "<<<<<<<" in head or ">>>>>>>" in head or "=======" in head:
            return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), (
                f"Filen {path} indeholder merge-conflicts (<<<<<<). Rens filen og commit igen."
            )

        df = pd.read_csv(path)
        df = _ensure_universe_schema(df)
        if df.empty:
            return df, f"Kunne ikke indlæse univers: {path} (tom eller ingen tickers)."
        return df, ""
    except Exception as e:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Kunne ikke læse {path}: {e}"

# -----------------------------
# Stooq price fetch
# -----------------------------
def _stooq_symbol(symbol: str) -> str:
    return (symbol or "").strip()

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_daily_ohlc_stooq(symbol: str, years: int = 5) -> pd.DataFrame:
    """
    Stooq daily OHLCV:
      https://stooq.com/q/d/l/?s={sym}&i=d
    Return: Date, Open, High, Low, Close, Volume
    """
    sym = _stooq_symbol(symbol).lower()
    if not sym:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
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

# -----------------------------
# Indicators + periods
# -----------------------------
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

def _pct(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return (a / b - 1.0) * 100.0

def period_returns(df: pd.DataFrame) -> dict:
    """
    Returns over: 1D, 1W, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y, 10Y, MAX
    Uses last available close <= target date.
    """
    if df is None or df.empty or "Close" not in df.columns or "Date" not in df.columns:
        return {}
    d = df[["Date", "Close"]].dropna().copy()
    d["Close"] = d["Close"].astype(float)
    d = d.sort_values("Date")
    if d.empty:
        return {}

    last_date = d["Date"].iloc[-1]
    last = float(d["Close"].iloc[-1])

    def close_on_or_before(target_date: pd.Timestamp) -> float:
        sub = d[d["Date"] <= target_date]
        if sub.empty:
            return float("nan")
        return float(sub["Close"].iloc[-1])

    out = {}
    out["1D"]  = _pct(last, float(d["Close"].iloc[-2])) if len(d) >= 2 else float("nan")
    out["1W"]  = _pct(last, close_on_or_before(last_date - pd.Timedelta(days=7)))
    out["1M"]  = _pct(last, close_on_or_before(last_date - pd.Timedelta(days=30)))
    out["3M"]  = _pct(last, close_on_or_before(last_date - pd.Timedelta(days=90)))
    out["6M"]  = _pct(last, close_on_or_before(last_date - pd.Timedelta(days=182)))

    ytd_start = pd.Timestamp(year=last_date.year, month=1, day=1)
    out["YTD"] = _pct(last, close_on_or_before(ytd_start))

    out["1Y"]  = _pct(last, close_on_or_before(last_date - pd.Timedelta(days=365)))
    out["3Y"]  = _pct(last, close_on_or_before(last_date - pd.Timedelta(days=365*3)))
    out["5Y"]  = _pct(last, close_on_or_before(last_date - pd.Timedelta(days=365*5)))
    out["10Y"] = _pct(last, close_on_or_before(last_date - pd.Timedelta(days=365*10)))

    first = float(d["Close"].iloc[0])
    out["MAX"] = _pct(last, first)

    return out

def compute_signals(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Close" not in df.columns:
        return {}
    close = df["Close"].astype(float).dropna()
    if len(close) < 120:
        return {}

    last = float(close.iloc[-1])

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    trend_up = bool(ma50.iloc[-1] > ma200.iloc[-1]) if len(close) >= 200 else False

    rsi14 = _rsi(close, 14)
    mom20 = _pct(last, float(close.iloc[-21])) if len(close) >= 21 else float("nan")

    ret = close.pct_change().dropna()
    vol20 = float(ret.rolling(20).std().iloc[-1] * 100.0) if len(ret) >= 25 else float("nan")

    window = 63
    dd = float("nan")
    if len(close) >= window:
        peak = float(close.iloc[-window:].max())
        dd = (last / peak - 1.0) * 100.0

    # score
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
    if trend_up: why.append("Trend op (MA50>MA200)")
    if not np.isnan(rsi14): why.append(f"RSI {rsi14:.0f}")
    if not np.isnan(mom20): why.append(f"Momentum 20d {mom20:.1f}%")
    if not np.isnan(vol20): why.append(f"Vol20 {vol20:.1f}%")

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
        "why": " • ".join(why),
    }

# -----------------------------
# News (free)
# -----------------------------
def google_news_link(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/search?q={q}&hl=da&gl=DK&ceid=DK%3Ada"

@st.cache_data(ttl=30 * 60, show_spinner=False)
def google_news_rss(query: str, limit: int = 10) -> list[dict]:
    """
    Google News RSS (gratis). Ikke perfekt, men brugbar.
    """
    q = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=da&gl=DK&ceid=DK:da"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        txt = r.text
    except Exception:
        return []

    # ultra-enkelt XML parse uden ekstra libs
    items = []
    parts = txt.split("<item>")
    for p in parts[1:]:
        title = ""
        link = ""
        pub = ""
        if "<title>" in p and "</title>" in p:
            title = p.split("<title>", 1)[1].split("</title>", 1)[0]
        if "<link>" in p and "</link>" in p:
            link = p.split("<link>", 1)[1].split("</link>", 1)[0]
        if "<pubDate>" in p and "</pubDate>" in p:
            pub = p.split("<pubDate>", 1)[1].split("</pubDate>", 1)[0]
        if title and link:
            items.append({"title": title, "link": link, "pub": pub})
        if len(items) >= limit:
            break
    return items

# -----------------------------
# Learning log
# -----------------------------
def append_signal_log(ticker: str, action: str, score: float, last: float):
    os.makedirs(LOG_DIR, exist_ok=True)
    row = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "ticker": ticker,
        "action": action,
        "score": score,
        "last": last,
    }
    df = pd.DataFrame([row])
    if os.path.exists(SIGNAL_LOG):
        try:
            old = pd.read_csv(SIGNAL_LOG)
            out = pd.concat([old, df], ignore_index=True)
        except Exception:
            out = df
    else:
        out = df
    out.to_csv(SIGNAL_LOG, index=False, encoding="utf-8")

def read_signal_log(ticker: str) -> pd.DataFrame:
    if not os.path.exists(SIGNAL_LOG):
        return pd.DataFrame()
    try:
        df = pd.read_csv(SIGNAL_LOG)
        df = df[df["ticker"] == ticker].copy()
        return df.tail(50)
    except Exception:
        return pd.DataFrame()

# -----------------------------
# UI
# -----------------------------
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Indstillinger")
    top_n = st.slider("Top N (screening)", 5, 50, DEFAULT_TOPN, 1)
    years = st.slider("Historik (år)", 1, 10, 5, 1)
    max_screen = st.slider("Max tickers pr. screening", 20, 1000, 250, 10)

    st.divider()
    st.subheader("📌 Hjælp (dansk)")
    st.markdown(
        """
- **Søg/analyse**: vælg papir, se kurs + periodeafkast + signal + nyheder.
- **Screening**: kør scoring på et univers (begrænset for hastighed).
- **Portefølje**: tilføj beholdninger løbende (antal) og få løbende signal pr. linje.
- **Tema/forecast**: teknisk momentum-proxy på tema-ETF’er (Stooq).
        """
    )

tab_search, tab_screener, tab_portfolio, tab_themes = st.tabs(
    ["🔎 Søg & analyse", "🏁 Screening (Top N)", "💼 Portefølje", "🧭 Tema/forecast"]
)

def get_universe(name: str) -> pd.DataFrame:
    path = UNIVERSE_FILES.get(name, "")
    df, err = load_universe_csv(path)
    if err:
        st.error(err)
    return df

# -----------------------------
# TAB: Search
# -----------------------------
with tab_search:
    st.subheader("🔎 Søg & analyse (dagskurser + perioder + nyheder)")

    left, right = st.columns([1, 2])

    with left:
        universe_name = st.selectbox("Vælg univers", list(UNIVERSE_FILES.keys()), index=0)
        uni = get_universe(universe_name)

        if uni.empty:
            st.warning("Universet er tomt/ulæseligt. Fix CSV (merge-conflict / tom fil) eller byg US_ALL.")
            st.info("Tip: Kør `python tools/build_universes.py`, commit `data/universes/us_all.csv`, redeploy.")
            st.stop()

        uni = uni.copy()
        uni["display"] = uni.apply(
            lambda r: f"{r['ticker']} — {r['name']}" if str(r.get("name", "")).strip() else f"{r['ticker']}",
            axis=1,
        )

        query = st.text_input("Søg navn eller ticker", "")
        view = uni
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
        sector = str(sel.get("sector", "")).strip()

        st.caption(f"Valgt: **{ticker}** {('— ' + name) if name else ''}")
        if sector:
            st.caption(f"Sektor: {sector}")

    with right:
        df = fetch_daily_ohlc_stooq(ticker, years=years)
        if df.empty:
            st.error("Kunne ikke hente dagskurser fra Stooq for denne ticker.")
            st.stop()

        sig = compute_signals(df)
        rets = period_returns(df)

        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
        chg = (last / prev - 1) * 100 if prev else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Seneste close", f"{last:,.2f}")
        m2.metric("Dag %", f"{chg:.2f}%")
        m3.metric("Seneste dato", df["Date"].iloc[-1].date().isoformat())
        m4.metric("Signal", sig.get("action", "—"))

        # log signal (learning)
        if sig:
            append_signal_log(ticker, sig.get("action",""), float(sig.get("score", np.nan)), float(sig.get("last", np.nan)))

        st.markdown("#### Periode-afkast")
        labels = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "10Y", "MAX"]
        cols = st.columns(len(labels))
        for i, k in enumerate(labels):
            v = rets.get(k, np.nan)
            txt = "—" if np.isnan(v) else f"{v:+.2f}%"
            cols[i].metric(k, txt)

        st.markdown("#### Kurs (Close)")
        st.line_chart(df.set_index("Date")["Close"])

        st.markdown("#### OHLC (seneste 10)")
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

        st.markdown("#### Nyheder (følger dit valg)")
        qtxt = f"{ticker} {name}".strip()
        st.markdown(f"- Google News (link): {google_news_link(qtxt)}")

        items = google_news_rss(qtxt, limit=10)
        if items:
            for it in items:
                st.markdown(f"- {it['title']}  \n  {it['link']}")
        else:
            st.caption("Ingen RSS resultater lige nu (eller rate-limit).")

        with st.expander("📈 Signal-historik (learning)"):
            hist = read_signal_log(ticker)
            if hist.empty:
                st.info("Ingen log endnu.")
            else:
                st.dataframe(hist, use_container_width=True, hide_index=True)

# -----------------------------
# TAB: Screener
# -----------------------------
with tab_screener:
    st.subheader("🏁 Screening (Top N) — global (vælg univers)")

    universe_name2 = st.selectbox("Vælg univers til screening", list(UNIVERSE_FILES.keys()), index=0, key="uni2")
    uni2 = get_universe(universe_name2)

    if uni2.empty:
        st.warning("Universet er tomt/ulæseligt.")
        st.stop()

    st.caption("Tryk **Kør screening** for at beregne signaler. Begrænset af max tickers for hastighed.")

    if st.button("Kør screening", type="primary"):
        tickers = uni2["ticker"].astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t][:max_screen]

        rows = []
        prog = st.progress(0)
        status = st.empty()

        for i, t in enumerate(tickers, start=1):
            status.write(f"Henter {i}/{len(tickers)}: {t}")
            dfp = fetch_daily_ohlc_stooq(t, years=max(2, years))
            sig = compute_signals(dfp)
            if sig:
                try:
                    r = uni2[uni2["ticker"] == t].iloc[0]
                    nm = str(r.get("name", "")).strip()
                    sec = str(r.get("sector", "")).strip()
                except Exception:
                    nm, sec = "", ""
                rows.append({
                    "Ticker": t,
                    "Navn": nm,
                    "Sektor": sec,
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
                })
            prog.progress(i / len(tickers))

        status.empty()
        prog.empty()

        if not rows:
            st.warning("Ingen tickers gav brugbar data. Tjek tickers (Stooq-format) i dit univers.")
            st.stop()

        out = pd.DataFrame(rows).sort_values(["Score"], ascending=False).reset_index(drop=True)
        top = out.head(top_n)

        st.markdown(f"### Top {top_n}")
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("### Vælg kandidat")
        choices = top.apply(lambda r: f"{r['Ticker']} — {r['Navn']}" if r["Navn"] else r["Ticker"], axis=1).tolist()
        pick = st.selectbox("Kandidat", choices)
        pick_ticker = pick.split(" — ")[0].strip()

        dfp = fetch_daily_ohlc_stooq(pick_ticker, years=years)
        if dfp.empty:
            st.error("Kunne ikke hente data for valgt kandidat.")
        else:
            dfi = compute_signals(df)
            latest = dfi.iloc[-1]
            label = trade_label(latest)

# -----------------------------
# TAB: Portfolio (dynamic)
# -----------------------------
def _portfolio_init():
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []  # list of dicts: ticker, shares, name(optional)

_portfolio_init()

def portfolio_to_df() -> pd.DataFrame:
    if not st.session_state["portfolio"]:
        return pd.DataFrame(columns=["ticker", "shares", "name"])
    df = pd.DataFrame(st.session_state["portfolio"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df = df[df["ticker"].str.len() > 0]
    df = df[df["shares"] > 0]
    return df.reset_index(drop=True)

with tab_portfolio:
    st.subheader("💼 Min portefølje (dynamisk) — køb/hold/sælg pr. linje")
    st.caption("Tilføj linjer løbende. Du kan eksportere/importere som JSON.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        t = st.text_input("Ticker (Stooq-format)", value="AAPL.US")
    with c2:
        sh = st.number_input("Antal", min_value=0.0, value=1.0, step=1.0)
    with c3:
        nm = st.text_input("Navn (valgfri)", value="")

    add = st.button("➕ Tilføj til portefølje")
    if add:
        st.session_state["portfolio"].append({"ticker": t.strip(), "shares": float(sh), "name": nm.strip()})

    dfp = portfolio_to_df()
    if dfp.empty:
        st.info("Porteføljen er tom. Tilføj mindst én linje.")
        st.stop()

    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown("### Beholdninger (rå)")
        st.dataframe(dfp, use_container_width=True, hide_index=True)

    with colB:
        st.markdown("### Export/Import")
        exp = json.dumps(st.session_state["portfolio"], ensure_ascii=False, indent=2)
        st.download_button("⬇️ Download JSON", data=exp, file_name="portfolio.json", mime="application/json")
        up = st.file_uploader("Upload portfolio.json", type=["json"])
        if up is not None:
            try:
                loaded = json.loads(up.read().decode("utf-8"))
                if isinstance(loaded, list):
                    st.session_state["portfolio"] = loaded
                    st.success("Importeret portfolio.json")
                    st.rerun()
                else:
                    st.error("JSON skal være en liste.")
            except Exception as e:
                st.error(f"Kunne ikke læse JSON: {e}")

    st.markdown("### Analyse pr. holding")
    rows = []
    with st.spinner("Henter dagskurser og beregner signaler ..."):
        for _, r in dfp.iterrows():
            tic = str(r["ticker"]).strip()
            shares = float(r["shares"])
            name = str(r.get("name","")).strip()
            dfx = fetch_daily_ohlc_stooq(tic, years=max(2, years))
            sig = compute_signals(dfx)
            last = float(dfx["Close"].iloc[-1]) if not dfx.empty else np.nan
            value = shares * last if np.isfinite(last) else np.nan
            rows.append({
                "Ticker": tic,
                "Navn": name,
                "Antal": shares,
                "Seneste": round(last, 4) if np.isfinite(last) else np.nan,
                "Værdi": round(value, 2) if np.isfinite(value) else np.nan,
                "Signal": sig.get("action","—"),
                "Score": sig.get("score", np.nan),
                "RSI": round(sig.get("rsi", np.nan), 1) if sig else np.nan,
                "Mom20%": round(sig.get("mom20", np.nan), 1) if sig else np.nan,
                "Risiko": sig.get("risk","—"),
                "Kort forklaring": sig.get("why",""),
                "Nyheder": google_news_link(f"{tic} {name}".strip()),
            })

    out = pd.DataFrame(rows)
    total = float(out["Værdi"].sum()) if "Værdi" in out.columns else 0.0
    if total > 0:
        out["Vægt %"] = (out["Værdi"] / total * 100.0).round(2)
    else:
        out["Vægt %"] = np.nan

    st.dataframe(out.sort_values("Vægt %", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("### Sektorfordeling (kræver sector i univers/portfolio)")
    st.caption("Næste step: merge sektor fra univers-CSV. (Kan vi lave i næste iteration.)")

# -----------------------------
# TAB: Themes (Stooq)
# -----------------------------
with tab_themes:
    st.subheader("🧭 Tema/forecast (momentum-proxy via ETF’er)")
    st.caption("Teknisk momentum-indikator (ikke rådgivning). Datakilde: Stooq.")

    # 50+ temaer (du kan udvide løbende)
    # NOTE: tickers skal være Stooq-format, ellers kommer der tom data.
    THEMES = {
        "AI & Software": ["QQQ.US", "XLK.US", "MSFT.US", "NVDA.US"],
        "Semiconductors": ["SOXX.US", "SMH.US", "NVDA.US", "AVGO.US"],
        "Cybersecurity": ["HACK.US", "CIBR.US", "PANW.US", "CRWD.US"],
        "Defense/Aerospace": ["ITA.US", "XAR.US", "LMT.US", "NOC.US"],
        "Space": ["ARKX.US", "RKLB.US", "ASTS.US"],
        "Robotics/Automation": ["BOTZ.US", "ROBO.US", "ISRG.US"],
        "Cloud/Datacenter": ["SKYY.US", "AMZN.US", "GOOGL.US"],
        "Energy (Oil&Gas)": ["XLE.US", "CVX.US", "XOM.US"],
        "Solar": ["TAN.US", "ENPH.US", "FSLR.US"],
        "Wind": ["VWS.CO", "ORSTED.CO"],
        "Clean Energy": ["ICLN.US", "PBW.US"],
        "Uranium": ["URA.US", "CCJ.US"],
        "EV & Batteries": ["LIT.US", "TSLA.US", "ALB.US"],
        "Autonomous/AV": ["TSLA.US", "GOOGL.US"],
        "Fintech": ["FINX.US", "PYPL.US", "SQ.US"],
        "Payments": ["V.US", "MA.US", "PYPL.US"],
        "Banks": ["XLF.US", "JPM.US", "BAC.US"],
        "Insurance": ["KIE.US", "AIG.US"],
        "Healthcare": ["XLV.US", "UNH.US", "JNJ.US"],
        "Biotech": ["IBB.US", "XBI.US"],
        "Medtech": ["IHI.US", "ISRG.US"],
        "Pharma": ["PFE.US", "LLY.US", "NVO.CO"],
        "Obesity/GLP-1": ["NVO.CO", "LLY.US"],
        "Consumer Staples": ["XLP.US", "PG.US", "KO.US"],
        "Luxury": ["MC.PA", "RMS.PA"],
        "E-commerce": ["AMZN.US", "BABA.US"],
        "China Tech": ["KWEB.US", "BABA.US", "TCEHY.US"],
        "India Growth": ["INDA.US"],
        "Japan": ["EWJ.US"],
        "Emerging Markets": ["EEM.US", "VWO.US"],
        "Commodities": ["DBC.US", "GLD.US"],
        "Gold": ["GLD.US", "IAU.US"],
        "Silver": ["SLV.US"],
        "Real Estate": ["VNQ.US"],
        "Utilities": ["XLU.US"],
        "Water": ["PHO.US", "FIW.US"],
        "Agriculture": ["DBA.US"],
        "Shipping": ["SEA? (placeholder)"],
        "Gaming": ["ESPO.US"],
        "Media/Streaming": ["NFLX.US", "DIS.US"],
        "Travel": ["JETS.US", "BKNG.US"],
        "Construction": ["XHB.US"],
        "Industrials": ["XLI.US"],
        "Materials": ["XLB.US"],
        "Metals & Mining": ["XME.US"],
        "Rare Earths": ["REMX.US"],
        "Telecom": ["IYZ.US"],
        "Dividends": ["VYM.US", "SCHD.US"],
        "Low Vol": ["SPLV.US"],
        "Momentum": ["MTUM.US"],
        "Small Cap": ["IWM.US"],
        "Growth": ["VUG.US"],
        "Value": ["VTV.US"],
        "Bonds (AGG)": ["AGG.US"],
        "Inflation": ["TIP.US"],
    }

    # rens placeholders der ikke findes på stooq
    # (du kan fjerne dem)
    THEMES = {k: [t for t in v if "?" not in t] for k, v in THEMES.items()}

    benchmark = "SPY.US"

    def rel_strength(ticker: str, bench: str, days: int) -> float:
        a = fetch_daily_ohlc_stooq(ticker, years=10)
        b = fetch_daily_ohlc_stooq(bench, years=10)
        if a.empty or b.empty:
            return float("nan")

        a = a[["Date","Close"]].dropna().copy()
        b = b[["Date","Close"]].dropna().copy()
        a["Date"] = pd.to_datetime(a["Date"])
        b["Date"] = pd.to_datetime(b["Date"])
        a = a.sort_values("Date")
        b = b.sort_values("Date")

        end = min(a["Date"].iloc[-1], b["Date"].iloc[-1])
        start = end - pd.Timedelta(days=days)

        def close_on(df, d):
            sub = df[df["Date"] <= d]
            if sub.empty:
                return float("nan")
            return float(sub["Close"].iloc[-1])

        a0 = close_on(a, start)
        a1 = close_on(a, end)
        b0 = close_on(b, start)
        b1 = close_on(b, end)

        if any(np.isnan(x) for x in [a0,a1,b0,b1]) or a0 == 0 or b0 == 0:
            return float("nan")

        ra = a1/a0 - 1.0
        rb = b1/b0 - 1.0
        return ra - rb  # outperformance vs SPY

    rows = []
    with st.spinner("Beregner tema-momentum (kan tage lidt tid) ..."):
        for theme, tickers in THEMES.items():
            # vælg første som proxy
            proxy = tickers[0] if tickers else ""
            if not proxy:
                continue
            rs_1m = rel_strength(proxy, benchmark, 30)
            rs_3m = rel_strength(proxy, benchmark, 90)

            score = 0.0
            if not np.isnan(rs_1m): score += rs_1m * 100
            if not np.isnan(rs_3m): score += rs_3m * 50

            rows.append({
                "Tema": theme,
                "Ticker (proxy)": proxy,
                "MomentumScore": round(score, 4),
                "RS_1M_vs_SPY": round(rs_1m, 4) if not np.isnan(rs_1m) else np.nan,
                "RS_3M_vs_SPY": round(rs_3m, 4) if not np.isnan(rs_3m) else np.nan,
            })

    dfm = pd.DataFrame(rows).sort_values("MomentumScore", ascending=False).reset_index(drop=True)
    st.dataframe(dfm, use_container_width=True, hide_index=True)

    st.markdown("### 🔥 Temaer at kigge nærmere på (stærk relativ styrke)")
    topk = dfm.head(10)
    for _, r in topk.iterrows():
        st.markdown(
            f"- **{r['Tema']}** ({r['Ticker (proxy)']}) — "
            f"RS 1M: {r['RS_1M_vs_SPY']:+.2%} , RS 3M: {r['RS_3M_vs_SPY']:+.2%}"
        )

st.caption("Data: Stooq (gratis dagsdata). Nyheder: Google News (link + RSS). Ikke finansiel rådgivning.")
