import math
from datetime import datetime, date
from io import StringIO
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =============================
# Config
# =============================
st.set_page_config(
    page_title="Stock Dashboard (gratis)",
    layout="wide",
    page_icon="📈",
)

APP_TITLE = "📈 Stock Dashboard (EU + US + Tema/forecast) — gratis"
DATA_DIR = "data/universes"
DEFAULT_TOPN = 10


# =============================
# Robust CSV helpers
# =============================
def safe_read_csv_file(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universe schema:
      ticker (required)
      name, sector, country (optional)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    df = normalize_cols(df)

    # alias
    rename = {}
    for c in df.columns:
        if c in ("symbol", "sym", "code"):
            rename[c] = "ticker"
        if c in ("company", "companyname", "instrument", "security", "name"):
            # keep name -> name
            pass
        if c in ("companyname", "company"):
            rename[c] = "name"
    if rename:
        df = df.rename(columns=rename)

    if "ticker" not in df.columns:
        # guess first col
        df = df.rename(columns={df.columns[0]: "ticker"})

    for col in ("name", "sector", "country"):
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name", "sector", "country"]].reset_index(drop=True)


# =============================
# Stooq data source
# =============================
def stooq_symbol(symbol: str, market_hint: str = "") -> str:
    """
    If user provides suffix (AAPL.US / NOVO-B.CO) keep it.
    If no suffix and market_hint==US => append .US
    Otherwise keep as-is.
    """
    s = (symbol or "").strip()
    if not s:
        return s
    if "." in s:
        return s
    if market_hint.upper() == "US":
        return f"{s}.US"
    return s


def safe_read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_daily_stooq(symbol: str, years: int = 10, market_hint: str = "") -> pd.DataFrame:
    """
    https://stooq.com/q/d/l/?s={sym}&i=d
    returns columns: Date, Open, High, Low, Close, Volume
    """
    sym = stooq_symbol(symbol, market_hint=market_hint).lower()
    if not sym:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        df = safe_read_csv_text(r.text)
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


def slice_timeframe(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    tf: 1D / 1U / 1M / 3M / 6M / YTD / 3Y / 5Y / 10Y / MAX
    Uses daily df.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df.sort_values("Date")

    last_dt = df["Date"].iloc[-1]

    if tf == "MAX":
        return df
    if tf == "1D":
        return df.tail(2)
    if tf == "1U":
        cutoff = last_dt - pd.Timedelta(days=7)
        return df[df["Date"] >= cutoff]
    if tf == "1M":
        cutoff = last_dt - pd.Timedelta(days=30)
        return df[df["Date"] >= cutoff]
    if tf == "3M":
        cutoff = last_dt - pd.Timedelta(days=90)
        return df[df["Date"] >= cutoff]
    if tf == "6M":
        cutoff = last_dt - pd.Timedelta(days=182)
        return df[df["Date"] >= cutoff]
    if tf == "YTD":
        cutoff = pd.Timestamp(year=last_dt.year, month=1, day=1)
        return df[df["Date"] >= cutoff]
    if tf == "3Y":
        cutoff = last_dt - pd.Timedelta(days=int(365.25 * 3))
        return df[df["Date"] >= cutoff]
    if tf == "5Y":
        cutoff = last_dt - pd.Timedelta(days=int(365.25 * 5))
        return df[df["Date"] >= cutoff]
    if tf == "10Y":
        cutoff = last_dt - pd.Timedelta(days=int(365.25 * 10))
        return df[df["Date"] >= cutoff]

    return df


# =============================
# Indicators + scoring
# =============================
def pct_change(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return (a / b - 1.0) * 100.0


def rsi(close: pd.Series, period: int = 14) -> float:
    close = close.dropna()
    if len(close) < period + 5:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    out = 100 - (100 / (1 + rs))
    return float(out.iloc[-1])


def compute_signals(df: pd.DataFrame) -> Dict:
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    close = df["Close"].astype(float).dropna()
    if len(close) < 80:
        return {}

    last = float(close.iloc[-1])

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    trend_up = bool(ma50.iloc[-1] > ma200.iloc[-1]) if len(close) >= 200 else False

    rsi14 = rsi(close, 14)
    mom20 = pct_change(last, float(close.iloc[-21])) if len(close) >= 21 else float("nan")

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


def google_news_link(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/search?q={q}&hl=da&gl=DK&ceid=DK%3Ada"


# =============================
# Universe loading
# =============================
UNIVERSE_FILES = {
    "S&P 500 (US)": f"{DATA_DIR}/sp500.csv",
    "STOXX Europe 600": f"{DATA_DIR}/stoxx600.csv",
    "Nordics (DK)": f"{DATA_DIR}/nordics_dk.csv",
    "Nordics (SE)": f"{DATA_DIR}/nordics_se.csv",
    "Germany (DE)": f"{DATA_DIR}/germany_de.csv",
    "Global (starter)": f"{DATA_DIR}/global_starter.csv",
}


def load_universe(name: str) -> Tuple[pd.DataFrame, str]:
    path = UNIVERSE_FILES.get(name, "")
    if not path:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), "Ukendt univers"
    df = safe_read_csv_file(path)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Kunne ikke læse {path} (tom/ugyldig CSV)."
    df = ensure_universe_schema(df)
    if df.empty:
        return df, f"{path} kunne læses, men der blev ikke fundet en 'ticker' kolonne med værdier."
    return df, ""


# =============================
# Session state: learning log + portfolio
# =============================
if "signal_log" not in st.session_state:
    # ticker -> list of dicts (date, action, last)
    st.session_state["signal_log"] = {}

if "portfolio_rows" not in st.session_state:
    st.session_state["portfolio_rows"] = []  # list of dicts: ticker, shares, name, sector, country


def log_signal(ticker: str, action: str, last: float):
    hist = st.session_state["signal_log"].setdefault(ticker, [])
    hist.append({"ts": datetime.utcnow().isoformat(timespec="seconds"), "action": action, "last": last})
    hist[:] = hist[-30:]


def signal_hit_rate(ticker: str) -> str:
    """
    Super simpel "læring": sammenlign sidste signal med efterfølgende pris.
    - Hvis sidste var KØB og pris steg siden => "ramte"
    - Hvis SÆLG og pris faldt siden => "ramte"
    """
    hist = st.session_state["signal_log"].get(ticker, [])
    if len(hist) < 2:
        return "— (mangler historik)"
    prev = hist[-2]
    cur = hist[-1]
    try:
        p0 = float(prev["last"])
        p1 = float(cur["last"])
    except Exception:
        return "—"
    action = prev["action"]
    if "KØB" in action:
        return "✅ Ramte" if p1 > p0 else "❌ Miss"
    if "SÆLG" in action:
        return "✅ Ramte" if p1 < p0 else "❌ Miss"
    return "—"


# =============================
# Sidebar
# =============================
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Indstillinger")
    top_n = st.slider("Top N (screening)", 5, 50, DEFAULT_TOPN, 1)
    years = st.slider("Historik (år til hentning)", 1, 15, 10, 1)
    max_screen = st.slider("Max tickers pr. screening", 20, 500, 150, 10)

    st.divider()
    st.subheader("📌 Signal (simpel forklaring)")
    st.markdown(
        """
- **KØB / KIG NÆRMERE**: trend op + RSI i rimelig zone + momentum ikke negativt  
- **HOLD / AFVENT**: blandet billede  
- **SÆLG / UNDGÅ**: svag trend / høj risiko  

*Ikke finansiel rådgivning.*
        """
    )


tab_search, tab_screener, tab_portfolio, tab_themes = st.tabs(
    ["🔎 Søg & analyse", "🏁 Auto screener", "💼 Portefølje", "🧭 Tema/forecast"]
)


# =============================
# TAB 1: Search & analyse
# =============================
with tab_search:
    st.subheader("🔎 Søg & analyse (dagsdata fra Stooq)")

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        universe_name = st.selectbox("Vælg univers", list(UNIVERSE_FILES.keys()), index=0)
        uni, err = load_universe(universe_name)
        if err:
            st.error(f"Kunne ikke indlæse univers: {err}")
            st.stop()

        uni = uni.copy()
        uni["display"] = uni.apply(
            lambda r: f"{r['ticker']} — {r['name']}" if str(r.get("name", "")).strip() else f"{r['ticker']}",
            axis=1,
        )

        q = st.text_input("Søg navn eller ticker", "")
        view = uni
        if q.strip():
            qq = q.strip().lower()
            view = view[
                view["ticker"].str.lower().str.contains(qq, na=False)
                | view["name"].astype(str).str.lower().str.contains(qq, na=False)
            ]

        if view.empty:
            st.info("Ingen match. Prøv anden søgning.")
            st.stop()

        selection = st.selectbox("Vælg papir", view["display"].tolist(), index=0)

        rows = view[view["display"] == selection]
        if rows.empty:
            st.warning("Ingen valg endnu. Prøv igen.")
            st.stop()

        sel = rows.iloc[0]
        ticker = str(sel["ticker"]).strip()
        name = str(sel.get("name", "")).strip()
        sector = str(sel.get("sector", "")).strip()

        market_hint = "US" if "S&P" in universe_name else ""

        tf = st.selectbox(
            "Visning (timeframe)",
            ["1D", "1U", "1M", "3M", "6M", "YTD", "3Y", "5Y", "10Y", "MAX"],
            index=3,
        )

        st.caption(f"Valgt: **{ticker}** {('— ' + name) if name else ''}")
        if sector:
            st.caption(f"Sektor: {sector}")

    with c2:
        df_all = fetch_daily_stooq(ticker, years=years, market_hint=market_hint)
        if df_all.empty:
            st.error(
                "Kunne ikke hente dagskurser fra Stooq for denne ticker.\n\n"
                "Tip: Brug tickers som Stooq forstår (fx AAPL.US, NOVO-B.CO)."
            )
            st.stop()

        df = slice_timeframe(df_all, tf)
        if df.empty:
            st.warning("Ingen data i valgt timeframe.")
            st.stop()

        last = float(df_all["Close"].iloc[-1])
        prev = float(df_all["Close"].iloc[-2]) if len(df_all) >= 2 else last
        chg = (last / prev - 1) * 100 if prev else 0.0

        sig = compute_signals(df_all)
        action = sig.get("action", "—")
        log_signal(ticker, action, last)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Seneste close", f"{last:,.2f}")
        m2.metric("Dag %", f"{chg:.2f}%")
        m3.metric("Seneste dato", df_all["Date"].iloc[-1].date().isoformat())
        m4.metric("Signal", action)
        m5.metric("Læring (sidste signal)", signal_hit_rate(ticker))

        st.markdown(f"#### Kurs (Close) — {tf}")
        st.line_chart(df.set_index("Date")["Close"])

        st.markdown("#### Nøgletal / hvorfor")
        st.write(sig.get("why", "—"))
        st.caption(f"Risiko: {sig.get('risk','—')} | Score: {sig.get('score','—')}")

        st.markdown("#### Nyheder (følger valgt papir)")
        qtxt = f"{ticker} {name}".strip()
        st.markdown(f"- Google News: {google_news_link(qtxt)}")

        with st.expander("Vis læringslog for dette papir"):
            hist = st.session_state["signal_log"].get(ticker, [])
            if not hist:
                st.info("Ingen historik endnu.")
            else:
                st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)


# =============================
# TAB 2: Auto screener
# =============================
with tab_screener:
    st.subheader("🏁 Auto screener (Top N) — RSI / momentum / trend")

    universe_name2 = st.selectbox("Vælg univers til screening", list(UNIVERSE_FILES.keys()), index=1, key="uni2")
    uni2, err2 = load_universe(universe_name2)
    if err2:
        st.error(f"Kunne ikke indlæse univers: {err2}")
        st.stop()

    market_hint2 = "US" if "S&P" in universe_name2 else ""

    st.caption(
        "Tryk **Kør screening** for at hente dagskurser og beregne signaler. "
        "Vi begrænser antal tickers for hastighed."
    )

    if st.button("Kør screening", type="primary"):
        tickers = uni2["ticker"].astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t][:max_screen]

        rows = []
        prog = st.progress(0)
        status = st.empty()

        for i, t in enumerate(tickers, start=1):
            status.write(f"Henter {i}/{len(tickers)}: {t}")
            df = fetch_daily_stooq(t, years=max(3, years), market_hint=market_hint2)
            sig = compute_signals(df)
            if sig:
                meta = uni2[uni2["ticker"] == t]
                nm = meta["name"].iloc[0] if not meta.empty else ""
                sc = meta["sector"].iloc[0] if not meta.empty else ""
                rows.append(
                    {
                        "Ticker": t,
                        "Navn": nm,
                        "Sektor": sc,
                        "Score": sig["score"],
                        "Signal": sig["action"],
                        "Trend op": "✅" if sig["trend_up"] else "—",
                        "RSI": round(sig["rsi"], 1) if not np.isnan(sig["rsi"]) else np.nan,
                        "Momentum 20d %": round(sig["mom20"], 1) if not np.isnan(sig["mom20"]) else np.nan,
                        "Vol20 %": round(sig["vol20"], 2) if not np.isnan(sig["vol20"]) else np.nan,
                        "Drawdown 3m %": round(sig["dd3m"], 1) if not np.isnan(sig["dd3m"]) else np.nan,
                        "Seneste": round(sig["last"], 2),
                        "Hvorfor": sig["why"],
                        "Risiko": sig["risk"],
                    }
                )
            prog.progress(i / len(tickers))

        status.empty()
        prog.empty()

        if not rows:
            st.warning("Ingen tickers gav brugbar data. Tjek tickers (Stooq-format) i dit univers.")
            st.stop()

        out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        top = out.head(top_n)

        st.markdown(f"### Top {top_n} kandidater")
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("### Vælg kandidat og se chart + nyheder")
        choices = top.apply(lambda r: f"{r['Ticker']} — {r['Navn']}" if r["Navn"] else r["Ticker"], axis=1).tolist()
        pick = st.selectbox("Kandidat", choices)
        pick_ticker = pick.split(" — ")[0].strip()

        dfp = fetch_daily_stooq(pick_ticker, years=years, market_hint=market_hint2)
        if dfp.empty:
            st.error("Kunne ikke hente data for valgt kandidat.")
        else:
            st.line_chart(dfp.set_index("Date")["Close"])
            st.caption(f"Nyheder: {google_news_link(pick)}")


# =============================
# TAB 3: Portfolio (dynamic)
# =============================
with tab_portfolio:
    st.subheader("💼 Portefølje (dynamisk)")

    st.caption("Tilføj dine positioner her. App’en vurderer løbende Køb/Hold/Sælg pr position (simpel model).")

    with st.expander("➕ Tilføj position"):
        t = st.text_input("Ticker (Stooq-format anbefalet, fx AAPL.US / NOVO-B.CO)", key="p_ticker").strip()
        sh = st.number_input("Antal (shares)", min_value=0.0, value=0.0, step=1.0, key="p_shares")
        nm = st.text_input("Navn (valgfrit)", key="p_name").strip()
        sc = st.text_input("Sektor (valgfrit)", key="p_sector").strip()
        co = st.text_input("Land (valgfrit)", key="p_country").strip()

        if st.button("Tilføj til portefølje"):
            if not t or sh <= 0:
                st.warning("Angiv ticker og shares > 0.")
            else:
                st.session_state["portfolio_rows"].append(
                    {"ticker": t, "shares": float(sh), "name": nm, "sector": sc, "country": co}
                )
                st.success("Tilføjet.")

    if st.button("🧹 Ryd portefølje"):
        st.session_state["portfolio_rows"] = []
        st.success("Portefølje ryddet.")

    if not st.session_state["portfolio_rows"]:
        st.info("Ingen positioner endnu. Tilføj en position ovenfor.")
        st.stop()

    pdf = pd.DataFrame(st.session_state["portfolio_rows"])
    pdf["shares"] = pd.to_numeric(pdf["shares"], errors="coerce").fillna(0.0)
    pdf = pdf[pdf["shares"] > 0].copy()
    if pdf.empty:
        st.info("Ingen positioner med shares > 0.")
        st.stop()

    price_map = {}
    signal_map = {}
    with st.spinner("Henter seneste dagskurser + signaler ..."):
        for t in pdf["ticker"].astype(str).tolist():
            df = fetch_daily_stooq(t, years=max(3, years), market_hint=("US" if t.upper().endswith(".US") else ""))
            if df.empty:
                price_map[t] = np.nan
                signal_map[t] = {"action": "—", "why": "Ingen data", "score": np.nan}
            else:
                price_map[t] = float(df["Close"].iloc[-1])
                signal_map[t] = compute_signals(df) or {"action": "—", "why": "For lidt data", "score": np.nan}

    pdf["last_price"] = pdf["ticker"].map(price_map)
    pdf["value"] = pdf["shares"] * pdf["last_price"]
    total = float(pdf["value"].sum()) if np.isfinite(pdf["value"].sum()) else 0.0
    pdf["weight_pct"] = (pdf["value"] / total * 100.0) if total > 0 else 0.0

    pdf["Signal"] = pdf["ticker"].map(lambda x: signal_map.get(x, {}).get("action", "—"))
    pdf["Forklaring"] = pdf["ticker"].map(lambda x: signal_map.get(x, {}).get("why", ""))
    pdf["Score"] = pdf["ticker"].map(lambda x: signal_map.get(x, {}).get("score", np.nan))

    st.markdown("### Overblik")
    show_cols = ["ticker", "name", "sector", "shares", "last_price", "value", "weight_pct", "Signal", "Score", "Forklaring"]
    for c in show_cols:
        if c not in pdf.columns:
            pdf[c] = ""
    st.dataframe(pdf[show_cols].sort_values("weight_pct", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("### Sektorfordeling")
    if pdf["sector"].astype(str).str.strip().eq("").all():
        st.info("Ingen sektor angivet. Tilføj sektor i portefølje for sektordiagram.")
    else:
        by_sector = (
            pdf.assign(sector=pdf["sector"].astype(str).replace("", "Ukendt"))
            .groupby("sector", dropna=False)["weight_pct"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(by_sector)

    st.markdown("### Nyheder (top 10 positioner)")
    for _, r in pdf.sort_values("weight_pct", ascending=False).head(10).iterrows():
        label = f"{r['ticker']} {r.get('name','')}".strip()
        st.markdown(f"- {label}: {google_news_link(label)}")


# =============================
# TAB 4: Tema/forecast (momentum proxy)
# =============================
with tab_themes:
    st.subheader("🧭 Tema/forecast (momentum-proxy via ETF’er) — Stooq dagsdata")
    st.caption("Teknisk momentum-indikator (ikke rådgivning). Forecast er *proxy* baseret på trend/RS — ikke en garanti.")

    # Theme ETFs (many are US tickers; Stooq may require .US)
    # Use suffix .US where possible to improve hit-rate
    themes = [
        {"Tema": "AI & Software", "Ticker": "QQQ.US"},
        {"Tema": "Cybersecurity", "Ticker": "HACK.US"},
        {"Tema": "Semiconductors", "Ticker": "SOXX.US"},
        {"Tema": "Elektrificering & batterier", "Ticker": "LIT.US"},
        {"Tema": "Solenergi", "Ticker": "TAN.US"},
        {"Tema": "Defense/Aerospace", "Ticker": "ITA.US"},
        {"Tema": "Robotics/Automation", "Ticker": "BOTZ.US"},
        {"Tema": "Grøn energi", "Ticker": "ICLN.US"},
        {"Tema": "Rumd / Space", "Ticker": "ARKX.US"},
        {"Tema": "Biotech", "Ticker": "XBI.US"},
    ]
    bench = "SPY.US"

    def rel_strength(series_a: pd.Series, series_b: pd.Series) -> float:
        # relative strength as percent change ratio over window
        if series_a is None or series_b is None:
            return float("nan")
        a = series_a.dropna()
        b = series_b.dropna()
        if len(a) < 5 or len(b) < 5:
            return float("nan")
        # align by index
        dfm = pd.DataFrame({"a": a, "b": b}).dropna()
        if dfm.empty or len(dfm) < 5:
            return float("nan")
        return float((dfm["a"].iloc[-1] / dfm["a"].iloc[0]) / (dfm["b"].iloc[-1] / dfm["b"].iloc[0]) - 1.0)

    df_b = fetch_daily_stooq(bench, years=10, market_hint="US")
    if df_b.empty:
        st.error("Kunne ikke hente benchmark (SPY.US) fra Stooq.")
        st.stop()

    b_close = df_b.set_index("Date")["Close"].astype(float)

    rows = []
    for t in themes:
        sym = t["Ticker"]
        df_t = fetch_daily_stooq(sym, years=10, market_hint="US")
        if df_t.empty:
            rows.append({**t, "MomentumScore": np.nan, "RS_1M_vs_SPY": np.nan, "RS_3M_vs_SPY": np.nan, "Trend": "Ingen data"})
            continue

        close = df_t.set_index("Date")["Close"].astype(float)
        # windows
        rs_1m = rel_strength(close.tail(22), b_close.tail(22))
        rs_3m = rel_strength(close.tail(63), b_close.tail(63))
        rs_6m = rel_strength(close.tail(126), b_close.tail(126))

        # simple momentum score
        # weight: 3m + 6m + trend
        sig = compute_signals(df_t)
        trend_bonus = 2.0 if sig.get("trend_up") else 0.0
        mom = sig.get("mom20", np.nan)
        mom_part = 0.0 if np.isnan(mom) else max(-5.0, min(10.0, mom)) / 2.0  # clamp
        score = 0.0
        for x, w in [(rs_1m, 1.0), (rs_3m, 2.0), (rs_6m, 2.0)]:
            if not np.isnan(x):
                score += float(x) * 100.0 * w
        score += trend_bonus + mom_part

        trend_txt = sig.get("action", "—")
        rows.append(
            {
                "Tema": t["Tema"],
                "Ticker": sym,
                "MomentumScore": round(score, 3),
                "RS_1M_vs_SPY": round(rs_1m, 4) if not np.isnan(rs_1m) else np.nan,
                "RS_3M_vs_SPY": round(rs_3m, 4) if not np.isnan(rs_3m) else np.nan,
                "RS_6M_vs_SPY": round(rs_6m, 4) if not np.isnan(rs_6m) else np.nan,
                "Signal": trend_txt,
            }
        )

    out = pd.DataFrame(rows).sort_values("MomentumScore", ascending=False, na_position="last").reset_index(drop=True)
    st.dataframe(out, use_container_width=True, hide_index=True)

    st.markdown("### 🔥 Temaer at kigge nærmere på (stærk relativ styrke)")
    top = out.dropna(subset=["MomentumScore"]).head(6)
    if top.empty:
        st.info("Ingen temaer med data endnu.")
    else:
        for _, r in top.iterrows():
            st.write(
                f"- **{r['Tema']} ({r['Ticker']})** — Score: {r['MomentumScore']} | "
                f"RS 1M: {r['RS_1M_vs_SPY']}, RS 3M: {r['RS_3M_vs_SPY']}, RS 6M: {r['RS_6M_vs_SPY']} | "
                f"Signal: {r['Signal']}"
            )

    st.markdown("### Trend-beskrivelse (kort)")
    st.write(
        "MomentumScore kombinerer relativ styrke vs SPY (1M/3M/6M) + trend (MA50/MA200) + kort momentum. "
        "Det er en *proxy* for hvad der er stærkt lige nu – ikke en forudsigelse."
    )

st.caption("Data: Stooq (gratis dagsdata). Nyheder: Google News links. Ikke finansiel rådgivning.")
