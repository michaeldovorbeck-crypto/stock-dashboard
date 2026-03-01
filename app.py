import math
from io import StringIO
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =============================
# App config
# =============================
st.set_page_config(
    page_title="Gratis Aktie Dashboard (dagskurser)",
    layout="wide",
    page_icon="📈",
)

APP_TITLE = "📈 Gratis Aktie Dashboard (dagskurser)"
DATA_DIR = "data/universes"
DEFAULT_TOPN = 10


# =============================
# CSV helpers
# =============================
def safe_read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        return pd.DataFrame()


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sikrer at df indeholder: ticker, name, sector, country (og ignorerer resten)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    df = normalize_cols(df)

    # alias mapping
    rename = {}
    for c in df.columns:
        if c in ("symbol", "sym", "code"):
            rename[c] = "ticker"
        if c in ("company", "companyname", "instrument", "security"):
            rename[c] = "name"
        if c in ("industry",):
            rename[c] = "sector"
    if rename:
        df = df.rename(columns=rename)

    if "ticker" not in df.columns:
        # gæt første kolonne er ticker
        df = df.rename(columns={df.columns[0]: "ticker"})

    for col in ("name", "sector", "country"):
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name", "sector", "country"]].reset_index(drop=True)


def load_universe_csv(path: str) -> tuple[pd.DataFrame, str]:
    """
    Returnerer (df, err). Err er "" hvis OK.
    """
    try:
        df = pd.read_csv(path)
        df = ensure_universe_schema(df)
        if df.empty:
            return df, f"Universe-fil er tom: {path}"
        return df, ""
    except Exception as e:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Kunne ikke læse {path}: {e}"


# =============================
# Stooq (gratis dagsdata)
# =============================
def stooq_symbol(symbol: str, market_hint: str = "") -> str:
    """
    Hvis ingen suffix og market_hint='US' -> tilføj .US
    Ellers returner som brugeren skrev (så DK tickers kan være NOVO-B.CO osv.)
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
def fetch_daily_ohlc_stooq(symbol: str, years: int = 5, market_hint: str = "") -> pd.DataFrame:
    """
    Gratis daglige OHLCV fra Stooq:
    https://stooq.com/q/d/l/?s={sym}&i=d
    Return: Date, Open, High, Low, Close, Volume
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


# =============================
# Indicators / signals
# =============================
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


def pct_change(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return (a / b - 1.0) * 100.0


def compute_signals(df: pd.DataFrame) -> dict:
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
    dd = float("nan")
    if len(close) >= window:
        peak = float(close.iloc[-window:].max())
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
# Universes (CSV-filer i repo)
# =============================
UNIVERSE_FILES = {
    "S&P 500 (US)": f"{DATA_DIR}/sp500.csv",
    "STOXX Europe 600": f"{DATA_DIR}/stoxx600.csv",
    "Nordics (DK)": f"{DATA_DIR}/nordics_dk.csv",
    "Nordics (SE)": f"{DATA_DIR}/nordics_se.csv",
    "Germany (DE)": f"{DATA_DIR}/germany_de.csv",
}


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_universe(universe_name: str) -> tuple[pd.DataFrame, str]:
    path = UNIVERSE_FILES.get(universe_name, "")
    if not path:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Ukendt univers: {universe_name}"
    return load_universe_csv(path)


# =============================
# UI
# =============================
st.title(APP_TITLE)

with st.sidebar:
    st.header("Univers")
    universe_name = st.selectbox("Vælg univers", list(UNIVERSE_FILES.keys()), index=0)

    st.divider()
    st.header("Indstillinger")
    years = st.slider("Historik (år)", 1, 10, 5, 1)
    top_n = st.slider("Top N (screening)", 5, 50, DEFAULT_TOPN, 1)
    max_screen = st.slider("Max tickers pr. screening (hastighed)", 20, 300, 120, 10)

    st.divider()
    st.markdown(
        """
**Signal (simpel forklaring)**
- **KØB / KIG NÆRMERE**: trend op + RSI ok + momentum ikke negativ
- **HOLD / AFVENT**: blandet billede
- **SÆLG / UNDGÅ**: svag trend / høj risiko

*Ikke finansiel rådgivning.*
"""
    )

tabs = st.tabs(["🔎 Søg / vælg papir", "🏁 Screening", "💼 Portefølje"])


# =============================
# TAB 1: Search
# =============================
with tabs[0]:
    st.subheader("🔎 Søg / vælg papir")
    uni, err = get_universe(universe_name)
    if err:
        st.error(f"Kunne ikke indlæse univers: {err}")
        st.stop()
    if uni.empty:
        st.warning("Universet er tomt.")
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
        st.info("Ingen match.")
        st.stop()

    selection = st.selectbox("Vælg papir", view["display"].tolist(), index=0)

    rows = view[view["display"] == selection]
    if rows.empty:
        st.warning("Ingen valg i listen endnu. Prøv igen.")
        st.stop()

    sel = rows.iloc[0]
    ticker = str(sel["ticker"]).strip()
    name = str(sel.get("name", "")).strip()
    sector = str(sel.get("sector", "")).strip()

    market_hint = "US" if "S&P" in universe_name else ""

    st.caption(f"Valgt: **{ticker}** {('— ' + name) if name else ''}")
    if sector:
        st.caption(f"Sektor: {sector}")

    df = fetch_daily_ohlc_stooq(ticker, years=years, market_hint=market_hint)
    if df.empty:
        st.error(
            "Kunne ikke hente dagskurser via Stooq.\n\n"
            "Tip: Brug tickers som Stooq forstår (fx AAPL.US, NOVO-B.CO)."
        )
        st.stop()

    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
    chg = (last / prev - 1) * 100 if prev else 0.0

    sig = compute_signals(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Seneste close", f"{last:,.2f}")
    m2.metric("Dag %", f"{chg:.2f}%")
    m3.metric("Seneste dato", df["Date"].iloc[-1].date().isoformat())
    m4.metric("Signal", sig.get("action", "—"))

    st.markdown("#### Kurs (Close)")
    st.line_chart(df.set_index("Date")["Close"])

    st.markdown("#### OHLC (seneste 10)")
    st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

    st.markdown("#### Nyheder")
    qtxt = f"{ticker} {name}".strip()
    st.markdown(f"- Google News: {google_news_link(qtxt)}")


# =============================
# TAB 2: Screening
# =============================
with tabs[1]:
    st.subheader("🏁 Screening (Top N)")
    st.caption("Tryk **Kør screening** for at hente data og beregne RSI/momentum/trend.")

    uni2, err2 = get_universe(universe_name)
    if err2:
        st.error(err2)
        st.stop()
    if uni2.empty:
        st.warning("Universet er tomt.")
        st.stop()

    market_hint2 = "US" if "S&P" in universe_name else ""

    if st.button("Kør screening", type="primary"):
        tickers = uni2["ticker"].astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t][:max_screen]

        rows = []
        prog = st.progress(0.0)
        status = st.empty()

        for i, t in enumerate(tickers, start=1):
            status.write(f"Henter {i}/{len(tickers)}: {t}")
            df = fetch_daily_ohlc_stooq(t, years=max(2, years), market_hint=market_hint2)
            sig = compute_signals(df)
            if sig:
                r = uni2[uni2["ticker"] == t]
                name = str(r.iloc[0].get("name", "")).strip() if not r.empty else ""
                sector = str(r.iloc[0].get("sector", "")).strip() if not r.empty else ""

                rows.append(
                    {
                        "Ticker": t,
                        "Navn": name,
                        "Sektor": sector,
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
            st.warning("Ingen tickers gav brugbar data (tjek ticker-format i CSV).")
            st.stop()

        out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        top = out.head(top_n)

        st.markdown(f"### Top {top_n}")
        st.dataframe(top, use_container_width=True, hide_index=True)

        choices = top.apply(lambda r: f"{r['Ticker']} — {r['Navn']}" if r["Navn"] else r["Ticker"], axis=1).tolist()
        pick = st.selectbox("Vælg kandidat", choices)
        pick_ticker = pick.split(" — ")[0].strip()

        dfp = fetch_daily_ohlc_stooq(pick_ticker, years=years, market_hint=market_hint2)
        if dfp.empty:
            st.error("Kunne ikke hente data for valgt kandidat.")
        else:
            st.line_chart(dfp.set_index("Date")["Close"])
            st.caption(f"Nyheder: {google_news_link(pick)}")


# =============================
# TAB 3: Portfolio
# =============================
with tabs[2]:
    st.subheader("💼 Portefølje (fordeling)")
    st.caption("Upload CSV med mindst: **ticker, shares**. Valgfrit: **name, sector, country**.")

    up = st.file_uploader("Upload portefølje CSV", type=["csv"])
    if up is None:
        st.info("Upload en CSV for at se fordeling.")
        st.stop()

    try:
        raw = pd.read_csv(up)
    except Exception as e:
        st.error(f"Kunne ikke læse CSV: {e}")
        st.stop()

    raw = normalize_cols(raw)

    # map shares kolonne
    if "shares" not in raw.columns:
        if "antal" in raw.columns:
            raw = raw.rename(columns={"antal": "shares"})
        elif "qty" in raw.columns:
            raw = raw.rename(columns={"qty": "shares"})
        else:
            st.error("Din CSV mangler kolonnen **shares** (eller qty/antal).")
            st.stop()

    # schema (ticker/name/sector/country)
    base = ensure_universe_schema(raw)
    base = base.merge(raw[["ticker", "shares"]], on="ticker", how="left")

    base["shares"] = pd.to_numeric(base["shares"], errors="coerce").fillna(0.0)
    base = base[base["shares"] > 0].copy()
    if base.empty:
        st.warning("Ingen linjer med shares > 0.")
        st.stop()

    # hent priser
    price_map = {}
    with st.spinner("Henter seneste dagskurser..."):
        for t in base["ticker"].astype(str).tolist():
            hint = "US" if t.upper().endswith(".US") else ""
            df = fetch_daily_ohlc_stooq(t, years=max(2, years), market_hint=hint)
            price_map[t] = float(df["Close"].iloc[-1]) if not df.empty else np.nan

    base["last_price"] = base["ticker"].map(price_map)
    base["value"] = base["shares"] * base["last_price"]
    total = float(base["value"].sum()) if np.isfinite(base["value"].sum()) else 0.0
    base["weight_pct"] = (base["value"] / total * 100.0) if total > 0 else 0.0

    st.markdown("### Beholdning")
    show_cols = ["ticker", "name", "sector", "shares", "last_price", "value", "weight_pct"]
    st.dataframe(
        base[show_cols].sort_values("weight_pct", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Sektorfordeling")
    if base["sector"].astype(str).str.strip().eq("").all():
        st.info("Ingen sektor i CSV. Tilføj kolonnen **sector** for sektorfordeling.")
    else:
        by_sector = (
            base.assign(sector=base["sector"].astype(str).replace("", "Ukendt"))
            .groupby("sector", dropna=False)["weight_pct"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(by_sector)

    st.markdown("### Nyheder (links)")
    for _, r in base.sort_values("weight_pct", ascending=False).head(10).iterrows():
        label = f"{r['ticker']} {r.get('name','')}".strip()
        st.markdown(f"- {label}: {google_news_link(label)}")


st.caption("Data: Stooq (gratis dagsdata). Nyheder: Google News links. Ikke finansiel rådgivning.")
