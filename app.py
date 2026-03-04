import math
from datetime import datetime
from io import StringIO
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st


# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Stock Dashboard (gratis)",
    layout="wide",
    page_icon="📊",
)

APP_TITLE = "📊 Stock Dashboard (EU + US + Tema-radar) — gratis"
DATA_DIR = "data/universes"  # repo path
DEFAULT_TOPN = 10

SIGNAL_LOG_PATH = "data/signal_log.csv"  # bruges til "læring" (simpel log)


# -----------------------------
# Helpers: CSV + tickers + data
# -----------------------------
def _safe_read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        return pd.DataFrame()


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universe CSV forventes at kunne indeholde:
    - ticker (påkrævet)
    - name (valgfri)
    - sector (valgfri)
    - country (valgfri)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    df = _normalize_cols(df)

    colmap = {}
    for c in df.columns:
        if c in ("symbol", "sym", "code", "ticker_symbol", "ric"):
            colmap[c] = "ticker"
        if c in ("company", "companyname", "security", "instrument", "name"):
            colmap[c] = "name"
        if c in ("gics sector", "sector_name", "industry", "gics_sector"):
            colmap[c] = "sector"
        if c in ("country_code", "market", "region"):
            colmap[c] = "country"

    if colmap:
        df = df.rename(columns=colmap)

    if "ticker" not in df.columns:
        # prøv at gætte første kolonne som ticker
        df = df.rename(columns={df.columns[0]: "ticker"})

    for col in ("name", "sector", "country"):
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name", "sector", "country"]].reset_index(drop=True)


def _stooq_symbol(symbol: str, market_hint: str = "") -> str:
    """
    Stooq bruger ofte suffix som:
      - AAPL.US (USA)
      - SAP.DE (Tyskland)
      - NOVO-B.CO (DK)
      - SAAB-B.ST (SE)

    Hvis symbol allerede indeholder '.', returneres det som det er.
    Ellers tilføjes suffix ud fra market_hint (US/DE/DK/SE/FI/NO/FR/UK).
    """
    s = (symbol or "").strip()
    if not s:
        return s
    if "." in s:
        return s

    mh = (market_hint or "").upper().strip()
    suffix_map = {
        "US": ".US",
        "DE": ".DE",
        "DK": ".CO",
        "SE": ".ST",
        "FI": ".HE",
        "NO": ".OL",
        "FR": ".PA",
        "UK": ".L",
    }
    if mh in suffix_map:
        return f"{s}{suffix_map[mh]}"

    return s


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_daily_ohlc_stooq(symbol: str, years: int = 5, market_hint: str = "") -> pd.DataFrame:
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


def slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    period: 1D/1W/1M/3M/6M/YTD/3Y/5Y/10Y/MAX
    """
    if df is None or df.empty:
        return df
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")

    last = d["Date"].max()
    if period == "MAX":
        return d

    if period == "1D":
        cutoff = last - pd.Timedelta(days=2)
    elif period == "1W":
        cutoff = last - pd.Timedelta(days=7)
    elif period == "1M":
        cutoff = last - pd.Timedelta(days=30)
    elif period == "3M":
        cutoff = last - pd.Timedelta(days=90)
    elif period == "6M":
        cutoff = last - pd.Timedelta(days=180)
    elif period == "YTD":
        cutoff = pd.Timestamp(year=last.year, month=1, day=1)
    elif period == "3Y":
        cutoff = last - pd.Timedelta(days=int(365.25 * 3))
    elif period == "5Y":
        cutoff = last - pd.Timedelta(days=int(365.25 * 5))
    elif period == "10Y":
        cutoff = last - pd.Timedelta(days=int(365.25 * 10))
    else:
        cutoff = last - pd.Timedelta(days=365)

    return d[d["Date"] >= cutoff].reset_index(drop=True)


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

    action = "HOLD / AFVENT"
    buy_zone = trend_up and (not np.isnan(rsi14)) and (35 <= rsi14 <= 60) and (not np.isnan(mom20)) and (mom20 >= 0)
    sell_zone = (not trend_up and (not np.isnan(rsi14)) and (rsi14 < 40)) or (risk in ("Meget høj",))
    if sell_zone:
        action = "SÆLG / UNDGÅ"
    elif buy_zone:
        action = "KØB / KIG NÆRMERE"

    why = []
    if trend_up:
        why.append("Trend op (MA50>MA200)")
    if not np.isnan(rsi14):
        why.append(f"RSI {rsi14:.0f}")
    if not np.isnan(mom20):
        why.append(f"Momentum 20d {mom20:.1f}%")

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


def load_universe_csv(path: str) -> tuple[pd.DataFrame, str]:
    try:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Filen findes ikke: {path}"

        # hvis filen er tom (0 bytes)
        if os.path.getsize(path) == 0:
            return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Filen er tom: {path}"

        df = pd.read_csv(path)
        df = _ensure_universe_schema(df)
        if df.empty:
            return df, f"Universet er tomt efter indlæsning: {path} (mangler data eller 'ticker')"
        return df, ""
    except Exception as e:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"]), f"Kunne ikke læse {path}: {e}"


# -----------------------------
# Universe builder (Wikipedia) - gratis stort univers (index-baseret)
# Kræver: lxml i requirements.txt
# -----------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def build_universe_from_wikipedia(kind: str) -> pd.DataFrame:
    """
    kind: SP500, NASDAQ100, DAX, CAC40
    Return: ticker,name,sector,country (sector kan være tom afhængigt af tabel)
    """
    if kind == "SP500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        market_hint = "US"
        tbl = pd.read_html(url, attrs={"id": "constituents"})[0]
        # columns: Symbol, Security, GICS Sector ...
        df = pd.DataFrame({
            "ticker": tbl["Symbol"].astype(str).str.strip().map(lambda x: _stooq_symbol(x, "US")),
            "name": tbl["Security"].astype(str),
            "sector": tbl.get("GICS Sector", "").astype(str) if "GICS Sector" in tbl.columns else "",
            "country": "US",
        })
        return _ensure_universe_schema(df)

    if kind == "NASDAQ100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tbl = pd.read_html(url)[0]  # "Current components"
        df = pd.DataFrame({
            "ticker": tbl["Ticker"].astype(str).str.strip().map(lambda x: _stooq_symbol(x, "US")),
            "name": tbl["Company"].astype(str),
            "sector": "",
            "country": "US",
        })
        return _ensure_universe_schema(df)

    if kind == "DAX":
        url = "https://en.wikipedia.org/wiki/DAX"
        tbl = pd.read_html(url)[0]
        # Nogle gange hedder ticker-kolonnen bare "Ticker"
        ticker_col = "Ticker" if "Ticker" in tbl.columns else tbl.columns[0]
        name_col = "Company" if "Company" in tbl.columns else tbl.columns[2]
        df = pd.DataFrame({
            "ticker": tbl[ticker_col].astype(str).str.strip().map(lambda x: _stooq_symbol(x, "DE")),
            "name": tbl[name_col].astype(str),
            "sector": "",
            "country": "DE",
        })
        # Ryd op: Wikipedia kan give tomme ticker-felter pga. logo-kolonner
        df["ticker"] = df["ticker"].astype(str).str.replace(r"\s+", "", regex=True)
        df = df[df["ticker"].str.len() > 3]
        return _ensure_universe_schema(df)

    if kind == "CAC40":
        url = "https://en.wikipedia.org/wiki/CAC_40"
        tbls = pd.read_html(url)
        # første tabel plejer at være components
        tbl = tbls[0]
        # typisk kolonner: "Ticker" / "Company" eller "Symbol"
        tcol = "Ticker" if "Ticker" in tbl.columns else ("Symbol" if "Symbol" in tbl.columns else tbl.columns[0])
        ncol = "Company" if "Company" in tbl.columns else ("Name" if "Name" in tbl.columns else tbl.columns[1])
        df = pd.DataFrame({
            "ticker": tbl[tcol].astype(str).str.strip().map(lambda x: _stooq_symbol(x, "FR")),
            "name": tbl[ncol].astype(str),
            "sector": "",
            "country": "FR",
        })
        df = df[df["ticker"].astype(str).str.len() > 3]
        return _ensure_universe_schema(df)

    return pd.DataFrame(columns=["ticker", "name", "sector", "country"])


def append_signal_log(ticker: str, signal: str, score: float, rsi: float, mom20: float, last: float):
    """
    Skriv en simpel log til CSV, så vi kan måle "ramte vi rigtigt?"
    (På Streamlit Cloud er filsystemet ikke en sikker database, men loggen er nyttig i Codespaces.
     Senere kan vi flytte log til GitHub (commit) eller en lille DB.)
    """
    os.makedirs("data", exist_ok=True)
    row = {
        "ts": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "signal": signal,
        "score": score,
        "rsi": rsi,
        "mom20": mom20,
        "last": last,
    }
    df = pd.DataFrame([row])
    if os.path.exists(SIGNAL_LOG_PATH) and os.path.getsize(SIGNAL_LOG_PATH) > 0:
        try:
            old = pd.read_csv(SIGNAL_LOG_PATH)
            out = pd.concat([old, df], ignore_index=True)
        except Exception:
            out = df
    else:
        out = df
    out.to_csv(SIGNAL_LOG_PATH, index=False)


# -----------------------------
# Sidebar
# -----------------------------
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Indstillinger")
    top_n = st.slider("Top N (screening)", 5, 50, DEFAULT_TOPN, 1)
    years = st.slider("Historik (år)", 1, 10, 5, 1)
    max_screen = st.slider("Max tickers pr. screening (for hastighed)", 20, 300, 120, 10)

    st.divider()
    st.subheader("📌 Hjælp (dansk)")
    st.markdown(
        """
**Hvad ser du?**
- **Søg/valg papir**: vælg aktie/ETF og se pris + perioder + signal + nyheder.
- **Screening**: scorer tickers ud fra trend (MA50/MA200), RSI og momentum.
- **Portefølje**: upload CSV med beholdninger for % fordeling + løbende signal.
- **Tema/forecast**: idé-radar (momentum-proxy).
- **Universe builder**: byg store universer gratis (index-lister) og download CSV.

**Signal-forklaring**
- **KØB / KIG NÆRMERE**: trend op + RSI i rimelig zone + momentum ikke negativt.
- **HOLD / AFVENT**: blandet billede.
- **SÆLG / UNDGÅ**: svag trend/høj risiko.
        """
    )


# -----------------------------
# Tabs
# -----------------------------
tab_search, tab_screener, tab_portfolio, tab_themes, tab_builder = st.tabs(
    ["🔎 Søg / vælg papir", "🏁 Screening (Top N)", "💼 Portefølje", "🧭 Tema/forecast", "🧰 Universe builder"]
)

# -----------------------------
# Universes
# -----------------------------
UNIVERSE_FILES = {
    "S&P 500 (US)": f"{DATA_DIR}/sp500.csv",
    "STOXX Europe 600": f"{DATA_DIR}/stoxx600.csv",
    "Nordics (DK)": f"{DATA_DIR}/nordics_dk.csv",
    "Nordics (SE)": f"{DATA_DIR}/nordics_se.csv",
    "Germany (DE)": f"{DATA_DIR}/germany_de.csv",
}


def get_universe(name: str) -> pd.DataFrame:
    path = UNIVERSE_FILES.get(name, "")
    df, err = load_universe_csv(path)
    if err:
        st.error(f"Kunne ikke indlæse univers: {err}")
    return df


def market_hint_from_universe(universe_name: str) -> str:
    if "S&P" in universe_name:
        return "US"
    if "Germany" in universe_name:
        return "DE"
    if "Nordics (DK)" in universe_name:
        return "DK"
    if "Nordics (SE)" in universe_name:
        return "SE"
    return ""


# -----------------------------
# TAB: Search / chart
# -----------------------------
with tab_search:
    st.subheader("🔎 Søg / vælg papir")

    c1, c2 = st.columns([1, 2])

    with c1:
        universe_name = st.selectbox("Vælg univers", list(UNIVERSE_FILES.keys()), index=0)
        uni = get_universe(universe_name)

        if uni.empty:
            st.warning("Universet er tomt. Udfyld CSV i data/universes (ticker,name,sector,country).")
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

        market_hint = market_hint_from_universe(universe_name)

        st.caption(f"Valgt: **{ticker}** {('— ' + name) if name else ''}")
        if sector:
            st.caption(f"Sektor: {sector}")

        period = st.selectbox(
            "Vis periode",
            ["1D", "1W", "1M", "3M", "6M", "YTD", "3Y", "5Y", "10Y", "MAX"],
            index=3,
        )

    with c2:
        df = fetch_daily_ohlc_stooq(ticker, years=max(2, years), market_hint=market_hint)
        if df.empty:
            st.error(
                "Kunne ikke hente dagskurser for denne ticker via Stooq.\n\n"
                "Tip: Brug tickers som Stooq forstår (fx AAPL.US, SAP.DE, NOVO-B.CO)."
            )
            st.stop()

        # periode-slice
        dfp = slice_period(df, period)

        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
        chg = (last / prev - 1) * 100 if prev else 0

        sig = compute_signals(df)
        if sig:
            append_signal_log(ticker, sig.get("action", ""), sig.get("score", np.nan), sig.get("rsi", np.nan), sig.get("mom20", np.nan), sig.get("last", np.nan))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Seneste close", f"{last:,.2f}")
        m2.metric("Dag %", f"{chg:.2f}%")
        m3.metric("Seneste dato", df["Date"].iloc[-1].date().isoformat())
        m4.metric("Signal", sig.get("action", "—") if sig else "—")

        st.markdown(f"#### Kurs (Close) — {period}")
        st.line_chart(dfp.set_index("Date")["Close"])

        st.markdown("#### Signal (forklaring)")
        if sig:
            st.write(f"**{sig['action']}**  | Score: **{sig['score']}**  | Risiko: **{sig['risk']}**")
            if sig.get("why"):
                st.caption(sig["why"])
        else:
            st.info("Ikke nok historik til signal (kræver typisk 80+ dage).")

        st.markdown("#### OHLC (tabel, seneste 10)")
        st.dataframe(dfp.tail(10), use_container_width=True, hide_index=True)

        st.markdown("#### Nyheder (følger valgt papir)")
        qtxt = f"{ticker} {name}".strip()
        st.markdown(f"- Google News: {google_news_link(qtxt)}")
        if name:
            st.markdown(f"- Google News (navn): {google_news_link(name)}")

        st.markdown("#### Læring (simpel log)")
        if os.path.exists(SIGNAL_LOG_PATH) and os.path.getsize(SIGNAL_LOG_PATH) > 0:
            try:
                log = pd.read_csv(SIGNAL_LOG_PATH).tail(20)
                st.dataframe(log, use_container_width=True, hide_index=True)
                st.caption("Næste step: vi beregner 'ramte vi rigtigt?' ved at måle efter 5/20/60 dage.")
            except Exception:
                st.caption("Kunne ikke læse signal_log.csv (ikke kritisk).")
        else:
            st.caption("Ingen signal-log endnu (oprettes automatisk når du åbner et papir).")


# -----------------------------
# TAB: Screener
# -----------------------------
with tab_screener:
    st.subheader("🏁 Screening (Top N) — RSI / momentum / trend")
    universe_name2 = st.selectbox("Vælg univers til screening", list(UNIVERSE_FILES.keys()), index=0, key="uni2")
    uni2 = get_universe(universe_name2)
    market_hint2 = market_hint_from_universe(universe_name2)

    if uni2.empty:
        st.warning("Universet er tomt. Tjek CSV.")
        st.stop()

    st.caption(
        "Tryk **Kør screening** for at hente dagskurser og beregne signaler. "
        "Vi begrænser antal tickers for hastighed."
    )

    if st.button("Kør screening", type="primary"):
        tickers = uni2["ticker"].astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t]
        tickers = tickers[: max_screen]

        rows = []
        prog = st.progress(0)
        status = st.empty()

        for i, t in enumerate(tickers, start=1):
            status.write(f"Henter {i}/{len(tickers)}: {t}")
            dfx = fetch_daily_ohlc_stooq(t, years=max(2, years), market_hint=market_hint2)
            sig = compute_signals(dfx)
            if sig:
                r = uni2[uni2["ticker"] == t]
                name = str(r["name"].iloc[0]).strip() if not r.empty else ""
                sector = str(r["sector"].iloc[0]).strip() if not r.empty else ""

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
            st.warning("Ingen tickers gav brugbar data. Tjek tickers (Stooq-format) i dit univers.")
            st.stop()

        out = pd.DataFrame(rows).sort_values(["Score"], ascending=False).reset_index(drop=True)
        top = out.head(top_n)

        st.markdown(f"### Top {top_n} kandidater")
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("### Vælg kandidat og se chart")
        choices = top.apply(lambda r: f"{r['Ticker']} — {r['Navn']}" if r["Navn"] else r["Ticker"], axis=1).tolist()
        pick = st.selectbox("Kandidat", choices)
        pick_ticker = pick.split(" — ")[0].strip()

        dfp = fetch_daily_ohlc_stooq(pick_ticker, years=max(2, years), market_hint=market_hint2)
        if dfp.empty:
            st.error("Kunne ikke hente data for valgt kandidat.")
        else:
            st.line_chart(dfp.set_index("Date")["Close"])
            st.caption(f"Nyheder: {google_news_link(pick)}")


# -----------------------------
# TAB: Portfolio (dynamisk – første version)
# -----------------------------
with tab_portfolio:
    st.subheader("💼 Portefølje (dynamisk)")

    st.caption(
        "Første gratis version: upload CSV (ticker,shares, optional name/sector/country). "
        "Næste step: gem portefølje i session + knap 'Tilføj papir' (uden upload)."
    )

    up = st.file_uploader("Upload portefølje CSV", type=["csv"])
    if up is None:
        st.info("Upload en CSV for at se portefølje og sektorfordeling.")
        st.stop()

    try:
        raw = pd.read_csv(up)
    except Exception as e:
        st.error(f"Kunne ikke læse CSV: {e}")
        st.stop()

    raw = _normalize_cols(raw)
    if "ticker" not in raw.columns:
        st.error("CSV skal have kolonnen 'ticker'.")
        st.stop()

    if "shares" not in raw.columns:
        if "antal" in raw.columns:
            raw = raw.rename(columns={"antal": "shares"})
        if "qty" in raw.columns:
            raw = raw.rename(columns={"qty": "shares"})
    if "shares" not in raw.columns:
        st.error("CSV mangler 'shares' (eller qty/antal).")
        st.stop()

    for col in ("name", "sector", "country"):
        if col not in raw.columns:
            raw[col] = ""

    pdf = raw[["ticker", "shares", "name", "sector", "country"]].copy()
    pdf["ticker"] = pdf["ticker"].astype(str).str.strip()
    pdf["shares"] = pd.to_numeric(pdf["shares"], errors="coerce").fillna(0.0)
    pdf = pdf[pdf["shares"] > 0].copy()
    if pdf.empty:
        st.warning("Ingen linjer med shares > 0.")
        st.stop()

    price_map = {}
    sig_map = {}
    why_map = {}

    with st.spinner("Henter seneste dagskurser (gratis) ..."):
        for t in pdf["ticker"].tolist():
            mh = "US" if t.upper().endswith(".US") else ""
            dfx = fetch_daily_ohlc_stooq(t, years=max(2, years), market_hint=mh)
            if dfx.empty:
                price_map[t] = np.nan
                sig_map[t] = "—"
                why_map[t] = "Ingen data"
            else:
                price_map[t] = float(dfx["Close"].iloc[-1])
                sig = compute_signals(dfx)
                sig_map[t] = sig.get("action", "—") if sig else "—"
                why_map[t] = sig.get("why", "") if sig else ""

    pdf["last_price"] = pdf["ticker"].map(price_map)
    pdf["value"] = pdf["shares"] * pdf["last_price"]
    total = float(pdf["value"].sum()) if np.isfinite(pdf["value"].sum()) else 0.0
    pdf["weight_pct"] = (pdf["value"] / total * 100.0) if total > 0 else 0.0
    pdf["signal"] = pdf["ticker"].map(sig_map)
    pdf["why"] = pdf["ticker"].map(why_map)

    st.markdown("### Beholdning + løbende vurdering")
    show_cols = ["ticker", "name", "sector", "shares", "last_price", "value", "weight_pct", "signal", "why"]
    st.dataframe(
        pdf[show_cols].sort_values("weight_pct", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Sektorfordeling")
    if pdf["sector"].astype(str).str.strip().eq("").all():
        st.info("Ingen sektor angivet i CSV. Tilføj 'sector' for korrekt sektorfordeling.")
    else:
        by_sector = (
            pdf.assign(sector=pdf["sector"].astype(str).replace("", "Ukendt"))
            .groupby("sector", dropna=False)["weight_pct"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(by_sector)

    st.markdown("### Nyheder (følger porteføljen)")
    for _, r in pdf.sort_values("weight_pct", ascending=False).head(10).iterrows():
        label = f"{r['ticker']} {r.get('name','')}".strip()
        st.markdown(f"- {label}: {google_news_link(label)}")


# -----------------------------
# TAB: Themes / forecast (Stooq som datakilde) - version 1
# -----------------------------
with tab_themes:
    st.subheader("🧭 Tema/forecast (momentum-proxy via ETF’er / tickers)")
    st.caption("Dette er en teknisk momentum-indikator (ikke rådgivning).")

    # Temaer (du kan ændre tickers – de skal være Stooq-format)
    themes = {
        "AI & Software": ["QQQ.US", "MSFT.US", "NVDA.US"],
        "Semiconductors": ["SOXX.US", "NVDA.US", "AMD.US"],
        "Elektrificering & batterier": ["LIT.US", "TSLA.US"],
        "Solenergi": ["TAN.US", "FSLR.US"],
        "Defense/Aerospace": ["ITA.US", "LMT.US", "RTX.US"],
        "Robotics/Automation": ["BOTZ.US", "ROK.US"],
        "Cybersecurity": ["HACK.US", "PANW.US"],
        "Rum / Space": ["ARKX.US", "RKLB.US"],
        "Grøn energi": ["ICLN.US", "ENPH.US"],
    }

    bench = st.text_input("Benchmark (relativ styrke)", value="SPY.US")

    def rel_strength(asset: str, bench: str, days: int) -> float:
        da = fetch_daily_ohlc_stooq(asset, years=10, market_hint="US")
        db = fetch_daily_ohlc_stooq(bench, years=10, market_hint="US")
        if da.empty or db.empty:
            return float("nan")
        da = da.set_index("Date")["Close"].astype(float)
        db = db.set_index("Date")["Close"].astype(float)
        common = da.index.intersection(db.index)
        da = da.loc[common]
        db = db.loc[common]
        if len(da) < days + 5:
            return float("nan")
        ra = da.iloc[-1] / da.iloc[-days]
        rb = db.iloc[-1] / db.iloc[-days]
        return float(ra / rb - 1.0)  # relativ outperformance

    rows = []
    for theme, tickers in themes.items():
        # brug første ticker som "tema-proxy" (du kan også lave gennemsnit senere)
        proxy = tickers[0]
        rs_1m = rel_strength(proxy, bench, days=21)
        rs_3m = rel_strength(proxy, bench, days=63)

        score = 0.0
        if not np.isnan(rs_1m):
            score += rs_1m * 100
        if not np.isnan(rs_3m):
            score += rs_3m * 100

        rows.append(
            {
                "Tema": theme,
                "Ticker": proxy,
                "MomentumScore": round(score, 4),
                "RS_1M_vs_SPY": round(rs_1m, 4) if not np.isnan(rs_1m) else np.nan,
                "RS_3M_vs_SPY": round(rs_3m, 4) if not np.isnan(rs_3m) else np.nan,
            }
        )

    df_theme = pd.DataFrame(rows).sort_values("MomentumScore", ascending=False).reset_index(drop=True)
    st.dataframe(df_theme, use_container_width=True, hide_index=True)

    st.markdown("🔥 Temaer at kigge nærmere på (stærk relativ styrke)")
    for _, r in df_theme.head(6).iterrows():
        st.write(f"- **{r['Tema']}** ({r['Ticker']}) — RS 1M: {r['RS_1M_vs_SPY']}, RS 3M: {r['RS_3M_vs_SPY']}")

    st.caption("Næste step: forecast (3/5/10 år) bør være scenarie-baseret (trend+nyheder), ikke 'garanteret' model.")


# -----------------------------
# TAB: Universe builder
# -----------------------------
with tab_builder:
    st.subheader("🧰 Auto-universe builder (gratis)")

    st.write(
        "Byg et stort univers automatisk (index-baseret) og download som CSV.\n\n"
        "1) Vælg univers-type\n"
        "2) Klik 'Byg'\n"
        "3) Download CSV\n"
        "4) Læg filen i repo: `data/universes/` og commit+push"
    )

    kind = st.selectbox("Vælg univers-type", ["SP500", "NASDAQ100", "DAX", "CAC40"])
    out_name = st.text_input("Gem som filnavn (i data/universes)", value=f"{kind.lower()}.csv")

    if st.button("Byg univers", type="primary"):
        try:
            dfu = build_universe_from_wikipedia(kind)
            if dfu.empty:
                st.error("Builder gav tomt resultat. Prøv igen.")
            else:
                st.success(f"Bygget: {len(dfu)} tickers")
                st.dataframe(dfu.head(50), use_container_width=True, hide_index=True)

                csv_bytes = dfu.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_bytes,
                    file_name=out_name,
                    mime="text/csv",
                )

                st.info(
                    "Når du har downloadet:\n"
                    "- Upload filen til GitHub: `data/universes/`\n"
                    "- Commit + Push\n"
                    "- Streamlit Cloud vil derefter kunne bruge universet"
                )
        except Exception as e:
            st.error(f"Kunne ikke bygge univers: {e}")


# Footer
st.caption("Data: Stooq (gratis dagsdata). Nyheder: Google News links. Ikke finansiel rådgivning.")
