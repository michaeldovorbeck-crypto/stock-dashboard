import os
import math
import time
from datetime import datetime, date
from io import StringIO
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# App config
# =========================
st.set_page_config(page_title="Gratis Aktie Dashboard (dagskurser)", layout="wide", page_icon="📈")
APP_TITLE = "📈 Gratis Aktie Dashboard (dagskurser)"
DATA_DIR = "data/universes"
DEFAULT_TOPN = 10

# Cache/timeout
REQ_TIMEOUT = 20
STOOQ_TTL_SECONDS = 15 * 60  # 15 min cache på dagsdata


# =========================
# Utilities
# =========================
def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text))
    except Exception:
        # prøv semikolon
        try:
            return pd.read_csv(StringIO(text), sep=";")
        except Exception:
            return pd.DataFrame()


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _ensure_universe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forventede kolonner:
    ticker (påkrævet), name, sector, country (valgfri)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    df = _normalize_cols(df)

    # aliaser
    rename = {}
    for c in df.columns:
        if c in ("symbol", "sym", "code"):
            rename[c] = "ticker"
        if c in ("company", "companyname", "instrument", "security", "navn"):
            rename[c] = "name"
        if c in ("sektor",):
            rename[c] = "sector"
        if c in ("land",):
            rename[c] = "country"
    if rename:
        df = df.rename(columns=rename)

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
    Stooq tickers er ofte som:
      AAPL.US  (USA)
      NOVO-B.CO (DK)
      SAAB-B.ST (SE)
      SAP.DE    (DE)  (varierer)
    """
    s = (symbol or "").strip()
    if not s:
        return s
    if "." in s:
        return s
    if market_hint.upper() == "US":
        return f"{s}.US"
    return s


@st.cache_data(ttl=STOOQ_TTL_SECONDS, show_spinner=False)
def fetch_daily_ohlc_stooq(symbol: str, years: int = 10, market_hint: str = "") -> pd.DataFrame:
    """
    Gratis dagsdata fra Stooq:
      https://stooq.com/q/d/l/?s={sym}&i=d
    Return: Date, Open, High, Low, Close, Volume
    """
    sym = _stooq_symbol(symbol, market_hint=market_hint).lower()
    if not sym:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=REQ_TIMEOUT)
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

    # Cut to last N years
    if years and years > 0 and not df.empty:
        cutoff = df["Date"].max() - pd.Timedelta(days=int(365.25 * years))
        df = df[df["Date"] >= cutoff]

    return df.reset_index(drop=True)


def google_news_link(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/search?q={q}&hl=da&gl=DK&ceid=DK%3Ada"


def stooq_rss_link(symbol: str) -> str:
    # Stooq har RSS for mange tickers (ikke alle), men linket er billigt at vise:
    sym = (symbol or "").strip().lower()
    return f"https://stooq.com/q/?s={sym}&c=1d&t=l&a=lg&b=0"


# =========================
# Indicators + labels
# =========================
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
    Input: OHLC df
    Output: signal dict
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

    # Score
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

    # Labels
    buy_zone = trend_up and (not np.isnan(rsi14)) and (35 <= rsi14 <= 60) and (not np.isnan(mom20)) and (mom20 >= 0)
    sell_zone = (not trend_up and (not np.isnan(rsi14)) and (rsi14 < 40)) or (risk in ("Meget høj",))

    if sell_zone:
        action = "SÆLG / UNDGÅ"
    elif buy_zone:
        action = "KØB / KIG NÆRMERE"
    else:
        action = "HOLD / AFVENT"

    why = []
    why.append("Trend op (MA50>MA200)" if trend_up else "Trend ikke op (MA50<=MA200)")
    if not np.isnan(rsi14):
        why.append(f"RSI {rsi14:.0f}")
    if not np.isnan(mom20):
        why.append(f"Momentum 20d {mom20:.1f}%")
    if risk != "OK":
        why.append(f"Risiko: {risk}")

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


def period_slice(df: pd.DataFrame, period_key: str) -> pd.DataFrame:
    """
    Returnerer subset til perioden:
    1d, 1w, 1m, 3m, 6m, ytd, 3y, 5y, 10y, max
    """
    if df is None or df.empty:
        return df
    d = df.copy()
    d = d.sort_values("Date")
    end = d["Date"].max()
    if period_key == "max":
        return d
    if period_key == "1d":
        # sidste ~2 uger for at se “dagligt” zoom
        cutoff = end - pd.Timedelta(days=14)
    elif period_key == "1w":
        cutoff = end - pd.Timedelta(days=7 * 8)
    elif period_key == "1m":
        cutoff = end - pd.Timedelta(days=31)
    elif period_key == "3m":
        cutoff = end - pd.Timedelta(days=31 * 3)
    elif period_key == "6m":
        cutoff = end - pd.Timedelta(days=31 * 6)
    elif period_key == "ytd":
        cutoff = pd.Timestamp(date(end.year, 1, 1))
    elif period_key == "3y":
        cutoff = end - pd.Timedelta(days=int(365.25 * 3))
    elif period_key == "5y":
        cutoff = end - pd.Timedelta(days=int(365.25 * 5))
    elif period_key == "10y":
        cutoff = end - pd.Timedelta(days=int(365.25 * 10))
    else:
        cutoff = end - pd.Timedelta(days=365)

    return d[d["Date"] >= cutoff].reset_index(drop=True)


def period_return(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Close" not in df.columns:
        return float("nan")
    c = df["Close"].astype(float).dropna()
    if len(c) < 2:
        return float("nan")
    return (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100.0


# =========================
# Learning log (best-effort)
# =========================
LOG_PATH = "data/signal_log.csv"


def append_signal_log(ticker: str, name: str, action: str, score: float, last: float) -> None:
    """
    Gemmer log lokalt. På Streamlit Cloud kan filsystem være midlertidigt,
    men vi forsøger (best-effort).
    """
    try:
        os.makedirs("data", exist_ok=True)
        row = {
            "ts": _now_ts(),
            "ticker": ticker,
            "name": name,
            "action": action,
            "score": score,
            "last": last,
        }
        df_row = pd.DataFrame([row])
        if os.path.exists(LOG_PATH):
            df_old = pd.read_csv(LOG_PATH)
            df_out = pd.concat([df_old, df_row], ignore_index=True)
        else:
            df_out = df_row
        df_out.to_csv(LOG_PATH, index=False)
    except Exception:
        pass


def load_signal_log() -> pd.DataFrame:
    try:
        if os.path.exists(LOG_PATH):
            return pd.read_csv(LOG_PATH)
    except Exception:
        pass
    return pd.DataFrame(columns=["ts", "ticker", "name", "action", "score", "last"])


# =========================
# Universe loading
# =========================
UNIVERSE_FILES = {
    "S&P 500 (US)": f"{DATA_DIR}/sp500.csv",
    "STOXX Europe 600": f"{DATA_DIR}/stoxx600.csv",
    "Nordics (DK)": f"{DATA_DIR}/nordics_dk.csv",
    "Nordics (SE)": f"{DATA_DIR}/nordics_se.csv",
    "Germany (DE)": f"{DATA_DIR}/germany_de.csv",
}


def load_universe_csv(path: str) -> Tuple[pd.DataFrame, str]:
    try:
        if not os.path.exists(path):
            return pd.DataFrame(), f"Fil findes ikke: {path}"
        # hvis fil er tom
        if os.path.getsize(path) == 0:
            return pd.DataFrame(), f"CSV er tom: {path}"
        df = pd.read_csv(path)
        df = _ensure_universe_schema(df)
        if df.empty:
            return df, f"Ingen rækker i univers: {path}"
        return df, ""
    except Exception as e:
        return pd.DataFrame(), f"Kunne ikke læse {path}: {e}"


def get_universe(name: str, uploaded_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, str]:
    if uploaded_df is not None and not uploaded_df.empty:
        return _ensure_universe_schema(uploaded_df), ""
    path = UNIVERSE_FILES.get(name, "")
    return load_universe_csv(path)


# =========================
# Session: Dynamic portfolio
# =========================
def init_portfolio_state():
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=["ticker", "shares", "name", "sector", "country"])


def add_position(ticker: str, shares: float, name: str = "", sector: str = "", country: str = ""):
    init_portfolio_state()
    df = st.session_state.portfolio.copy()
    ticker = (ticker or "").strip()
    if not ticker or shares <= 0:
        return
    # hvis ticker findes, læg til
    if (df["ticker"] == ticker).any():
        df.loc[df["ticker"] == ticker, "shares"] = df.loc[df["ticker"] == ticker, "shares"].astype(float) + float(shares)
        # opdater meta hvis udfyldt
        if name:
            df.loc[df["ticker"] == ticker, "name"] = name
        if sector:
            df.loc[df["ticker"] == ticker, "sector"] = sector
        if country:
            df.loc[df["ticker"] == ticker, "country"] = country
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame([{"ticker": ticker, "shares": float(shares), "name": name, "sector": sector, "country": country}]),
            ],
            ignore_index=True,
        )
    st.session_state.portfolio = df


def remove_position(ticker: str):
    init_portfolio_state()
    df = st.session_state.portfolio.copy()
    st.session_state.portfolio = df[df["ticker"] != ticker].reset_index(drop=True)


# =========================
# UI
# =========================
st.title(APP_TITLE)

with st.sidebar:
    st.header("Univers")
    universe_name = st.selectbox("Vælg univers", list(UNIVERSE_FILES.keys()), index=0)

    st.header("Indstillinger")
    years = st.slider("Historik (år)", 1, 10, 5, 1)
    top_n = st.slider("Top N (screening)", 5, 50, DEFAULT_TOPN, 1)
    max_screen = st.slider("Max tickers pr. screening (hastighed)", 20, 300, 120, 10)

    st.divider()
    st.subheader("Upload univers (hvis repo-CSV er tom)")
    st.caption("Hvis dine `data/universes/*.csv` er tomme, kan du uploade en univers-CSV her for at teste.")
    up_uni = st.file_uploader("Upload universe CSV", type=["csv"], key="upload_universe")

    st.divider()
    st.subheader("Signal (simpel forklaring)")
    st.markdown(
        """
- **KØB / KIG NÆRMERE**: trend op + RSI “sund” zone + momentum ikke negativ  
- **HOLD / AFVENT**: blandet billede  
- **SÆLG / UNDGÅ**: svag trend og/eller høj risiko  

*Ikke finansiel rådgivning.*
        """
    )


uploaded_universe_df = None
if up_uni is not None:
    try:
        uploaded_universe_df = pd.read_csv(up_uni)
    except Exception:
        uploaded_universe_df = None

tabs = st.tabs(["🔎 Søg / vælg papir", "🏁 Screening", "💼 Portefølje", "🧭 Tema/forecast", "📚 Læring/log"])


# =========================
# TAB 1: Search / choose paper
# =========================
with tabs[0]:
    st.subheader("🔎 Søg / vælg papir")

    uni, err = get_universe(universe_name, uploaded_df=uploaded_universe_df)
    if err:
        st.error(f"Kunne ikke indlæse univers: {err}")
        st.stop()

    uni = uni.copy()
    uni["display"] = uni.apply(
        lambda r: f"{r['ticker']} — {r['name']}" if str(r.get("name", "")).strip() else f"{r['ticker']}",
        axis=1,
    )

    c1, c2 = st.columns([1, 2])

    with c1:
        query = st.text_input("Søg navn eller ticker", "")
        view = uni
        if query.strip():
            q = query.strip().lower()
            view = view[
                view["ticker"].astype(str).str.lower().str.contains(q, na=False)
                | view["name"].astype(str).str.lower().str.contains(q, na=False)
            ]

        if view.empty:
            st.info("Ingen match. Prøv en anden søgning eller upload et univers.")
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

        st.divider()
        st.markdown("**Kursperiode**")
        period_labels = {
            "1d": "1 dag",
            "1w": "1 uge",
            "1m": "1 måned",
            "3m": "3 mdr",
            "6m": "6 mdr",
            "ytd": "i år",
            "3y": "3 år",
            "5y": "5 år",
            "10y": "10 år",
            "max": "max",
        }
        period_key = st.radio(
            " ",
            list(period_labels.keys()),
            format_func=lambda k: period_labels[k],
            horizontal=True,
            index=5,  # ytd som default
        )

    with c2:
        df = fetch_daily_ohlc_stooq(ticker, years=max(1, years), market_hint=market_hint)
        if df.empty:
            st.error(
                "Kunne ikke hente dagskurser fra Stooq for denne ticker.\n\n"
                "Tip: Brug tickers som Stooq forstår (fx AAPL.US, NOVO-B.CO, SAAB-B.ST)."
            )
            st.stop()

        # period slice
        dfp = period_slice(df, period_key)

        last = float(dfp["Close"].iloc[-1])
        prev = float(dfp["Close"].iloc[-2]) if len(dfp) >= 2 else last
        chg = (last / prev - 1) * 100 if prev else 0.0

        sig = compute_signals(df)  # signal på “hele” datasættet
        append_signal_log(ticker, name, sig.get("action", ""), sig.get("score", float("nan")), sig.get("last", float("nan")))

        # periodeafkast
        pr = period_return(dfp)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Seneste close", f"{last:,.2f}")
        m2.metric("Dag %", f"{chg:.2f}%")
        m3.metric("Periode %", f"{pr:.2f}%")
        m4.metric("Seneste dato", dfp["Date"].iloc[-1].date().isoformat())
        m5.metric("Signal", sig.get("action", "—"))

        st.caption(sig.get("why", ""))

        st.markdown("#### Kurs (Close)")
        st.line_chart(dfp.set_index("Date")["Close"])

        st.markdown("#### OHLC (seneste 10)")
        st.dataframe(dfp.tail(10), use_container_width=True, hide_index=True)

        st.markdown("#### Nyheder")
        qtxt = f"{ticker} {name}".strip()
        st.markdown(f"- Google News: {google_news_link(qtxt)}")
        if name:
            st.markdown(f"- Google News (kun navn): {google_news_link(name)}")
        st.markdown(f"- Stooq side (ofte med nyheder): {stooq_rss_link(ticker)}")

        st.divider()
        st.markdown("#### Tilføj til dynamisk portefølje")
        add_cols = st.columns([2, 1, 2])
        with add_cols[0]:
            add_shares = st.number_input("Antal (shares)", min_value=0.0, value=0.0, step=1.0)
        with add_cols[1]:
            if st.button("Tilføj", type="primary"):
                add_position(ticker=ticker, shares=float(add_shares), name=name, sector=sector, country=str(sel.get("country", "")).strip())
                st.success("Tilføjet til porteføljen (session).")
        with add_cols[2]:
            st.caption("Tip: Porteføljen gemmes i denne session. Du kan eksportere/importere under fanen Portefølje.")


# =========================
# TAB 2: Screener
# =========================
with tabs[1]:
    st.subheader("🏁 Screening — Top N (RSI / momentum / trend)")
    uni2, err2 = get_universe(universe_name, uploaded_df=uploaded_universe_df)
    if err2:
        st.error(f"Kunne ikke indlæse univers: {err2}")
        st.stop()

    market_hint2 = "US" if "S&P" in universe_name else ""
    st.caption("Tryk **Kør screening**. Vi begrænser antal tickers for hastighed.")

    if st.button("Kør screening", type="primary"):
        tickers = uni2["ticker"].astype(str).str.strip().tolist()
        tickers = [t for t in tickers if t][:max_screen]

        rows = []
        prog = st.progress(0)
        status = st.empty()

        for i, t in enumerate(tickers, start=1):
            status.write(f"Henter {i}/{len(tickers)}: {t}")
            df = fetch_daily_ohlc_stooq(t, years=max(2, years), market_hint=market_hint2)
            sig = compute_signals(df)
            if sig:
                r = uni2[uni2["ticker"] == t]
                name = str(r.iloc[0]["name"]).strip() if not r.empty else ""
                sector = str(r.iloc[0]["sector"]).strip() if not r.empty else ""
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
            st.warning("Ingen tickers gav brugbar data. Tjek tickers i dit univers (Stooq-format).")
            st.stop()

        out = pd.DataFrame(rows).sort_values(["Score"], ascending=False).reset_index(drop=True)
        top = out.head(top_n)

        st.markdown(f"### Top {top_n} kandidater")
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("### Vælg kandidat og se chart + nyheder")
        choices = top.apply(lambda r: f"{r['Ticker']} — {r['Navn']}" if r["Navn"] else r["Ticker"], axis=1).tolist()
        pick = st.selectbox("Kandidat", choices)
        pick_ticker = pick.split(" — ")[0].strip()

        dfp = fetch_daily_ohlc_stooq(pick_ticker, years=max(2, years), market_hint=market_hint2)
        if dfp.empty:
            st.error("Kunne ikke hente data for valgt kandidat.")
        else:
            st.line_chart(dfp.set_index("Date")["Close"])
            st.caption(f"Nyheder: {google_news_link(pick)}")


# =========================
# TAB 3: Dynamic Portfolio
# =========================
with tabs[2]:
    st.subheader("💼 Portefølje (dynamisk) — løbende køb/hold/salg vurdering")
    init_portfolio_state()

    st.caption(
        "Her kan du løbende tilføje/fjerne positioner. "
        "Kursdata hentes gratis fra Stooq. "
        "Du kan også importere/eksportere porteføljen som CSV."
    )

    cA, cB = st.columns([2, 1])

    with cA:
        st.markdown("### Dine positioner (session)")
        if st.session_state.portfolio.empty:
            st.info("Porteføljen er tom. Tilføj fra 'Søg / vælg papir' eller via formularen herunder.")
        else:
            st.dataframe(st.session_state.portfolio, use_container_width=True, hide_index=True)

        st.markdown("### Tilføj manuelt")
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            p_ticker = st.text_input("Ticker (fx AAPL.US, NOVO-B.CO)", key="p_ticker")
        with f2:
            p_shares = st.number_input("Antal", min_value=0.0, value=0.0, step=1.0, key="p_shares")
        with f3:
            if st.button("Tilføj position", type="primary"):
                add_position(p_ticker, p_shares)
                st.success("Tilføjet.")

        st.markdown("### Import/eksport")
        exp = st.session_state.portfolio.copy()
        csv_bytes = exp.to_csv(index=False).encode("utf-8")
        st.download_button("Download portefølje CSV", data=csv_bytes, file_name="portfolio.csv", mime="text/csv")

        up_pf = st.file_uploader("Import portefølje CSV", type=["csv"], key="import_pf")
        if up_pf is not None:
            try:
                rdf = pd.read_csv(up_pf)
                rdf = _normalize_cols(rdf)
                # map kolonner
                if "ticker" not in rdf.columns and len(rdf.columns) > 0:
                    rdf = rdf.rename(columns={rdf.columns[0]: "ticker"})
                if "shares" not in rdf.columns:
                    if "antal" in rdf.columns:
                        rdf = rdf.rename(columns={"antal": "shares"})
                    if "qty" in rdf.columns:
                        rdf = rdf.rename(columns={"qty": "shares"})
                for col in ("name", "sector", "country"):
                    if col not in rdf.columns:
                        rdf[col] = ""
                rdf["shares"] = pd.to_numeric(rdf["shares"], errors="coerce").fillna(0.0)
                rdf = rdf[rdf["shares"] > 0].copy()
                st.session_state.portfolio = rdf[["ticker", "shares", "name", "sector", "country"]].copy()
                st.success("Portefølje importeret til session.")
            except Exception as e:
                st.error(f"Kunne ikke importere: {e}")

    with cB:
        st.markdown("### Fjern position")
        if not st.session_state.portfolio.empty:
            rm = st.selectbox("Vælg ticker", st.session_state.portfolio["ticker"].astype(str).tolist())
            if st.button("Fjern", type="secondary"):
                remove_position(rm)
                st.success("Fjernet.")

    # Værdi + vurdering
    if not st.session_state.portfolio.empty:
        pdf = st.session_state.portfolio.copy()
        pdf["ticker"] = pdf["ticker"].astype(str).str.strip()
        pdf["shares"] = pd.to_numeric(pdf["shares"], errors="coerce").fillna(0.0)

        price_map = {}
        sig_map = {}

        with st.spinner("Henter seneste dagskurser og beregner signaler..."):
            for t in pdf["ticker"].tolist():
                mh = "US" if t.upper().endswith(".US") else ""
                dfx = fetch_daily_ohlc_stooq(t, years=max(2, years), market_hint=mh)
                if dfx.empty:
                    price_map[t] = np.nan
                    sig_map[t] = {}
                else:
                    price_map[t] = float(dfx["Close"].iloc[-1])
                    sig_map[t] = compute_signals(dfx)

        pdf["last_price"] = pdf["ticker"].map(price_map)
        pdf["value"] = pdf["shares"] * pdf["last_price"]
        total = float(pdf["value"].sum()) if np.isfinite(pdf["value"].sum()) else 0.0
        pdf["weight_pct"] = (pdf["value"] / total * 100.0) if total > 0 else 0.0

        # signal kolonner
        pdf["signal"] = pdf["ticker"].apply(lambda t: sig_map.get(t, {}).get("action", "—"))
        pdf["score"] = pdf["ticker"].apply(lambda t: sig_map.get(t, {}).get("score", np.nan))
        pdf["forklaring"] = pdf["ticker"].apply(lambda t: sig_map.get(t, {}).get("why", ""))

        st.markdown("## Porteføljeoversigt")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total værdi (beregnet)", f"{total:,.2f}")
        m2.metric("Antal positioner", f"{len(pdf)}")
        m3.metric("Sidst opdateret", _now_ts())

        show_cols = ["ticker", "shares", "last_price", "value", "weight_pct", "signal", "score", "forklaring"]
        st.dataframe(pdf[show_cols].sort_values("weight_pct", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("### Fordeling")
        st.bar_chart(pdf.set_index("ticker")["weight_pct"].sort_values(ascending=False))

        st.markdown("### Nyheder (top 10)")
        for _, r in pdf.sort_values("weight_pct", ascending=False).head(10).iterrows():
            label = f"{r['ticker']} {r.get('name','')}".strip()
            st.markdown(f"- {label}: {google_news_link(label)}")


# =========================
# TAB 4: Themes / forecast (lightweight)
# =========================
with tabs[3]:
    st.subheader("🧭 Tema/forecast — momentum nu + 3/5/10 år (heuristik)")
    st.caption(
        "Dette er en idé-radar. Gratis data = dagskurser + simple indikatorer. "
        "‘Forecast’ er heuristik (ikke AI-prognose) og bruges som pejlemærke."
    )

    themes = {
        "AI / compute": ["NVDA.US", "MSFT.US", "GOOGL.US", "AMZN.US", "AVGO.US"],
        "Elektrificering": ["TSLA.US", "ALB.US", "SQM.US", "VWS.CO"],
        "Defense / sikkerhed": ["LMT.US", "NOC.US", "RTX.US", "SAAB-B.ST"],
        "Grøn udvikling": ["ORSTED.CO", "ENPH.US", "FSLR.US"],
        "Rumfart": ["RKLB.US", "ASTS.US"],
    }

    pick_theme = st.selectbox("Vælg tema", list(themes.keys()))
    tickers = themes[pick_theme]

    st.write("Tickers:", ", ".join(tickers))

    rows = []
    for t in tickers:
        dfx = fetch_daily_ohlc_stooq(t, years=10, market_hint=("US" if t.endswith(".US") else ""))
        sig = compute_signals(dfx)
        if not sig:
            continue

        # “momentum nu” = score + mom20
        mom_now = sig.get("mom20", np.nan)

        # pseudo “3/5/10 år” pejlemærker: bruger trend + volatilitet til en “robusthed”
        vol = sig.get("vol20", np.nan)
        robustness = np.nan
        if not np.isnan(mom_now) and not np.isnan(vol) and vol > 0:
            robustness = mom_now / vol  # højere = bedre risk-adjusted momentum

        # simple bucket labels
        def bucket(x):
            if np.isnan(x):
                return "—"
            if x > 1.2:
                return "Stærk"
            if x > 0.4:
                return "OK"
            if x > 0:
                return "Svag"
            return "Negativ"

        rows.append(
            {
                "Ticker": t,
                "Signal nu": sig.get("action", "—"),
                "Score": sig.get("score", np.nan),
                "Momentum 20d %": round(mom_now, 1) if not np.isnan(mom_now) else np.nan,
                "Robusthed": round(robustness, 2) if not np.isnan(robustness) else np.nan,
                "3 år (pejl)": bucket(robustness),
                "5 år (pejl)": bucket(robustness * 0.9 if not np.isnan(robustness) else np.nan),
                "10 år (pejl)": bucket(robustness * 0.8 if not np.isnan(robustness) else np.nan),
            }
        )

    if rows:
        out = pd.DataFrame(rows).sort_values("Score", ascending=False)
        st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        st.info("Ingen data i temaet (tjek tickers i Stooq-format).")

    st.markdown("#### Hvad betyder ‘3/5/10 år (pejl)’?")
    st.markdown(
        """
Det er **ikke** en egentlig prognose. Det er en **simpel pejling** baseret på:
- momentum nu (20 dage)
- volatilitet (20 dage)
- trend (MA50 vs MA200)

Formålet er at give et hurtigt “temperatur-tjek”.
        """
    )


# =========================
# TAB 5: Learning / log
# =========================
with tabs[4]:
    st.subheader("📚 Læring/log — ‘ramte vi?’")
    st.caption(
        "Denne log gemmer dine beregnede signaler når du kigger på tickers. "
        "På Streamlit Cloud kan log-filen blive nulstillet ved gen-deploy."
    )
    log = load_signal_log()
    if log.empty:
        st.info("Ingen log endnu. Gå til ‘Søg / vælg papir’ og kig på nogle tickers.")
    else:
        st.dataframe(log.tail(200), use_container_width=True, hide_index=True)

        st.markdown("### Hurtig ‘hit-rate’ (meget grov)")
        st.caption("Vi kan kun vurdere ‘ramte vi?’ hvis vi senere kender efterfølgende udvikling – her laver vi en simpel check på prisændring siden log.")
        # Denne del kan senere udbygges (fx re-fetch pris efter X dage). For nu: vis kun log.

st.caption("Data: Stooq (gratis dagsdata). Nyheder: Google News/Stooq links. Ikke finansiel rådgivning.")
