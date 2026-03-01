import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_sources import fetch_daily_ohlcv_stooq, load_universe_csv
from src.indicators import compute_signals, trade_label
from src.portfolio import parse_portfolio_csv, weight_by_value
from src.ui_text_da import help_text

st.set_page_config(page_title="Gratis Aktie Dashboard", layout="wide")

@st.cache_data(ttl=3600)
def cached_price(stooq_symbol: str, years: int = 5) -> pd.DataFrame:
    return fetch_daily_ohlcv_stooq(stooq_symbol, years=years)

def price_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Kurs"
    ))
    if "sma20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], mode="lines", name="SMA20"))
    if "sma50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], mode="lines", name="SMA50"))
    fig.update_layout(title=title, height=520, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def line_chart(series: pd.Series, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines"))
    fig.update_layout(title=title, height=240)
    st.plotly_chart(fig, use_container_width=True)

st.title("📈 Gratis Aktie Dashboard (dagskurser)")

# Sidebar: univers valg
st.sidebar.header("Univers")
universe_choice = st.sidebar.selectbox(
    "Vælg univers",
    [
        "S&P 500 (US)",
        "STOXX Europe 600",
        "Danmark (ekstra)",
        "Sverige (ekstra)",
        "Tyskland (ekstra)",
    ],
)

universe_map = {
    "S&P 500 (US)": "data/universes/sp500.csv",
    "STOXX Europe 600": "data/universes/stoxx600.csv",
    "Danmark (ekstra)": "data/universes/nordics_dk.csv",
    "Sverige (ekstra)": "data/universes/nordics_se.csv",
    "Tyskland (ekstra)": "data/universes/germany_de.csv",
}

try:
    uni = load_universe_csv(universe_map[universe_choice])
except Exception as e:
    st.error(f"Kunne ikke indlæse univers: {e}")
    st.stop()

# Tabs
tab_overview, tab_screener, tab_portfolio, tab_themes = st.tabs(
    ["Oversigt", "Screener (Top 10)", "Portefølje", "Temaer"]
)

# ---------- Oversigt ----------
with tab_overview:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Søg / vælg papir")
        query = st.text_input("Søg navn eller ticker", "")
        view = uni.copy()
        if query.strip():
            q = query.strip().lower()
            view = view[
                view["name"].str.lower().str.contains(q) |
                view["ticker"].str.lower().str.contains(q)
            ]

        # show with name + ticker
        view_display = view.copy()
        view_display["display"] = view_display["name"] + " (" + view_display["ticker"] + ")"
        selection = st.selectbox("Vælg", view_display["display"].tolist())
        sel_row = view_display[view_display["display"] == selection].iloc[0]

        st.write("**Ticker:**", sel_row["ticker"])
        st.write("**Stooq symbol:**", sel_row["stooq_symbol"])
        if sel_row.get("sector"):
            st.write("**Sektor:**", sel_row.get("sector", ""))
        if sel_row.get("country"):
            st.write("**Land:**", sel_row.get("country", ""))

        years = st.slider("Historik (år)", 1, 15, 5)

    with right:
        st.subheader("Kurs & indikatorer")
        df = cached_price(sel_row["stooq_symbol"], years=years)
        if df.empty:
            st.warning("Ingen data fundet for denne ticker hos Stooq. Prøv et andet papir.")
        else:
            dfi = compute_signals(df)
            latest = dfi.iloc[-1]
            label = trade_label(latest)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Seneste close", f"{latest['close']:.2f}")
            c2.metric("RSI(14)", f"{latest['rsi14']:.1f}")
            c3.metric("Momentum 20d", f"{latest['mom20']:.1f}%")
            c4.metric("Signal", label)

            price_chart(dfi, f"{sel_row['name']} ({sel_row['ticker']}) – dagskurser")
            line_chart(dfi["rsi14"].dropna(), "RSI(14)")
            line_chart(dfi["mom20"].dropna(), "Momentum (20 dage) %")

# ---------- Screener ----------
with tab_screener:
    st.subheader("Top 10 kandidater i valgt univers (dagskurser)")

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        min_rows = st.number_input("Minimum datapunkter", min_value=60, max_value=2000, value=200, step=20)
    with colB:
        only_with_sector = st.checkbox("Vis kun med sektor udfyldt", value=False)
    with colC:
        st.caption("Scoring: Trend (close>SMA20>SMA50) + Momentum + RSI i “sund zone” (45-65).")

    rows = []
    progress = st.progress(0)
    total = len(uni)

    for i, r in uni.iterrows():
        df = cached_price(r["stooq_symbol"], years=5)
        if df.empty or len(df) < min_rows:
            progress.progress(int((i + 1) / total * 100))
            continue
        dfi = compute_signals(df)
        last = dfi.iloc[-1]
        if pd.isna(last.get("sma20")) or pd.isna(last.get("sma50")) or pd.isna(last.get("rsi14")) or pd.isna(last.get("mom20")):
            progress.progress(int((i + 1) / total * 100))
            continue

        # score
        trend = 1 if (last["close"] > last["sma20"] > last["sma50"]) else 0
        rsi_ok = 1 if (45 <= last["rsi14"] <= 65) else 0
        mom = last["mom20"]
        score = trend * 2 + rsi_ok * 1 + (1 if mom > 0 else 0)

        rows.append({
            "name": r["name"],
            "ticker": r["ticker"],
            "stooq_symbol": r["stooq_symbol"],
            "sector": r.get("sector", ""),
            "country": r.get("country", ""),
            "close": float(last["close"]),
            "rsi14": float(last["rsi14"]),
            "mom20": float(last["mom20"]),
            "score": float(score),
            "label": trade_label(last),
        })
        progress.progress(int((i + 1) / total * 100))

    progress.empty()

    if not rows:
        st.warning("Ingen kandidater fundet (måske pga. ticker coverage). Prøv et andet univers.")
    else:
        df_s = pd.DataFrame(rows)
        if only_with_sector:
            df_s = df_s[df_s["sector"].astype(str).str.len() > 0]

        df_s = df_s.sort_values(["score", "mom20"], ascending=[False, False])
        top10 = df_s.head(10).copy()

        st.dataframe(
            top10[["name", "ticker", "sector", "country", "close", "rsi14", "mom20", "label", "score"]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown(help_text(top10.to_dict(orient="records")))

# ---------- Portefølje ----------
with tab_portfolio:
    st.subheader("Din portefølje (upload CSV)")

    st.caption(
        "Gratis version: upload din beholdning som CSV (fx eksport fra Nordnet). "
        "Automatisk sync kræver broker-API/integration."
    )

    sample = pd.DataFrame({
        "ticker": ["AAPL", "NOVO-B", "MSFT"],
        "name": ["Apple", "Novo Nordisk B", "Microsoft"],
        "shares": [10, 5, 3],
        "avg_price": [120, 900, 250],
        "currency": ["USD", "DKK", "USD"],
        "stooq_symbol": ["aapl.us", "novo-b.dk", "msft.us"],
        "sector": ["Technology", "Healthcare", "Technology"],
        "country": ["US", "DK", "US"],
    })
    with st.expander("Se eksempel på CSV-format"):
        st.dataframe(sample, use_container_width=True, hide_index=True)

    uploaded = st.file_uploader("Upload portefølje CSV", type=["csv"])
    if uploaded:
        try:
            pf = parse_portfolio_csv(uploaded)

            # fetch prices for each row (if stooq_symbol missing, try map from ticker in universes could be done later)
            price_map = {}
            for _, row in pf.iterrows():
                sym = str(row.get("stooq_symbol", "")).strip()
                if not sym:
                    continue
                df = cached_price(sym, years=5)
                if not df.empty:
                    price_map[row["ticker"]] = float(df["close"].iloc[-1])

            pf2 = weight_by_value(pf, price_map)

            st.subheader("Fordeling (%)")
            show_cols = ["ticker", "name", "sector", "country", "shares", "last_price", "value", "weight_pct"]
            for c in show_cols:
                if c not in pf2.columns:
                    pf2[c] = ""
            st.dataframe(pf2[show_cols].sort_values("weight_pct", ascending=False), use_container_width=True, hide_index=True)

            # sector summary
            if "sector" in pf2.columns and pf2["sector"].astype(str).str.len().sum() > 0:
                by_sector = pf2.groupby("sector", dropna=False)["value"].sum().sort_values(ascending=False)
                st.bar_chart(by_sector)

        except Exception as e:
            st.error(f"Kunne ikke læse portefølje: {e}")

# ---------- Temaer ----------
with tab_themes:
    st.subheader("Temaer / mulige fokusområder (idébank)")
    st.write(
        "Dette er **inspiration** til hvilke områder du kan bygge watchlists omkring. "
        "Du kan selv lave en CSV-universe pr. tema og lægge i `data/universes/`."
    )

    themes = {
        "AI & compute": ["Halvledere", "Cloud", "Datacentre", "ML software"],
        "Defense & sikkerhed": ["Aerospace", "Cybersecurity", "Forsvarsleverandører"],
        "Elektrificering": ["Batterier", "Ladning", "Power electronics", "Net-infrastruktur"],
        "Grøn omstilling": ["Vind", "Sol", "Power-to-X", "Energieffektivitet"],
        "Rummet": ["Satellitter", "Opsendelser", "Jordobservation", "Navigation"],
        "Sundhed & bio": ["Medtech", "Pharma", "Diagnostics"],
    }
    for k, v in themes.items():
        with st.expander(k):
            st.write(", ".join(v))

    st.info("Næste step (valgfrit): Jeg kan hjælpe dig med at lave *tema-universer* og en fane der viser deres top 10.")
