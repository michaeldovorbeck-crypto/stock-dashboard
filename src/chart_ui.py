from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def prepare_ohlcv_chart_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardiserer input til chart-visning.
    Forventer kolonner som minimum:
    Date, Open, High, Low, Close
    Volume er valgfri.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    if "Date" not in work.columns:
        return pd.DataFrame()

    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume", "EMA20", "EMA50", "EMA200"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    required = [c for c in ["Date", "Open", "High", "Low", "Close"] if c in work.columns]
    if len(required) < 5:
        return pd.DataFrame()

    work = work.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)
    return work


def render_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Candlestick chart",
    show_volume: bool = True,
    show_ema: bool = True,
    height: int = 650,
) -> None:
    """
    Viser candlestick-chart med optional volume og EMA overlays.
    Falder tilbage til almindelig line chart hvis Plotly ikke er tilgængelig.
    """
    chart_df = prepare_ohlcv_chart_df(df)

    if chart_df.empty:
        st.info("Ingen chart-data tilgængelig.")
        return

    # Fallback hvis plotly ikke er installeret
    if go is None:
        st.warning("Plotly er ikke tilgængelig. Viser fallback line chart.")
        cols = [c for c in ["Close", "EMA20", "EMA50", "EMA200"] if c in chart_df.columns]
        if cols:
            st.line_chart(chart_df.set_index("Date")[cols])
        return

    if show_volume and "Volume" in chart_df.columns and chart_df["Volume"].notna().any():
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=chart_df["Date"],
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="OHLC",
            )
        )

        if show_ema:
            for ema_col in ["EMA20", "EMA50", "EMA200"]:
                if ema_col in chart_df.columns and chart_df[ema_col].notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=chart_df["Date"],
                            y=chart_df[ema_col],
                            mode="lines",
                            name=ema_col,
                        )
                    )

        # Volume på sekundær y-akse
        fig.add_trace(
            go.Bar(
                x=chart_df["Date"],
                y=chart_df["Volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.35,
            )
        )

        fig.update_layout(
            title=title,
            height=height,
            xaxis=dict(rangeslider=dict(visible=False)),
            yaxis=dict(title="Pris"),
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(orientation="h"),
            margin=dict(l=20, r=20, t=50, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)
        return

    # Uden volume
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df["Date"],
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name="OHLC",
        )
    )

    if show_ema:
        for ema_col in ["EMA20", "EMA50", "EMA200"]:
            if ema_col in chart_df.columns and chart_df[ema_col].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=chart_df["Date"],
                        y=chart_df[ema_col],
                        mode="lines",
                        name=ema_col,
                    )
                )

    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(title="Pris"),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_volume_panel(df: pd.DataFrame, title: str = "Volume") -> None:
    """
    Simpel separat volume panel hvis du vil vise volume uden candlestick.
    """
    chart_df = prepare_ohlcv_chart_df(df)
    if chart_df.empty or "Volume" not in chart_df.columns:
        st.info("Ingen volume-data.")
        return

    if go is None:
        st.bar_chart(chart_df.set_index("Date")[["Volume"]])
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["Date"],
            y=chart_df["Volume"],
            name="Volume",
            opacity=0.5,
        )
    )
    fig.update_layout(
        title=title,
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)