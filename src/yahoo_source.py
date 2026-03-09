from __future__ import annotations

import pandas as pd
import requests

YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
    "Origin": "https://finance.yahoo.com",
    "Connection": "keep-alive",
}


def fetch_yahoo_ohlcv(symbol: str, years: int = 5) -> tuple[pd.DataFrame, str]:
    params = {
        "range": "max" if years >= 10 else f"{max(1, int(years))}y",
        "interval": "1d",
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }

    r = requests.get(
        f"{YAHOO_CHART_BASE}/{symbol}",
        params=params,
        headers=YAHOO_HEADERS,
        timeout=25,
    )

    if r.status_code != 200:
        return pd.DataFrame(), f"HTTP {r.status_code}"

    try:
        js = r.json()
    except Exception:
        preview = (r.text or "")[:180].replace("\n", " ")
        return pd.DataFrame(), f"Invalid JSON | preview={preview}"

    chart = js.get("chart", {})
    error = chart.get("error")
    if error:
        if isinstance(error, dict):
            desc = error.get("description", "Yahoo error")
            return pd.DataFrame(), str(desc)
        return pd.DataFrame(), str(error)

    result = chart.get("result") or []
    if not result:
        return pd.DataFrame(), "No result"

    try:
        block = result[0]
        timestamps = block.get("timestamp") or []
        indicators = block.get("indicators", {})
        quote_list = indicators.get("quote") or []

        if not timestamps or not quote_list:
            return pd.DataFrame(), "No chart data"

        quote = quote_list[0]

        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(timestamps, unit="s", errors="coerce"),
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", []),
            }
        )

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

        if df.empty:
            return pd.DataFrame(), "Empty dataframe"

        return df, "Success"

    except Exception as e:
        return pd.DataFrame(), f"Error: {e}"