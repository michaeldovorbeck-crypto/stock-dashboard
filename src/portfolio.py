from __future__ import annotations

import pandas as pd


def _norm_col(c: str) -> str:
    return (
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def parse_portfolio_csv(uploaded_file) -> pd.DataFrame:
    """
    Forventer et CSV (fx fra dig selv), typisk med kolonner som:
    - ticker / symbol
    - name (valgfri)
    - shares / antal
    - sector (valgfri)
    - country (valgfri)

    Returnerer en dataframe med mindst: ticker, shares, name, sector, country
    """
    df = pd.read_csv(uploaded_file)

    # normaliser kolonnenavne
    df.columns = [_norm_col(c) for c in df.columns]

    # map mulige navne -> standard
    rename_map = {}
    if "symbol" in df.columns and "ticker" not in df.columns:
        rename_map["symbol"] = "ticker"
    if "antal" in df.columns and "shares" not in df.columns:
        rename_map["antal"] = "shares"
    if "quantity" in df.columns and "shares" not in df.columns:
        rename_map["quantity"] = "shares"
    if "units" in df.columns and "shares" not in df.columns:
        rename_map["units"] = "shares"

    df = df.rename(columns=rename_map)

    # minimum checks
    if "ticker" not in df.columns:
        raise ValueError("CSV mangler kolonnen 'ticker' (eller 'symbol').")
    if "shares" not in df.columns:
        # fallback: hvis brugeren ikke har shares, sæt 1 pr række
        df["shares"] = 1

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)

    # valgfri felter
    if "name" not in df.columns:
        df["name"] = ""
    if "sector" not in df.columns:
        df["sector"] = ""
    if "country" not in df.columns:
        df["country"] = ""

    # fjern tomme tickers og nul shares
    df = df[(df["ticker"] != "") & (df["shares"] > 0)].copy()

    # standard rækkefølge
    cols = ["ticker", "name", "sector", "country", "shares"]
    df = df[[c for c in cols if c in df.columns]]

    return df.reset_index(drop=True)


def weight_by_value(portfolio_df: pd.DataFrame, price_map: dict) -> pd.DataFrame:
    """
    Tilføjer last_price, value og weight_pct baseret på price_map[ticker] = pris
    """
    df = portfolio_df.copy()

    def _price(t: str) -> float:
        try:
            return float(price_map.get(t, 0.0))
        except Exception:
            return 0.0

    df["last_price"] = df["ticker"].apply(_price)
    df["value"] = df["shares"].astype(float) * df["last_price"].astype(float)

    total = float(df["value"].sum()) if "value" in df.columns else 0.0
    if total > 0:
        df["weight_pct"] = (df["value"] / total) * 100.0
    else:
        df["weight_pct"] = 0.0

    return df
