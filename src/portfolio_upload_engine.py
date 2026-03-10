from __future__ import annotations

from io import BytesIO, StringIO

import pandas as pd

from src.portfolio_transactions_engine import normalize_transactions_df


COLUMN_ALIASES = {
    "date": "Date",
    "dato": "Date",
    "trade_date": "Date",
    "booking_date": "Date",
    "order_date": "Date",

    "ticker": "Ticker",
    "symbol": "Ticker",
    "isin_ticker": "Ticker",

    "account": "Account",
    "konto": "Account",
    "depot": "Account",

    "side": "Side",
    "type": "Side",
    "action": "Side",
    "transaction_type": "Side",
    "køb/salg": "Side",
    "koeb/salg": "Side",
    "buy_sell": "Side",

    "shares": "Shares",
    "antal": "Shares",
    "quantity": "Shares",
    "qty": "Shares",

    "price": "Price",
    "kurs": "Price",
    "trade_price": "Price",

    "fee": "Fee",
    "gebyr": "Fee",
    "commission": "Fee",
    "cost": "Fee",

    "note": "Note",
    "comment": "Note",
    "tekst": "Note",
    "description": "Note",
}


SIDE_MAP = {
    "BUY": "BUY",
    "KØB": "BUY",
    "KOEB": "BUY",
    "B": "BUY",

    "SELL": "SELL",
    "SÆLG": "SELL",
    "SAELG": "SELL",
    "S": "SELL",
}


def _clean_colname(name: str) -> str:
    return str(name or "").strip().lower()


def _read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()

    try:
        text = raw.decode("utf-8")
        return pd.read_csv(StringIO(text))
    except Exception:
        pass

    try:
        text = raw.decode("latin-1")
        return pd.read_csv(StringIO(text))
    except Exception:
        pass

    try:
        return pd.read_csv(BytesIO(raw))
    except Exception:
        return pd.DataFrame()


def standardize_uploaded_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Account", "Side", "Shares", "Price", "Fee", "Note"])

    work = df.copy()
    rename_map = {}

    for col in work.columns:
        cleaned = _clean_colname(col)
        if cleaned in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[cleaned]

    work = work.rename(columns=rename_map)

    for col in ["Date", "Ticker", "Account", "Side", "Shares", "Price", "Fee", "Note"]:
        if col not in work.columns:
            work[col] = ""

    work = work[["Date", "Ticker", "Account", "Side", "Shares", "Price", "Fee", "Note"]].copy()

    work["Side"] = (
        work["Side"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(SIDE_MAP)
    )

    return normalize_transactions_df(work)


def load_transactions_from_upload(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is None:
        return pd.DataFrame(), "Ingen fil valgt."

    raw_df = _read_uploaded_csv(uploaded_file)
    if raw_df.empty:
        return pd.DataFrame(), "Kunne ikke læse CSV eller filen var tom."

    tx_df = standardize_uploaded_transactions(raw_df)
    if tx_df.empty:
        return pd.DataFrame(), "Filen blev læst, men ingen gyldige handler kunne udledes."

    return tx_df, f"Import preview klar: {len(tx_df)} gyldige handler."


def merge_transactions(existing_df: pd.DataFrame, imported_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        return normalize_transactions_df(imported_df)

    if imported_df is None or imported_df.empty:
        return normalize_transactions_df(existing_df)

    combined = pd.concat([existing_df, imported_df], ignore_index=True)
    combined = normalize_transactions_df(combined)

    combined = combined.drop_duplicates(
        subset=["Date", "Ticker", "Account", "Side", "Shares", "Price", "Fee", "Note"],
        keep="first",
    ).reset_index(drop=True)

    return combined