from __future__ import annotations

import io
import re
from typing import Iterable

import pandas as pd


ACCOUNT_NUMBER_MAP = {
    "11005683": {
        "account_code": "AKT",
        "account_label": "Månedopsparing - 11005683",
        "account_group": "Konti",
    },
    "11005956": {
        "account_code": "RPP",
        "account_label": "Rate pension - 11005956",
        "account_group": "Pension",
    },
    "47994264": {
        "account_code": "AOP",
        "account_label": "Aldersopsparing - 47994264",
        "account_group": "Pension",
    },
}


DEFAULT_NAME_TO_TICKER_MAP = {
    "UiPath A": "PATH",
    "Cipher Digital": "CIFR",
    "EHang ADR": "EH",
    "Evaxion ADR": "EVAX",
    "Spire Global A": "SPIR",
    "ATHA Energy Corp": "",
    "Definium Therapeutics": "",
    "NanoEcho": "",
    "NanoEcho AB UR": "",
    "NewDeal Invest, kl n": "",
    "iShares AI Infrastructure UCITS ETF USD (Acc)": "",
}


def _dk_to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace(".", "")
    text = text.replace(",", ".")
    text = text.replace("%", "")
    text = text.replace(" ", "")

    try:
        return float(text)
    except ValueError:
        return None


def _extract_account_number(filename: str) -> str:
    match = re.search(r"kontonummer\s+(\d+)", filename, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    digits = re.findall(r"\d{6,}", filename)
    return digits[0] if digits else ""


def _read_depot_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    if hasattr(uploaded_file, "getvalue"):
        content = uploaded_file.getvalue()
    else:
        with open(uploaded_file, "rb") as f:
            content = f.read()

    if not content:
        return pd.DataFrame()

    return pd.read_csv(
        io.BytesIO(content),
        sep="\t",
        encoding="utf-16",
    )


def _map_account_metadata(account_number: str) -> dict:
    if account_number in ACCOUNT_NUMBER_MAP:
        return ACCOUNT_NUMBER_MAP[account_number]

    return {
        "account_code": "UKN",
        "account_label": f"Ukendt depot - {account_number}" if account_number else "Ukendt depot",
        "account_group": "Ukendt",
    }


def _normalize_depot_df(
    raw_df: pd.DataFrame,
    source_name: str,
    name_to_ticker_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    ticker_map = dict(DEFAULT_NAME_TO_TICKER_MAP)
    if name_to_ticker_map:
        ticker_map.update(name_to_ticker_map)

    account_number = _extract_account_number(source_name)
    account_meta = _map_account_metadata(account_number)

    work = raw_df.copy()

    rename_map = {
        "Navn": "Asset Name",
        "Valuta": "Currency",
        "Antal": "Net Shares",
        "GAK/gns. kurs": "Avg Cost",
        "I dag %": "Day Change %",
        "Seneste kurs": "Last Price",
        "Belåningsværdi DKK": "Loan Value DKK",
        "Værdi DKK": "Market Value DKK",
        "Afkast": "Return %",
        "Afkast DKK": "Return DKK",
    }
    work = work.rename(columns=rename_map)

    for col in [
        "Net Shares",
        "Avg Cost",
        "Day Change %",
        "Last Price",
        "Loan Value DKK",
        "Market Value DKK",
        "Return %",
        "Return DKK",
    ]:
        if col in work.columns:
            work[col] = work[col].apply(_dk_to_float)

    if "Asset Name" not in work.columns:
        return pd.DataFrame()

    work["Asset Name"] = work["Asset Name"].astype(str).str.strip()
    work["Ticker"] = work["Asset Name"].map(ticker_map).fillna("").astype(str).str.upper().str.strip()

    work["Account Number"] = account_number
    work["Account Code"] = account_meta["account_code"]
    work["Account Group"] = account_meta["account_group"]
    work["Account"] = account_meta["account_label"]
    work["Source File"] = source_name
    work["Source Type"] = "Depot Overview Upload"

    ordered_cols = [
        "Ticker",
        "Asset Name",
        "Account",
        "Account Number",
        "Account Code",
        "Account Group",
        "Currency",
        "Net Shares",
        "Avg Cost",
        "Last Price",
        "Day Change %",
        "Loan Value DKK",
        "Market Value DKK",
        "Return %",
        "Return DKK",
        "Source File",
        "Source Type",
    ]

    for col in ordered_cols:
        if col not in work.columns:
            work[col] = None

    work = work[ordered_cols].copy()
    work = work[work["Net Shares"].fillna(0) != 0].copy()
    work = work.reset_index(drop=True)

    return work


def load_positions_from_depot_uploads(
    uploaded_files: Iterable,
    name_to_ticker_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, str]:
    if not uploaded_files:
        return pd.DataFrame(), "Ingen depotfiler valgt."

    frames = []
    messages = []

    for file in uploaded_files:
        source_name = getattr(file, "name", "uploaded_file.csv")
        try:
            raw_df = _read_depot_csv(file)
            norm_df = _normalize_depot_df(
                raw_df=raw_df,
                source_name=source_name,
                name_to_ticker_map=name_to_ticker_map,
            )
            if not norm_df.empty:
                frames.append(norm_df)
                messages.append(f"{source_name}: {len(norm_df)} positioner indlæst")
            else:
                messages.append(f"{source_name}: ingen positioner fundet")
        except Exception as exc:
            messages.append(f"{source_name}: fejl ved indlæsning ({exc})")

    if not frames:
        return pd.DataFrame(), " | ".join(messages)

    out = pd.concat(frames, ignore_index=True)

    grouped = (
        out.groupby(
            [
                "Ticker",
                "Asset Name",
                "Account",
                "Account Number",
                "Account Code",
                "Account Group",
                "Currency",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            {
                "Net Shares": "sum",
                "Avg Cost": "mean",
                "Last Price": "mean",
                "Day Change %": "mean",
                "Loan Value DKK": "sum",
                "Market Value DKK": "sum",
                "Return %": "mean",
                "Return DKK": "sum",
                "Source File": "first",
                "Source Type": "first",
            }
        )
    )

    return grouped, " | ".join(messages)


def normalize_uploaded_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Asset Name",
                "Account",
                "Account Number",
                "Account Code",
                "Account Group",
                "Currency",
                "Net Shares",
                "Avg Cost",
                "Last Price",
                "Day Change %",
                "Loan Value DKK",
                "Market Value DKK",
                "Return %",
                "Return DKK",
                "Source File",
                "Source Type",
            ]
        )

    work = df.copy()

    for col in [
        "Net Shares",
        "Avg Cost",
        "Last Price",
        "Day Change %",
        "Loan Value DKK",
        "Market Value DKK",
        "Return %",
        "Return DKK",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "Ticker" in work.columns:
        work["Ticker"] = work["Ticker"].fillna("").astype(str).str.upper().str.strip()

    if "Asset Name" in work.columns:
        work["Asset Name"] = work["Asset Name"].fillna("").astype(str).str.strip()

    if "Account" in work.columns:
        work["Account"] = work["Account"].fillna("").astype(str).str.strip()

    return work.reset_index(drop=True)


def portfolio_positions_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    cols = [
        "Account Code",
        "Account",
        "Ticker",
        "Asset Name",
        "Currency",
        "Net Shares",
        "Avg Cost",
        "Last Price",
        "Market Value DKK",
        "Return %",
        "Return DKK",
    ]

    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()