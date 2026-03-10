from __future__ import annotations

import io

import pandas as pd


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("%", "")
    text = text.replace(" ", "")

    # håndter både dansk og engelsk talformat
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def normalize_transactions_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "Account",
                "Side",
                "Shares",
                "Price",
                "Fee",
                "Note",
            ]
        )

    work = df.copy()

    rename_map = {}
    col_map = {
        "Date": ["date", "dato", "trade date"],
        "Ticker": ["ticker", "symbol", "papir", "instrument"],
        "Account": ["account", "konto", "depot"],
        "Side": ["side", "type", "buy/sell", "action"],
        "Shares": ["shares", "antal", "quantity", "qty"],
        "Price": ["price", "kurs", "trade price"],
        "Fee": ["fee", "gebyr", "commission", "omkostning"],
        "Note": ["note", "kommentar", "tekst"],
    }

    for target, candidates in col_map.items():
        found = _find_col(work, candidates)
        if found is not None:
            rename_map[found] = target

    work = work.rename(columns=rename_map)

    for col in ["Date", "Ticker", "Account", "Side", "Shares", "Price", "Fee", "Note"]:
        if col not in work.columns:
            work[col] = None

    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["Ticker"] = work["Ticker"].fillna("").astype(str).str.upper().str.strip()
    work["Account"] = work["Account"].fillna("").astype(str).str.strip()
    work["Side"] = work["Side"].fillna("").astype(str).str.upper().str.strip()
    work["Note"] = work["Note"].fillna("").astype(str).str.strip()

    for col in ["Shares", "Price", "Fee"]:
        work[col] = work[col].apply(_to_float)

    work = work[["Date", "Ticker", "Account", "Side", "Shares", "Price", "Fee", "Note"]].copy()
    work = work.dropna(subset=["Date"], how="all").reset_index(drop=True)

    return work


def load_transactions_from_upload(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is None:
        return pd.DataFrame(), "Ingen fil valgt."

    try:
        content = uploaded_file.getvalue()

        # prøv først almindelig csv
        try:
            raw_df = pd.read_csv(io.BytesIO(content))
        except Exception:
            # fallback til semikolon
            try:
                raw_df = pd.read_csv(io.BytesIO(content), sep=";")
            except Exception:
                # fallback til tab
                raw_df = pd.read_csv(io.BytesIO(content), sep="\t")

        clean_df = normalize_transactions_df(raw_df)

        if clean_df.empty:
            return pd.DataFrame(), f"{uploaded_file.name}: ingen gyldige handler fundet."

        return clean_df, f"{uploaded_file.name}: {len(clean_df)} handler indlæst."

    except Exception as exc:
        return pd.DataFrame(), f"{getattr(uploaded_file, 'name', 'upload')}: fejl ved import ({exc})"


def merge_transactions(existing_df: pd.DataFrame, imported_df: pd.DataFrame) -> pd.DataFrame:
    existing = normalize_transactions_df(existing_df)
    imported = normalize_transactions_df(imported_df)

    if existing.empty:
        return imported.reset_index(drop=True)

    if imported.empty:
        return existing.reset_index(drop=True)

    merged = pd.concat([existing, imported], ignore_index=True)

    merged = merged.drop_duplicates(
        subset=["Date", "Ticker", "Account", "Side", "Shares", "Price", "Fee", "Note"],
        keep="first",
    ).reset_index(drop=True)

    merged = merged.sort_values(["Date", "Ticker", "Account"], ascending=[True, True, True]).reset_index(drop=True)
    return merged