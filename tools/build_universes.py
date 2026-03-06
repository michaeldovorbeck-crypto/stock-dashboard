import os
from io import StringIO

import pandas as pd
import requests

NASDAQ_LISTED = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
OTHER_LISTED = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"

OUT_DIR = os.path.join("data", "universes")
US_ALL_OUT = os.path.join(OUT_DIR, "us_all.csv")
GLOBAL_ALL_OUT = os.path.join(OUT_DIR, "global_all.csv")


def download_text(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def parse_pipe_table(txt: str) -> pd.DataFrame:
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 3:
        return pd.DataFrame()

    header = lines[0]
    data_lines = lines[1:]
    data_lines = [ln for ln in data_lines if not ln.lower().startswith("file creation time")]

    buf = header + "\n" + "\n".join(data_lines)
    return pd.read_csv(StringIO(buf), sep="|", dtype=str).fillna("")


def clean_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if not s:
        return ""
    return s.replace(".", "-")


def to_stooq_us(sym: str) -> str:
    s = clean_symbol(sym)
    if not s:
        return ""
    return f"{s}.US"


def build_us_all(include_etf: bool = True) -> pd.DataFrame:
    t1 = download_text(NASDAQ_LISTED)
    t2 = download_text(OTHER_LISTED)

    df1 = parse_pipe_table(t1)
    df2 = parse_pipe_table(t2)

    rows = []

    if not df1.empty and "Symbol" in df1.columns:
        for _, r in df1.iterrows():
            sym = r.get("Symbol", "")
            name = r.get("Security Name", "")
            etf = (r.get("ETF", "") or "").strip().upper() == "Y"

            if (not include_etf) and etf:
                continue

            s = to_stooq_us(sym)
            if s:
                rows.append(
                    {
                        "ticker": s,
                        "name": name,
                        "sector": "",
                        "country": "US",
                    }
                )

    if not df2.empty and "ACT Symbol" in df2.columns:
        for _, r in df2.iterrows():
            sym = r.get("ACT Symbol", "")
            name = r.get("Security Name", "")
            etf = (r.get("ETF", "") or "").strip().upper() == "Y"

            if (not include_etf) and etf:
                continue

            s = to_stooq_us(sym)
            if s:
                rows.append(
                    {
                        "ticker": s,
                        "name": name,
                        "sector": "",
                        "country": "US",
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

    out = out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out


def read_existing_universe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=["ticker", "name", "sector", "country"])

        df.columns = [str(c).strip().lower() for c in df.columns]

        rename_map = {}
        if "symbol" in df.columns:
            rename_map["symbol"] = "ticker"
        if "security name" in df.columns:
            rename_map["security name"] = "name"
        if rename_map:
            df = df.rename(columns=rename_map)

        for col in ["ticker", "name", "sector", "country"]:
            if col not in df.columns:
                df[col] = ""

        df["ticker"] = df["ticker"].astype(str).str.strip()
        df = df[df["ticker"].str.len() > 0]
        return df[["ticker", "name", "sector", "country"]].copy()
    except Exception:
        return pd.DataFrame(columns=["ticker", "name", "sector", "country"])


def build_global_all(us_df: pd.DataFrame) -> pd.DataFrame:
    extra_files = [
        os.path.join(OUT_DIR, "stoxx600.csv"),
        os.path.join(OUT_DIR, "germany_de.csv"),
        os.path.join(OUT_DIR, "nordics_dk.csv"),
        os.path.join(OUT_DIR, "nordics_se.csv"),
        os.path.join(OUT_DIR, "sp500.csv"),
    ]

    frames = [us_df]

    for f in extra_files:
        df = read_existing_universe(f)
        if not df.empty:
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out = out[out["ticker"].str.len() > 0].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building US_ALL...")
    us_all = build_us_all(include_etf=True)
    us_all.to_csv(US_ALL_OUT, index=False, encoding="utf-8")
    print(f"Wrote {len(us_all)} rows -> {US_ALL_OUT}")

    print("Building GLOBAL_ALL...")
    global_all = build_global_all(us_all)
    global_all.to_csv(GLOBAL_ALL_OUT, index=False, encoding="utf-8")
    print(f"Wrote {len(global_all)} rows -> {GLOBAL_ALL_OUT}")


if __name__ == "__main__":
    main()
