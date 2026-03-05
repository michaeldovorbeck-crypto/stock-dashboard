# tools/build_universes.py
import os
from io import StringIO
import pandas as pd
import requests

NASDAQ_LISTED = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
OTHER_LISTED  = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"

OUT_DIR = os.path.join("data", "universes")
US_ALL_OUT = os.path.join(OUT_DIR, "us_all.csv")

def download_text(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def parse_pipe_table(txt: str) -> pd.DataFrame:
    """
    Nasdaq Trader filer er pipe-separerede og slutter med en footer-linje.
    Vi fjerner footer og læser med sep='|'.
    """
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 3:
        return pd.DataFrame()

    header = lines[0]
    data_lines = lines[1:]
    # fjern footer "File Creation Time" osv.
    data_lines = [ln for ln in data_lines if not ln.lower().startswith("file creation time")]
    buf = header + "\n" + "\n".join(data_lines)
    return pd.read_csv(StringIO(buf), sep="|", dtype=str).fillna("")

def clean_symbol(sym: str) -> str:
    """
    Nasdaq bruger fx BRK.B, Stooq bruger BRK-B. Vi konverterer . -> -
    """
    s = (sym or "").strip().upper()
    if not s:
        return ""
    return s.replace(".", "-")

def to_stooq_us(sym: str) -> str:
    """
    Stooq US tickers: AAPL.US
    """
    s = clean_symbol(sym)
    if not s:
        return ""
    return f"{s}.US"

def build_us_all(include_etf: bool = True) -> pd.DataFrame:
    t1 = download_text(NASDAQ_LISTED)
    t2 = download_text(OTHER_LISTED)

    df1 = parse_pipe_table(t1)
    df2 = parse_pipe_table(t2)

    # nasdaqlisted.txt: Symbol, Security Name, ETF, ...
    # otherlisted.txt: ACT Symbol, Security Name, ETF, Exchange, ...
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
                rows.append({"ticker": s, "name": name, "sector": "", "country": "US"})

    if not df2.empty and "ACT Symbol" in df2.columns:
        for _, r in df2.iterrows():
            sym = r.get("ACT Symbol", "")
            name = r.get("Security Name", "")
            etf = (r.get("ETF", "") or "").strip().upper() == "Y"
            if (not include_etf) and etf:
                continue
            s = to_stooq_us(sym)
            if s:
                rows.append({"ticker": s, "name": name, "sector": "", "country": "US"})

    out = pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading symbol lists...")
    df = build_us_all(include_etf=True)

    print("Rows:", len(df))
    df.to_csv(US_ALL_OUT, index=False, encoding="utf-8")
    print("Wrote:", US_ALL_OUT)

if __name__ == "__main__":
    main()
