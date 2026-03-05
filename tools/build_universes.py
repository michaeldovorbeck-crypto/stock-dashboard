import os
from io import StringIO
import pandas as pd
import requests

NASDAQ_NASDAQLISTED = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
NASDAQ_OTHERLISTED  = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"

def download_text(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text

def parse_pipe(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return pd.DataFrame()
    header = lines[0]
    data_lines = lines[1:]
    data_lines = [ln for ln in data_lines if not ln.lower().startswith("file creation time")]
    buf = header + "\n" + "\n".join(data_lines)
    df = pd.read_csv(StringIO(buf), sep="|", dtype=str).fillna("")
    return df

def clean_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if not s:
        return ""
    return s.replace(".", "-")  # BRK.B -> BRK-B

def build_us_all(include_etf: bool = True) -> pd.DataFrame:
    t1 = download_text(NASDAQ_NASDAQLISTED)
    t2 = download_text(NASDAQ_OTHERLISTED)
    df1 = parse_pipe(t1)
    df2 = parse_pipe(t2)

    rows = []

    # Nasdaq listed
    if not df1.empty and "Symbol" in df1.columns:
        for _, r in df1.iterrows():
            sym = clean_symbol(r.get("Symbol",""))
            if not sym:
                continue
            if (r.get("Test Issue","").strip().upper() == "Y"):
                continue
            etf = (r.get("ETF","").strip().upper() == "Y")
            if (not include_etf) and etf:
                continue
            name = (r.get("Security Name","") or "").strip()
            rows.append({
                "ticker": f"{sym}.US",
                "name": name,
                "sector": "",
                "country": "US",
                "asset_type": "ETF" if etf else "Stock",
                "exchange": "NASDAQ",
            })

    # Other listed (NYSE/AMEX/ARCA etc.)
    act_col = "ACT Symbol" if ("ACT Symbol" in df2.columns) else (df2.columns[0] if not df2.empty else None)
    if act_col:
        for _, r in df2.iterrows():
            sym = clean_symbol(r.get(act_col,""))
            if not sym:
                continue
            if (r.get("Test Issue","").strip().upper() == "Y"):
                continue
            etf = (r.get("ETF","").strip().upper() == "Y")
            if (not include_etf) and etf:
                continue
            name = (r.get("Security Name","") or "").strip()
            exch = (r.get("Exchange","") or "").strip().upper()
            rows.append({
                "ticker": f"{sym}.US",
                "name": name,
                "sector": "",
                "country": "US",
                "asset_type": "ETF" if etf else "Stock",
                "exchange": exch,
            })

    out = pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    out = out[out["ticker"].str.match(r"^[A-Z0-9\-\^]+\.US$")].copy()
    # Standard schema for app:
    out = out[["ticker","name","sector","country","asset_type","exchange"]]
    return out

def main():
    os.makedirs("data/universes", exist_ok=True)
    df = build_us_all(include_etf=True)
    path = "data/universes/us_all.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")

if __name__ == "__main__":
    main()
