# src/storage_engine.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

WATCHLIST_FILE = DATA_DIR / "watchlist.csv"
PORTFOLIO_FILE = DATA_DIR / "portfolio_positions.csv"
RECENT_FILE = DATA_DIR / "recent_assets.csv"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_watchlist(watchlist: list[str]) -> None:
    rows = [{"Ticker": str(x).strip().upper()} for x in watchlist if str(x).strip()]
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(WATCHLIST_FILE, index=False)


def load_watchlist() -> list[str]:
    df = _safe_read_csv(WATCHLIST_FILE)
    if df.empty or "Ticker" not in df.columns:
        return []
    return df["Ticker"].astype(str).str.upper().str.strip().dropna().tolist()


def save_portfolio_positions(positions: list[dict]) -> None:
    df = pd.DataFrame(positions)
    if df.empty:
        if PORTFOLIO_FILE.exists():
            PORTFOLIO_FILE.unlink()
        return
    df.to_csv(PORTFOLIO_FILE, index=False)


def load_portfolio_positions() -> list[dict]:
    df = _safe_read_csv(PORTFOLIO_FILE)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def add_recent_asset(ticker: str, max_items: int = 20) -> None:
    t = str(ticker).strip().upper()
    if not t:
        return

    df = _safe_read_csv(RECENT_FILE)
    rows = []

    if not df.empty and "Ticker" in df.columns:
        existing = df["Ticker"].astype(str).str.upper().str.strip().tolist()
        existing = [x for x in existing if x != t]
        rows.extend([{"Ticker": x} for x in existing])

    rows.insert(0, {"Ticker": t})
    out = pd.DataFrame(rows).head(max_items)
    out.to_csv(RECENT_FILE, index=False)


def load_recent_assets() -> list[str]:
    df = _safe_read_csv(RECENT_FILE)
    if df.empty or "Ticker" not in df.columns:
        return []
    return df["Ticker"].astype(str).str.upper().str.strip().dropna().tolist()